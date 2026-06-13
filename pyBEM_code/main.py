import sys
import os
import numpy as np
from numba import set_num_threads
import traceback
import time
from tqdm import tqdm

# Import custom modules
from version import __solver__
from pmx_parser import PMXParser
from solver_core import global_shm_cleanup, promote_to_shm, init_worker, frequency_worker, pre_assembly, pre_mics
# from exporter import PVExporter
from exporter_2 import PVExporter
from utils import get_ram, prepare_geometry, get_zone_data, validate_and_log_zones, resolve_tie_interfaces, compute_tie_projection_matrix, get_global_offsets

# Paralel libraries
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

# limit terminal prints size
np.set_printoptions(threshold=100)  
gc.disable()  # Disable automatic garbage collection

def start_pybem_app():
    print(f"{__solver__}")
    
    # --- 1. COLLECT ARGUMENTS ---
    args = sys.argv[1:] # Skip the script name itself
    # print(args)
    filename = None
    user_ncpus = None # Default is None, so auto-logic can take over

    for arg in args:
        if "=" in arg:
            key, val = arg.split("=", 1)
            if key.lower() == "cpus":
                try:
                    user_ncpus = int(val)
                except ValueError:
                    print(f" ( ! ) Warning: Invalid cpus value '{val}'. Using auto-parallel.")
            else:
                raise RuntimeError(f" ( ! ) ERROR: Invalid parameter '{key}'. Did you mean 'cpus'?")
        else:
            # If it doesn't have an '=', assume it's the filename
            filename = arg.strip()

    # --- 2. FALLBACK TO INTERACTIVE ---
    if not filename:
        filename = input("\n Enter PrePoMax *.inp filename: ").strip()

    if not filename or not os.path.exists(filename):
        print(f"ERROR: File '{filename}' not found.")
        return

    # --- 3. APPLY SETTINGS ---
    if user_ncpus is not None:
        num_workers = user_ncpus
        print(f" ( ! ) User Override: Setting parallel workers to ( {num_workers} )\n")
    else:
        # Leave existing auto-RAM/CPU logic for later; see num_workers AFTER PRE RAM logic
        pass

    try:
        # --- 4. SETUP PARSER & MODEL LOADING ---
        parser = PMXParser(filename)
        parser.load_model()
        damping = parser.damping if parser.damping else {'value': 0.0}
        amps = parser.amplitudes

        # # DEBUG
        # print("\n=== DIAGNOSTIC 1: PARSER RAW MICS ===")
        # print(f"Total entries in parser.mics_elements: {len(parser.mics_elements)}")
        # if parser.mics_elements:
        #     sample_eids = list(parser.mics_elements.keys())[:5]
        #     print(f"Sample MICS element IDs from parser: {sample_eids}")

        # # Check if element_to_zone holds the microphone elements
        # mic_eids_in_zone_map = [eid for eid in parser.mics_elements if eid in parser.element_to_zone]
        # print(f"Number of MICS elements successfully registered in element_to_zone: {len(mic_eids_in_zone_map)}\n\n")
        # # DEBUG_end

        # --- 5. MULTI-ZONE GEOMETRY EXTRACTION ---
        # 5.1 Global Sort (Ensures index maps and arrays match input sequentially)
        sorted_nodes = dict(sorted(parser.nodes.items()))
        sorted_node_ids = list(sorted_nodes.keys())
        sorted_bem_els = dict(sorted(parser.elements.items()))
        sorted_bem_ids = list(sorted_bem_els.keys())
        n_bem_els = len(sorted_bem_els)
        
        # 5.2 Extract separated BEM, MICS data by material zones
        zones_mesh = get_zone_data(parser, sorted_nodes)

        # # DEBUG
        # # print(parser.zone_to_elsets)
        # print(zones_mesh)
        # # DEBUG_end
        
        # 5.3 Count entire microphones across all zones for the memory allocator governor.
        # If there are no mics anywhere, this safely sums up to 0. 
        n_mics_nodes = sum(zone['n_mics'] for zone in zones_mesh.values())
        parser.n_mics_nodes = n_mics_nodes 

        # --- 6. INITIALISE LOG & ZONES DATA ---
        # 6.1 Log file & prints setup
        log_f = f"{parser.model_name}.log"
        model_summary = parser.print_model_summary()
        print(model_summary)
        with open(log_f, "w") as log:
            log.write(__solver__)
            log.write(model_summary)
            log.flush()
        log_top = ''

        # 6.2 Execute Strict Water-Tight Checks & Log Summaries Per BEM Zone
        # This replaces the old single-domain geometry checks and establishes:
        #   - global_h_signs: Dict containing individual zone orientations (Interior -1.0 vs Exterior 1.0)
        #   - global_order_lengths: Dict tracking element lengths for Numba numerical integration bounds
        log_info, global_h_signs, global_order_lengths = validate_and_log_zones(
            zones_mesh, sorted_nodes, parser, log_f, log_top
        )

        # --- 7. SETUP GLOBAL EXPORTER PACKAGE (ParaView) ---
        # 7.1 Map nodal identifiers to a clean 0-indexed flat VTK table array
        nodal_id_map = {inp_id: i for i, inp_id in enumerate(sorted_node_ids)}
        # # DEBUG
        # print(nodal_id_map)
        # # DEBUG_end
        
        # 7.2 Accumulate all localized microphone arrays across zones into a unified export dictionary
        sorted_mics_els = {}
        for zone in zones_mesh.values():
            if zone['mics_elements']:
                sorted_mics_els.update(zone['mics_elements'])
                
        sorted_all_els = {**sorted_bem_els, **sorted_mics_els}
        sorted_all_el_ids = sorted_all_els.keys()
        
        # 7.3 Tag cell indices by identities for ParaView color mapping filters (BEM=1, MICS=2)
        group_ids = {node_id: 1 for node_id in sorted_nodes}
        if sorted_mics_els:
            for conn in sorted_mics_els.values():
                for node_id in conn:
                    group_ids[node_id] = 2
        
        # 7.4 Setup EXPORTER for PV results
        exporter = PVExporter(
            parser.model_name, sorted_nodes, sorted_node_ids, 
            nodal_id_map, sorted_all_els, sorted_all_el_ids, group_ids
        )

        # --- 8. RESOLVE BOUNDARY CONDITIONS MAPPING ---
        bc_map, log_bc_info, surface_to_elements = parser.get_bcs()
        log_info += '\n' + log_bc_info
        print(f"{log_info}")
        with open(log_f, "a") as log:
            log.write(log_info + '\n')
            log.flush()
        
        # ==================================================================
        # --- 9. RESOLVE MULTI-ZONE TIE COUPLINGS ---
        # ==================================================================
        # Append connection discoveries to terminal output and log
        log_tie_info = """
    ==========================
    *** SURFACE TIED PAIRS ***
    ==========================
"""
        # Check if ties were expected globally in the model file
        has_global_ties = bool(getattr(parser, 'ties', None))
        
        if has_global_ties:
            log_tie_info += f"""    Found a total of ( {len(parser.ties)} ) TIED pair constraints.
    [ i ] It is recommended that the slave side has a finer mesh vs the master one.
          If the number of 'mapped pairs' found is less than number of slave elements, 
          you may need to increase the 'Position tolerance' value in the *Tie command.\n\n"""
            # 1. First run the basic search to check for isolation/errors and pull basic tie data
            # Locate matching node identities along touching zone boundaries
            tie_registry = resolve_tie_interfaces(parser, zones_mesh, sorted_nodes, default_tolerance=1e-3)
    
            # 2. Build the continuous projection matrix [W] and get precise unique lists of elements
            W_mapping, master_elements, slave_elements = compute_tie_projection_matrix(tie_registry, zones_mesh, sorted_nodes)
            
            for tie_name, info in tie_registry.items():
                n_el_pairs = len(info['element_pairs'])
                n_zone_input_slave = tie_registry[tie_name]['n_input_slave_els']
                n_zone_input_master = tie_registry[tie_name]['n_input_master_els']
                
                log_tie_info += f"--> TIE: [ {tie_name} ]\n"
                log_tie_info += f"    Input Master Elements: ( {n_zone_input_master} ) | Input Slave Elements: ( {n_zone_input_slave} )\n"
                log_tie_info += f"    Mapping Search Tolerance: ( {info['tolerance_used']} L )\n"
                log_tie_info += f"    Interface Elements on Tied Zones : ( {n_el_pairs} ) mapped pairs\n"
                
                # Double safety: 
                # handle the case where e.g. Tie_1 matched pairs but Tie_2 found 0 nodal pairs
                if n_el_pairs == 0:
                    log_tie_info += f"\n\n    [!] FATAL ERROR: Tie contact group '{tie_name}' failed to pair any elements!\n"
                    with open(log_f, "a") as log:
                        log.write(log_tie_info)
                    raise RuntimeError(f"\n [pyBEM] PRE-PROCESSING FAILED: Tie '{tie_name}' has 0 matched elements. See '{log_f}'")
            if not tie_registry:
                # Fatal Error: The parser had *Tie definitions, but completely failed to pair any nodes
                log_tie_info += "\n" + "!"*60 + "\n"
                log_tie_info += " FATAL ERROR: MODEL TIE PAIR PROBLEM(s) DETECTED\n"
                log_tie_info += "!"*60 + "\n"
                log_tie_info += " [!] CRITICAL: *Tie definitions exist, but 0 node pairs were matched.\n"
                log_tie_info += "     REASON: The acoustic zones are physically disconnected.\n"
                log_tie_info += "     SOLUTION: Increase the 'POSITION TOLERANCE' on your *Tie card in PrePoMax,\n"
                log_tie_info += "               or check for mesh misalignments on TIED pair surfaces.\n"
                log_tie_info += "!"*60 + "\n"
                
                with open(log_f, "a") as log:
                    log.write(log_tie_info)
                raise RuntimeError(f"\n[pyBEM] PRE-PROCESSING FAILED: 0 tie connections matched. Zones are not connected. See '{log_f}'")
        else:
            tie_registry = {}
            W_mapping = {}
            master_elements = []
            slave_elements = []
            log_tie_info += " [ i ] No *Tie constraints active or found in model.\n"
            
        print(log_tie_info)
        # Re-flush everything to LOG
        with open(log_f, "a") as log:
            log.write(log_tie_info)
            log.flush()
        
        # ==================================================================
        # --- 10. MULTI-ZONE MATRIX ALLOCATION & SYSTEM OFFSETS ---
        # ==================================================================
        # Compute the exact memory matrix footprint for the solver
        zone_offsets, total_matrix_size = get_global_offsets(zones_mesh, tie_registry)
        
        # # DEBUG
        # print(zone_offsets)
        # # DEBUG_end
        
        # --- Start Timers & UX Metric Initializations ---
        all_t_avr = 0
        all_t_exp = 0
        global_t0 = time.time()
        global_assy = 0
        global_BEM = 0
        global_mics = 0

        min_freq = min(parser.frequencies)
        max_freq = max(parser.frequencies)
        num_freqs = len(parser.frequencies)
        del_freq = parser.frequencies[1] - parser.frequencies[0] if num_freqs > 1 else 0.0

        # Set initial hardware thread counts for the Numba compile phase
        set_num_threads(n_CPUs)  
        
        str_CPUs = (f"""
 Number of CPUs found on this machine: ( {n_CPUs} ) |  Number of threads: ( {n_threads} )
 MAX number of CPUs assigned to Numba parallel loops in PRE is: ( {used_CPUs} )
 Available RAM found at job start:     ( {RAM_gb:.2f} GB )
 Global Multi-Zone Matrix Structure:   ( {total_matrix_size} x {total_matrix_size} ) DOF

 [ i ] First PRE Assembly (and compile) of [G] & [H] matrices takes longer. 
       Hold tight, it gets faster after, see times per Freq table in '{parser.model_name}.log'.
""")
        for z_name, alloc in zone_offsets.items():
            str_CPUs += f" --> Zone [ {z_name:<12} ]: Matrix Index Range [{alloc['start_idx']:>5} -> {alloc['start_idx'] + alloc['n_elements'] - 1:<5}] ( {alloc['n_elements']} elements )\n"

        print(f"{'=' * 80}")
        print(f"    ACOUSTICS multi-zone job started at:  {time.ctime()}")
        print(f"{'=' * 80}")
        print(f" -- Solving {num_freqs} Frequencies [{min_freq:.1f}Hz --> {max_freq:.1f}Hz | delta_Hz = {del_freq:.2f}] (Steady State Direct) -- \n{str_CPUs}")
        
        with open(log_f, "a") as log:
            log.write(f"\n{'=' * 98}\n    ACOUSTICS multi-zone job started at: {time.ctime()}\n{'=' * 98}")
            log.write(f"\n -- Solving {num_freqs} Frequencies [{min_freq:.1f}Hz --> {max_freq:.1f}Hz | delta_Hz = {del_freq:.2f}] -- \n")
            log.write(str_CPUs)
            log.flush()

        # ==================================================================
        # --- 11. LOCAL GEOMETRIC PRE-ASSEMBLY PASS (PER ZONE) ---
        # ==================================================================
        t_pre_start = time.time()
        t_pre_assy = 0
        t_pre_mics = 0
        pre_bem_data = {}
        pre_mics_data = {}
        # --- INITIALIZE UNIFIED GLOBAL TRACKING STRUCTURES ---
        global_eid_to_col = {}
        global_bem_areas = {}
        global_bem_normals = {}
        
        global_nid_to_mic_col = {}
        global_mics_areas = {}
        global_mics_normals = {}
        global_mics_elements_conn = {}
        mic_node_counter = 0
    
        for zone_name, z_mesh in zones_mesh.items():
            t_pre_0 = time.time()
            print(f"\n---> Pre-assembling Geometric Static Kernels for Zone: [ {zone_name} ]")
            
            # 11.1 Extract BEM geometry data local to this zone
            z_nodes, z_centers, z_areas, z_normals, _, _ = prepare_geometry(sorted_nodes, z_mesh['elements'])
            
            # 11.2 Run pre_assembly function per BEM Zone
            z_gp, z_gp_start, z_R, z_G_stat, z_H_stat, z_g_diag, z_h_diag = pre_assembly(
                z_nodes, z_centers, z_areas, z_normals
            )
            
            # Save local results cleanly into the zone storage dictionary
            pre_bem_data[zone_name] = {
                'gp_per_element': z_gp,
                'GP_start_idx': z_gp_start,
                'R_map': z_R,
                'G_static_map': z_G_stat,
                'H_static_map': z_H_stat,
                'G_diag_static': z_g_diag,
                'H_diag_static': z_h_diag,
                'centers': z_centers,
                'normals': z_normals,
                'areas': z_areas
            }

            # --- POPULATE GLOBAL BEM MAPS ---
            alloc = zone_offsets[zone_name]
            for local_idx, eid in enumerate(z_mesh['elements'].keys()):
                g_idx = alloc['start_idx'] + local_idx
                global_eid_to_col[eid] = g_idx
                global_bem_areas[eid] = z_areas[local_idx]
                global_bem_normals[eid] = z_normals[local_idx]
            
            t_pre_1 = time.time()
            t_pre_assy += t_pre_1 - t_pre_0
    
            # 11.3 Process Optional Microphones within this zone
            if z_mesh['n_mics'] > 0:
                print(f"     Pre-calculating microphone distances for Zone: [ {zone_name} ]")
                z_mics_nodes, z_mics_centers, z_mics_areas, z_mics_normals, _, _ = prepare_geometry(sorted_nodes, z_mesh['mics_elements'])

                pm_G, pm_H, pm_R, pre_mics_dx, pre_mics_dy, pre_mics_dz, n_mics = pre_mics(z_mesh['mics_nodes'], z_centers, z_normals)
                mics_nodes_dict = z_mesh.get('mics_nodes_dict', {}) if z_mesh.get('mics_nodes_dict') else z_mesh.get('mics_nodes', {})
                
                pre_mics_data[zone_name] = {
                    'pre_mics_G': pm_G,
                    'pre_mics_H': pm_H,
                    'pre_mics_R': pm_R,
                    'pre_mics_dx': pre_mics_dx,
                    'pre_mics_dy': pre_mics_dy,
                    'pre_mics_dz': pre_mics_dz,
                    'num_mics': n_mics,
                    'mics_nodes': mics_nodes_dict,
                    'mics_elements': z_mesh['mics_elements']
                }

                # --- POPULATE GLOBAL MICROPHONE NODAL & ELEMENTAL MAPS ---
                for nid in mics_nodes_dict.keys():
                    global_nid_to_mic_col[nid] = mic_node_counter
                    mic_node_counter += 1
                    
                for local_idx, meid in enumerate(z_mesh['mics_elements'].keys()):
                    global_mics_areas[meid] = z_mics_areas[local_idx]
                    global_mics_normals[meid] = z_mics_normals[local_idx]
                    global_mics_elements_conn[meid] = z_mesh['mics_elements'][meid]
            
            t_pre_2 = time.time()
            t_pre_mics += t_pre_2 - t_pre_1
        # Capture exact structural shapes for array initialization
        n_bem_els = len(sorted_bem_ids)
        n_mics_nodes = mic_node_counter
        t_pre = time.time() - t_pre_start
        
        # ==================================================================
        # --- 12. PACK MULTI-ZONE DATA FOR PARALLEL WORKERS ---
        # ==================================================================
        # Build dictionaries holding the explicit material props for each zone
        global_c = {}
        global_rho = {}
        for zone_name, z_mesh in zones_mesh.items():
            global_c[zone_name] = parser.materials[zone_name]['c']
            global_rho[zone_name] = parser.materials[zone_name]['density']

        # Pack up all static data cleanly for the shared container in solve
        static_data = {
            # Map the per-zone c & rho here
            'global_c': global_c, 
            'global_rho': global_rho,
            'global_h_signs': global_h_signs, 
            'global_order_lengths': global_order_lengths,
            'damping': damping,
            'amplitudes': amps,
            'pre_bem_data': pre_bem_data,
            'pre_mics_data': pre_mics_data,
            'sorted_nodes': sorted_nodes,
            'nodal_id_map': nodal_id_map,
            'sorted_bem_els': sorted_bem_els,
            'sorted_bem_ids': sorted_bem_ids,
            'zones_mesh': zones_mesh,
            'zone_offsets': zone_offsets,
            'total_matrix_size': total_matrix_size,
            'tie_registry': tie_registry,
            'W_mapping': W_mapping,  
            'master_elements': master_elements, 
            'slave_elements': slave_elements,
            # --- INJECT THE FLAT GLOBAL MAPS AND SIZES ---
            'global_eid_to_col': global_eid_to_col,
            'global_nid_to_mic_col': global_nid_to_mic_col,
            'global_bem_areas': global_bem_areas,
            'global_bem_normals': global_bem_normals,
            'global_mics_areas': global_mics_areas,
            'global_mics_normals': global_mics_normals,
            'global_mics_elements_conn': global_mics_elements_conn,
            'n_bem_els': n_bem_els,
            'n_mics_nodes': n_mics_nodes
        }

        # # DEBUG
        # # print(static_data['pre_bem_data'])
        # # print(static_data['pre_mics_data'])
        # print(static_data['zones_mesh'])
        # # print(static_data['pre_mics_data'].keys())
        # # print("\n=== DIAGNOSTIC 3: STATIC PACKING FOR WORKERS ===")
        # # for z_name, z_data in pre_mics_data.items():
        # #     print(f"Zone: [ {z_name} ] in pre_mics_data:")
        # #     print(f"  -> num_mics scalar: {z_data.get('num_mics')}")
        # #     if 'pre_mics_G' in z_data and z_data['pre_mics_G'] is not None:
        # #         print(f"  -> pre_mics_G matrix shape: {z_data['pre_mics_G'].shape}\n")
        # # DEBUG_end
        
        # --- Dynamic Resource Governor Allocation ---
        # Here we try to auto-balance: RAM || py workers || n_CPUS for parallel solve/loops 
        # Start costing workers for parallel freq solve based on n_CPUS & RAM estimates at PRE
        # We can actually estimate the +RAM at solve based on BEM n_els & MICS n_mics_nodes
        # We add 25% on top for overheads in py & windows, etc as per testing to date.
        pre_RAM_gb = get_ram() / 1024.0
        cost_per_worker_gb = (((total_matrix_size**2 * 16 * 2.5) + (n_bem_els * n_mics_nodes * 32)) * 1.25 / 1024**3) + 0.5
        safe_RAM_limit_gb = max(0.5, RAM_gb - pre_RAM_gb)
        n_potential_workers = int(safe_RAM_limit_gb // cost_per_worker_gb)
        
        if user_ncpus is not None:
            num_workers = user_ncpus
        else:
            # we force 1 CPU (sequential) solve, it seems fast enough vs parallel freqs; 
            # based on loads of testing to date on a single PC.
            num_workers = 1

            # OLD code for auto-parallel workers (Freqs) experiments.
            # num_workers = max(1, min(n_CPUs - 1, n_potential_workers)) if n_potential_workers > 1 else 1
        
        # Determine Numba threads per worker based on Physical Cores
        threads_per_worker = max(1, n_CPUs // num_workers)
        set_num_threads(threads_per_worker)
        
        log_pre_stats = (f"""
 PRE Assembly (and compile) of multi-zone [G] & [H] matrices took: ( {t_pre:.2f}s )
     BEM: ( {t_pre_assy:.3f}s ) | MICS: ( {t_pre_mics:.3f}s ) | RAM ( {pre_RAM_gb:.3f}GB )
     
 Heuristic estimates for Frequency Sweep based on RAM available ( {RAM_gb:.2f}GB ):
 Estimated (+)RAM per Freq: ( {cost_per_worker_gb:.2f} GB ) | Parallel Frequency Workers: ( {num_workers} )
 [ i ] To avoid race conditions in parallel sums, Numba MAX CPUs is set to ( {threads_per_worker} )
       Numpy solve [np.linalg.solve(A, B)] calls into LAPACK (Intel MKL or OpenBLAS), 
       which already does parallel solving. MAX CPUs for Numpy is set to ( {n_CPUs} )
""")
        print(log_pre_stats)
        print("       [ i ] Promoting heavy arrays to Shared Memory...\n")
        shm_static_data = promote_to_shm(static_data)

        with open(log_f, "a") as log:
            log.write(log_pre_stats)
            log.write(f"       [ i ] Promoting heavy arrays to Shared Memory...\n")
            log.write(f"\n{'=' * 98}")
            log.write(f"\n {'Freq (Hz)':<9} | {'Assembly':^8} | {'Solve All':>9}: {'BEM':^8} + {'Mics':^8} | {'RAM (MB)':^10} | {'Results file':<18} | {'Status':^6}")
            log.write(f"\n{'=' * 98}\n")
            log.flush()

        # ==================================================================
        # --- 13. PRE-ALLOCATE GLOBAL SOLVER RESULTS FOR SOUND POWER ---
        # ==================================================================
        # Create an ordered frequency-to-index dictionary map to handle out-of-order pool resolution
        freq_to_idx_map = {freq: idx for idx, freq in enumerate(parser.frequencies)}
        
        # Allocate master global matrices for raw frequency-domain fields
        global_p_surf = np.zeros((num_freqs, n_bem_els), dtype=np.complex128)
        global_v_surf = np.zeros((num_freqs, n_bem_els), dtype=np.complex128)
        
        global_p_mics = np.zeros((num_freqs, n_mics_nodes), dtype=np.complex128)
        global_v_mics_x = np.zeros((num_freqs, n_mics_nodes), dtype=np.complex128)
        global_v_mics_y = np.zeros((num_freqs, n_mics_nodes), dtype=np.complex128)
        global_v_mics_z = np.zeros((num_freqs, n_mics_nodes), dtype=np.complex128)
        
        # ==================================================================
        # --- 13. THE PARALLEL SWEEP POOL INTERFACE ---
        # ==================================================================
        rslt_f = 'In Memory'  
        print(f"{'=' * 80}")
        pbar = tqdm(total=num_freqs, desc=" Done", ncols=80, unit="Freq", colour="#ddcd3e")
        
        try:
            with ProcessPoolExecutor(
                max_workers = num_workers,
                initializer = init_worker,
                initargs = (shm_static_data, threads_per_worker)
            ) as executor:
                
                # Submit all multi-zone frequency calculations to the pool
                futures = {
                    executor.submit(frequency_worker, f, bc_map, sorted_bem_ids, threads_per_worker): f 
                    for f in parser.frequencies
                }

                for future in as_completed(futures):
                    # Unpack the frequency step results returned by the worker
                    f_done, nodal_pressures, meta = future.result()
                    
                    # Get the correct sorted row sequence index for this frequency
                    f_matrix_idx = freq_to_idx_map[f_done]
                    
                    # --- STORE THE UN-AVERAGED FIELDS DIRECTLY INTO GLOBAL MATRICES ---
                    global_p_surf[f_matrix_idx, :] = meta['p_surf']
                    global_v_surf[f_matrix_idx, :] = meta['v_surf']
                    
                    if n_mics_nodes > 0 and meta['p_mics'] is not None:
                        global_p_mics[f_matrix_idx, :] = meta['p_mics']
                        global_v_mics_x[f_matrix_idx, :] = meta['v_mics_x']
                        global_v_mics_y[f_matrix_idx, :] = meta['v_mics_y']
                        global_v_mics_z[f_matrix_idx, :] = meta['v_mics_z']
                    
                    # 13.1 Update precise timing and benchmark info
                    t_assembly = meta['t_assembly']
                    global_assy += t_assembly
                    
                    t_solve_bem = meta['t_solve_bem']
                    global_BEM += t_solve_bem
                    
                    t_solve_mics = meta['t_solve_mics']
                    t_solve = t_solve_bem + t_solve_mics
                    global_mics += t_solve_mics
                    
                    all_t_avr += meta['t_avrg_nodes']
                    solve_RAM = meta['solve_RAM']
                    
                    # 13.2 Pass results to ParaView exporter in RAM
                    exporter.add_frequency_step(f_done, nodal_pressures)
                    
                    # 13.3 Update UI progress bars
                    pbar.update(1)
                    pbar.set_postfix({"Freq": f"{f_done:.1f}Hz"})
                    # solve_RAM_2 = get_ram()  # Monitor local parent RAM step changes
                    
                    # 13.4 Write to log using your exact scaling layout format
                    log_line = (f" {f_done:<7.1f}Hz | {t_assembly/num_workers:^7.3f}s | {t_solve/num_workers:>7.3f}s : {t_solve_bem/num_workers:^8.3f} + {t_solve_mics/num_workers:^8.3f} | {solve_RAM:^10.1f} | {rslt_f:<18} | {'OK':^6}\n")
                    
                    with open(log_f, "a") as log:
                        log.write(log_line)
                        log.flush()  # Forces real-time tail tracking updates on disk
                        
        finally:
            # Safely releases shared memory allocations regardless of success or sudden crash
            global_shm_cleanup()
        
        pbar.close()  # Close terminal progress visualization cleanly
        t_exp_0 = time.time()
        print(f"{'=' * 80}")
        
        # 13.5 Finalize and write complete VTU outputs to disk
        exporter.finalise()

        # ==================================================================
        # --- 13.6 COMPUTE AND EXPORT TOTAL SURFACE SOUND POWER ---
        # ==================================================================
        if len(parser.surfaces) > 0:
            from utils import calculate_total_sound_power, generate_power_flux_plot
            calculate_total_sound_power(
                model_name = parser.model_name,
                surfaces = parser.surfaces,
                surface_elements = surface_to_elements,
                freqs = parser.frequencies,
                global_p_surf = global_p_surf,
                global_v_surf = global_v_surf,
                global_p_mics = global_p_mics,
                global_v_mics_x = global_v_mics_x,
                global_v_mics_y = global_v_mics_y,
                global_v_mics_z = global_v_mics_z,
                global_bem_elements_map = static_data['global_eid_to_col'],
                global_mics_nodes_map = static_data['global_nid_to_mic_col'],
                global_bem_areas = static_data['global_bem_areas'],
                global_mics_areas = static_data['global_mics_areas'],
                global_mics_normals = static_data['global_mics_normals'],
                global_mics_elements_conn = static_data['global_mics_elements_conn']
            )

            # Trigger the headless plot generation right after the CSV writes out
            generate_power_flux_plot(model_name = parser.model_name)

        t_exp_1 = time.time()
        all_t_exp += t_exp_1 - t_exp_0
        # --- Final Timing Summary Calculations --- 
        total_elapsed = time.time() - global_t0
        avg_time = total_elapsed / num_freqs
        avg_assy = (t_pre_assy + global_assy) / num_freqs
        avg_BEM = global_BEM / num_freqs
        avg_mics = (t_pre_mics + global_mics) / num_freqs
        avg_to_nodes = all_t_avr / num_freqs
        avg_vtu = all_t_exp / num_freqs
        avg_export = avg_to_nodes + avg_vtu

        summary_log = f"""
    [ i ] Shared Memory released.

 -- All Multi-Zone Frequency Steps Completed Successfully --
    Function 'averaged_at_nodes' took: ( {all_t_avr:.2f}s )
    Export Write and VTU Processing:   ( {all_t_exp:.2f}s )
{'-' * 80}"""
        print(summary_log)
        with open(log_f, "a") as log:
            log.write(summary_log)
            log.flush()

            def SUMMARY(): return f"""
    ==========================
    *** SIMULATION SUMMARY ***
    ==========================
    Simulation Finished at: {time.ctime()}
    Total Elapsed Time:     {total_elapsed:.2f} seconds ( {total_elapsed/60:.2f} minutes )
    Total Frequencies:      {num_freqs}
    Avg Time per Freq:      {avg_time:.3f} seconds/Freq: 
                            Assy ( {avg_assy:.3f}s ) + BEM ( {avg_BEM:.3f}s ) + Mics ( {avg_mics:.3f}s ) + Export ( {avg_export:.3f}s )

    Check '{log_f}' for more details.
    Open '{parser.model_name}_Results.pvd' in ParaView.
{"=" * 98}
"""
            print(SUMMARY())
            log.write("\n"+SUMMARY())
    except ValueError as e:
        print(f"\n ERROR loading model: [FATAL INPUT ERROR] {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()

if __name__ == "__main__":
    from utils import get_cpus, set_hardware_limits
    # Get number of physical CPUs to pass onto Numpy libraries for the solve
    # This is required before any `import numpy`
    n_CPUs, n_threads, RAM_gb = get_cpus()
    used_CPUs = n_CPUs
    # This makes sure at the start that all solve libraries are set to a CPU max.
    # This is to minimise race conditions on multi-threading.
    set_hardware_limits(used_CPUs)
    # AFTER, in the code we do try better with Numba, as it has the function
    # set_num_threads(), which the other libraries don't.
    start_pybem_app()
