import sys
import os
import gc
import time
from tqdm import tqdm
import numpy as np
from numba import set_num_threads
from concurrent.futures import ProcessPoolExecutor, as_completed

# Version & Core Imports
from version import __solver__
from pmx_parser import PMXParser
from solver_core import (
    global_shm_cleanup, promote_to_shm, init_worker, 
    frequency_worker, pre_assembly, pre_mics
)
from exporter_2 import PVExporter
from utils import (
    get_cpus, set_hardware_limits, get_ram, prepare_geometry, 
    get_zone_data, validate_and_log_zones, resolve_tie_interfaces, 
    compute_tie_area_weights, get_global_offsets, format_per_tie_mortar_weights,
    setup_logger
)

# Configuration
np.set_printoptions(threshold=100) # limit terminal prints size
gc.disable()  # Disable automatic garbage collection


# MAIN APP
def start_pybem_app(n_CPUs, used_CPUs, n_threads, RAM_gb):
    # --- 1. COLLECT ARGUMENTS ---
    args = sys.argv[1:] # Skip the script name itself
    filename = None
    user_ncpus = None # Default is None, so auto-logic can take over
    debug_mode = False

    for arg in args:
        if "=" in arg:
            key, val = arg.split("=", 1)
            clean_key = key.lstrip("-").lower()
            
            if clean_key == "cpus":
                try:
                    user_ncpus = int(val)
                except ValueError:
                    print(f" [!] Warning: Invalid cpus value '{val}'. Using auto-parallel.")
            elif clean_key == "debug":
                debug_mode = val.lower() in ("true", "1", "yes")
            else:
                raise RuntimeError(f" [!] ERROR: Unknown parameter '{key}'. Valid options are 'cpus=N' or 'debug=yes'.")
        elif arg.lower() in ("--debug", "-debug"):
            debug_mode = True
        else:
            # If it doesn't have an '=', treat as input file
            filename = arg.strip()

    # --- 2. FALLBACK TO INTERACTIVE ---
    if not filename:
        filename = input("\n Enter PrePoMax *.inp filename: ").strip()

    if not filename or not os.path.exists(filename):
        print(f" [!] ERROR: File '{filename}' not found.")
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

        # 4.2 Log file & prints setup (Initializes directly into model_name.log)
        log_f = f"{parser.model_name}.log"
        log_f_debug = f"{parser.model_name}_debug.log"

        for f in [log_f, log_f_debug]:
            if os.path.exists(f):
                os.remove(f)
        logger, file_logger = setup_logger(log_f, debug_mode=debug_mode)

        # --- 5. MULTI-ZONE GEOMETRY EXTRACTION ---
        # 5.1 Global Sort (Ensures index maps and arrays match input sequentially)
        sorted_nodes = dict(sorted(parser.nodes.items()))
        sorted_node_ids = list(sorted_nodes.keys())
        sorted_bem_els = dict(sorted(parser.elements.items()))
        sorted_bem_ids = list(sorted_bem_els.keys())
        n_bem_els = len(sorted_bem_els)
        
        # 5.2 Extract separated BEM, MICS data by material zones
        zones_mesh = get_zone_data(parser, sorted_nodes)
        
        # 5.3 Count entire microphones across all zones for the memory allocator governor.
        # If there are no mics anywhere, this safely sums up to 0. 
        n_mics_nodes = sum(zone['n_mics'] for zone in zones_mesh.values())
        parser.n_mics_nodes = n_mics_nodes 

        # 5.4 Write solver banner and model summary to log & terminal
        logger.info(__solver__.strip())
        logger.info(parser.print_model_summary())

        if debug_mode:
            logger.info("-" * 80)
            logger.info(f"[DEBUG MODE ACTIVATED] Writing some diagnostics to file: '{parser.model_name}_debug.log'")
            logger.info("-" * 80)

        # DEBUG
        logger.debug("\nDEBUG: === PARSER RAW MICS ===")
        logger.debug(f"Total entries in parser.mics_elements: {len(parser.mics_elements)}")
        if parser.mics_elements:
            sample_eids = list(parser.mics_elements.keys())[:5]
            logger.debug(f"Sample MICS element IDs from parser: {sample_eids}")

        # Check if element_to_zone holds the microphone elements
        mic_eids_in_zone_map = [eid for eid in parser.mics_elements if eid in parser.element_to_zone]
        logger.debug(f"Number of MICS elements successfully registered in element_to_zone: {len(mic_eids_in_zone_map)}\n")
        logger.debug(zones_mesh)
        # DEBUG_end

        log_top = ''

        # --- 6. ZONES DATA CHECKS ---
        # 6.1 Execute Strict Water-Tight Checks & Log Summaries Per BEM Zone
        # This replaces the old single-domain geometry checks and establishes:
        #   - global_h_signs: Dict containing individual zone orientations (Interior -1.0 vs Exterior 1.0)
        #   - global_order_lengths: Dict tracking element lengths for Numba numerical integration bounds
        log_zones_info, global_h_signs, global_order_lengths = validate_and_log_zones(
            zones_mesh, sorted_nodes, parser, log_f, log_top
        )

        # --- 7. SETUP GLOBAL EXPORTER PACKAGE (ParaView) ---
        # 7.1 Map nodal identifiers to a clean 0-indexed flat VTK table array
        nodal_id_map = {inp_id: i for i, inp_id in enumerate(sorted_node_ids)}

        # DEBUG
        logger.debug("\nDEBUG: === nodal input IDs --> index map ===")
        logger.debug(nodal_id_map)
        # DEBUG_end
        
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
        logger.info(f"{log_zones_info}")
        logger.info(f"{log_bc_info}")
        if len(bc_map) == 0:
            raise ValueError(f" [!] ERROR: Cannot find any BCs on any element.\n     Please check your '{parser.model_name}.inp' file.\n")

        # ==================================================================
        # --- 9. RESOLVE MULTI-ZONE TIED PAIRS ---
        # ==================================================================
        log_tie_info = """
    ==========================
    *** SURFACE TIED PAIRS ***
    ==========================
"""
        has_global_ties = bool(getattr(parser, 'ties', None))
        
        if has_global_ties:
            log_tie_info += f"""    Found a total of ( {len(parser.ties)} ) TIED pair constraint(s).
    [i] It is recommended equal mesh, or that the slave side has a coarser mesh vs the master one.
        This is to ensure stable area-weighted polygon clipping and mortar flux integration 
        across overlapping patches. However, this is automatically handled by pyBEM during PRE, 
        which shows as '[ Auto-Swap ]'.\n\n"""
            
            def indent_text(text, prefix="    "):
                return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())

            # 1. Resolve matching interface elements
            tie_registry = resolve_tie_interfaces(parser, zones_mesh, sorted_nodes, default_tolerance=1e-3)
    
            if not tie_registry:
                log_tie_info += f"""
    [!] ERROR: MODEL TIE PAIR PROBLEM(s) DETECTED
        *Tie definitions exist, but 0 element pairs were matched.
"""
                logger.error(log_tie_info)
                raise RuntimeError(f" [!] PRE-PROCESSING FAILED: 0 tie connections found.\n")

            # 2. Build continuous projection weights & geometry diagnostic logs
            W_slave_to_master, W_master_to_slave, master_elements, slave_elements, log_pre_ties = compute_tie_area_weights(tie_registry, zones_mesh, sorted_nodes)

            # Append interface areas
            log_tie_info += indent_text(log_pre_ties) + "\n"

            # 3. Format mortar weights grouped per *TIE definition
            weights_ties = format_per_tie_mortar_weights(tie_registry, W_slave_to_master, W_master_to_slave)
            log_tie_info += indent_text(weights_ties) + "\n"

            # 4. Build individual tie surface summaries
            for tie_name, info in tie_registry.items():
                tie_slave_eids = set(info['active_slave_eids'])
                n_el_pairs = sum(
                    len(master_weights) 
                    for s_eid, master_weights in W_slave_to_master.items()
                    if s_eid in tie_slave_eids
                )
                
                m_surf_name = info['master_surface']
                s_surf_name = info['slave_surface']
                n_zone_input_slave = info['n_input_slave_els']
                n_zone_input_master = info['n_input_master_els']
                n_active_slave = len(info['active_slave_eids'])
                n_active_master = len(info['active_master_eids'])
                
                log_tie_info += f"\n    --> TIE: [ {tie_name} ]\n"
                if info.get('is_swapped'):
                    log_tie_info += f"        [ Auto-Swap ] Master/Slave roles inverted: Slave designated as the coarser mesh surface; i.e. higher 'h_avg'.\n"
    
                log_tie_info += f"        Master Surface '{m_surf_name}': {n_zone_input_master} total elements | Active Sub-Patch: {n_active_master} elements (h_avg = {info['h_master']:.4f})\n"
                log_tie_info += f"        Slave Surface '{s_surf_name}': {n_zone_input_slave} total elements | Active Sub-Patch: {n_active_slave} elements (h_avg = {info['h_slave']:.4f})\n"
                log_tie_info += f"        Mapping Search Tolerance: {info['tolerance_used']} L\n"
                log_tie_info += f"        Area-Weighted Collocation Mapped Pairs: {n_el_pairs} element intersections\n"
                
                if n_el_pairs == 0:
                    log_tie_info += f"\n [!] ERROR: Tie contact group '{tie_name}' failed to pair any elements!"
                    logger.error(log_tie_info)
                    raise RuntimeError(f" [!] PRE-PROCESSING FAILED: Tie '{tie_name}' has 0 matched element intersections.")

        else:
            tie_registry = {}
            W_slave_to_master = {}
            W_master_to_slave = {}
            master_elements = []
            slave_elements = []
            log_tie_info += "\n    [i] No *Tie constraints active or found in model.\n"
            
        logger.info(log_tie_info)

        # ==================================================================
        # --- 10. MULTI-ZONE MATRIX ALLOCATION & SYSTEM OFFSETS ---
        # ==================================================================
        # Compute the exact memory matrix footprint for the solver
        zone_offsets, total_matrix_size = get_global_offsets(zones_mesh, tie_registry)
        
        # DEBUG
        logger.debug(f"DEBUG: zone_offsets")
        logger.debug(zone_offsets)
        # DEBUG_end
        
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
        
        log_CPUs = (f"""
 Number of CPUs found on this machine: ( {n_CPUs} ) |  Number of threads: ( {n_threads} )
 MAX number of CPUs assigned to Numba parallel loops in PRE is: ( {used_CPUs} )
 Available RAM found at job start:     ( {RAM_gb:.2f} GB )
 Global Multi-Zone Matrix Structure:   ( {total_matrix_size} x {total_matrix_size} ) DOF

 [i] First PRE Assembly (and compile) of [G] & [H] matrices takes longer. 
     Hold tight, it gets faster after, see times per Freq table in '{parser.model_name}.log'.
""")
        for z_name, alloc in zone_offsets.items():
            log_CPUs += f" --> Zone [ {z_name:<12} ]: Matrix Index Range [{alloc['start_idx']:>5} -> {alloc['start_idx'] + alloc['n_elements'] - 1:<5}] ( {alloc['n_elements']} elements )\n"

        logger.info(f"\n{'=' * 70}")
        logger.info(f"*** ACOUSTICS multi-zone job started at:  {time.ctime()} ***")
        logger.info(f"{'=' * 70}")
        logger.info(f"\n--> SOLVING {num_freqs} Frequencies [{min_freq:.1f}Hz --> {max_freq:.1f}Hz | delta_Hz = {del_freq:.2f}] (Steady State Direct) <--")
        logger.info(log_CPUs)

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
            logger.info(f" --> Pre-assembling BEM Geometric Static Kernels for Zone: [ {zone_name} ]")
            
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
                logger.info(f"     Pre-calculating MICS distances for Zone: [ {zone_name} ]")
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
            'W_slave_to_master': W_slave_to_master,  
            'W_master_to_slave': W_master_to_slave,
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
 [i] PRE Assembly (and compile) of multi-zone [G] & [H] matrices took: ( {t_pre:.2f}s )
     BEM: ( {t_pre_assy:.3f}s ) | MICS: ( {t_pre_mics:.3f}s ) | RAM ( {pre_RAM_gb:.3f}GB )
     
 Heuristic estimates for Frequency Sweep based on RAM available ( {RAM_gb:.2f}GB ):
 Estimated (+)RAM per Freq: ( {cost_per_worker_gb:.2f} GB ) | Parallel Frequency Workers: ( {num_workers} )
 [i] To avoid race conditions in parallel sums, Numba MAX CPUs is set to ( {threads_per_worker} )
     Numpy solve [np.linalg.solve(A, B)] calls into LAPACK (Intel MKL or OpenBLAS), 
     which already does parallel solving. MAX CPUs for Numpy is set to ( {n_CPUs} )
""")
        logger.info(log_pre_stats)
        logger.info(" [i] Promoting heavy arrays to Shared Memory ...")
        shm_static_data = promote_to_shm(static_data)

        file_logger.info(f"\n{'=' * 98}")
        file_logger.info(f" {'Freq (Hz)':<9} | {'Assembly':^8} | {'Solve All':>9}: {'BEM':^8} + {'Mics':^8} | {'RAM (MB)':^10} | {'Results file':<18} | {'Status':^6}")
        file_logger.info(f"{'=' * 98}")

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
        # Terminal progress bar  
        print(f"{'=' * 80}")
        pbar = tqdm(total=num_freqs, desc=" Done", ncols=80, unit="Freq", colour="#ddcd3e")
        
        try:
            with ProcessPoolExecutor(
                max_workers = num_workers,
                initializer = init_worker,
                initargs = (shm_static_data, threads_per_worker, log_f, debug_mode)
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
                    
                    # 13.4 Write to log
                    log_line = (f" {f_done:<7.1f}Hz | {t_assembly/num_workers:^7.3f}s | {t_solve/num_workers:>7.3f}s : {t_solve_bem/num_workers:^8.3f} + {t_solve_mics/num_workers:^8.3f} | {solve_RAM:^10.1f} | {rslt_f:<18} | {'OK':^6}")
                    file_logger.info(log_line)
                        
        finally:
            # Safely releases shared memory allocations regardless of success or sudden crash
            global_shm_cleanup()

        file_logger.info(f"{'=' * 98}")
        pbar.close()  # Close terminal progress bar
        t_exp_0 = time.time()
        print(f"{'=' * 80}")
        
        # 13.5 Finalize and write complete VTU outputs to disk
        logger.info(f"\n --> Writing {num_freqs} binary frequency steps")
        exporter.finalise()
        logger.info(f"     Export Complete. PV results file written to: ( '{parser.model_name}_Results.pvd' )")

        # # DEBUG
        # print("global_p_surf")
        # print(global_p_surf)
        # print("global_v_surf")
        # print(global_v_surf)
        # # DEBUG_end

        # ==================================================================
        # --- 13.6 COMPUTE AND EXPORT TOTAL SURFACE SOUND POWER ---
        # ==================================================================
        if len(parser.surfaces) > 0:
            log_post = f"""
 --> Calculating TOTAL SOUND POWER for all input surfaces:
     A = surface Area | TSW = Total Sound Power, from all frequencies"""
            logger.info(log_post)
            # ----------------------------------
            # ELEMENT-CENTROID POWER CALCULATION
            # ----------------------------------
            from utils import calculate_total_sound_power, generate_power_flux_plot
            surf_pwr_labels = calculate_total_sound_power(
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
                global_mics_elements_conn = static_data['global_mics_elements_conn'],
                tie_registry = tie_registry
            )
            for label in surf_pwr_labels:
                logger.info(f"     {label}")

            csv_filename = f"{parser.model_name}_power.csv"
            png_filename = f"{parser.model_name}_power.png"
            log_post = f"\n     Freq / Power results file written to: ( '{csv_filename}' )"
            # Trigger the headless plot generation right after the CSV writes out
            generate_power_flux_plot(model_name = parser.model_name, suffix="")
            log_post += f"\n     Freq / Power graph plotted to: ( '{png_filename}' )"
            logger.info(log_post)
            
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

--> COMPLETED all Multi-Zone Frequency Steps <--
    Function 'averaged_at_nodes' took: ( {all_t_avr:.2f}s )
    Export Write and VTU Processing:   ( {all_t_exp:.2f}s )

 [i] Shared Memory released.
"""
        summary_log += f"""
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
        logger.info(summary_log)
    except (ValueError, RuntimeError) as e:
        # Expected user/input/mesh errors -> Clean 1-line log, no traceback
        logger.error(f"\n [!] Input ERROR, please check your BEM model setup.\n{e}")
    except Exception as e:
        # Unexpected Python coding bugs (NameError, TypeError, etc.) -> Full traceback for debugging
        logger.error(f"\n [!] UNEXPECTED BUG DETECTED\n{e}")

def main():
    """Application entry point: manages hardware initialization and runs pyBEM."""

    # Get number of physical CPUs to pass onto Numpy libraries for the solve
    # This is required before any `import numpy`
    n_CPUs, n_threads, RAM_gb = get_cpus()
    used_CPUs = n_CPUs

    # This makes sure at the start that all solve libraries are set to a CPU max.
    # This is to minimise race conditions on multi-threading.
    set_hardware_limits(used_CPUs)
    # AFTER, in the code we do try better with Numba, as it has the function
    # set_num_threads(), which the other libraries don't.
    
    start_pybem_app(n_CPUs, used_CPUs, n_threads, RAM_gb)

if __name__ == "__main__":
    main()
