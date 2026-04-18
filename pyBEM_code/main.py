import os
import numpy as np
from numba import get_num_threads, set_num_threads
import traceback
import time
from tqdm import tqdm

# Import our custom modules
from version import __solver__
from pmx_parser import PMXParser
from solver_core import assemble_static, assemble_system, solve_bem_system, calculate_field_points
from exporter import PVExporter
from utils import prepare_geometry, calculate_signed_volume, averaged_at_nodes
from constants import DEBUG, P_REF

def start_pybem_app():
    # 0. Ask the user for the file name
    print(f"{__solver__}")
    
    filename = input("\n Enter PrePoMax *.inp filename: ").strip()

    # Check if the file actually exists before starting
    if not os.path.exists(filename):
        print(f"ERROR: File '{filename}' not found in the current folder.")
        return

    # Assign number of CPU cores for the parallel solve with numba (@njit in 'solver_core.py')
    n_CPUs = get_num_threads()
    used_CPUs = n_CPUs
    if n_CPUs > 4:
        used_CPUs = n_CPUs - 2   # leave 2 CPU cores free on machine for user to do other work
        set_num_threads(used_CPUs)
    str_CPUs = (f"""
 Number of CPU cores found on this machine: ( {n_CPUs} )
 MAX number of cores used for parallel solve is: ( {used_CPUs} )
""")
    
    try:
        # 1. Setup Parser, Load Model and prepare sorted mesh IDs
        parser = PMXParser(filename)
        parser.load_model()
        rho = parser.density

        # Sort nodes & element dictionaries as read from the input parser
        # This ensures later on that index 0 is always Element 1, index 1 is 2, etc.
        sorted_nodes = dict(sorted(parser.nodes.items()))
        sorted_node_ids = list(sorted_nodes.keys())
        sorted_bem_els = dict(sorted(parser.elements.items()))
        sorted_bem_ids = list(sorted_bem_els.keys())

        sorted_mics_els = {}
        # If mics mesh exists in the model
        if parser.mics_elements:
            sorted_mics_els = dict(sorted(parser.mics_elements.items()))
            mics_nodes = {}
            for conn in sorted_mics_els.values():
                for nid in conn:
                    n_coords = sorted_nodes[nid]
                    mics_nodes[nid] = n_coords
            sorted_mics_nodes = dict(sorted(mics_nodes.items()))
            sorted_mics_centers = np.array(list(sorted_mics_nodes.values()))
            parser.n_mics_nodes = len(sorted_mics_nodes)

        # 2. LOG setup | print & LOG model info
        parser.print_model_summary()
        log_f = f"{parser.model_name}.log"
        log_top = f"""{__solver__}{str(parser.header_comments)}{str(parser.top_log)}"""
        log_info = ''

        # 3. Setup Exporter
        # 3.1 Create nodal ID Map (Input mesh node ID -> 0-based VTK Index)
        nodal_id_map = {inp_id: i for i, inp_id in enumerate(sorted_node_ids)}
        # 3.2 Combine all elements for the mesh definition.
        #     We merge BEM and MICS into one dictionary for the export PV 'Cells' section
        sorted_all_els = {**sorted_bem_els, **sorted_mics_els}
        sorted_all_el_ids = sorted_all_els.keys()
        
        # 3.3 Group IDs for PV: BEM = 1 & MICS = 2
        group_ids = {}
        # Default everyone to 1 (BEM)
        for node_id in sorted_nodes:
            group_ids[node_id] = 1 
        # Identify Mic nodes and tag them as 2
        if sorted_mics_els:
            for conn in sorted_mics_els.values():
                for node_id in conn:
                    group_ids[node_id] = 2
        
        # 3.4 We pass sorted nodes, bem & mics elements to the exporter to write them into the VTU files
        exporter = PVExporter(parser.model_name, sorted_nodes, sorted_node_ids, nodal_id_map, sorted_all_els, sorted_all_el_ids, group_ids)

        # 4. Geometry Prep (Centers, Areas, Normals)
        bem_nodal_coords, bem_centers, bem_areas, bem_normals, elem_ratio, max_el_length = prepare_geometry(sorted_nodes, sorted_bem_els)
        # print(bem_nodal_coords)
        # print(bem_centers)

        # 5. CHECK on BEM volume: is it interior or exterior, 
        #    or check for holes / free edges in mesh.
        bem_total_vol, bem_total_area, bem_CoG = calculate_signed_volume(bem_centers, bem_areas, bem_normals)
        log_info += f"""
 INPUT MESH: Total BEM AREA is: ( {bem_total_area:.4} L**2 )
             CoG of the BEM domain is at: [ {bem_CoG[0]:.4}, {bem_CoG[1]:.4}, {bem_CoG[2]:.4} ] L
             ( i ) The MAX BEM element aspect ratio found is: ( {np.max(elem_ratio):.2f} )
                   Mesh element size is (approx.): ( {np.mean(max_el_length):.2f} L )
             ( i ) The MAX Freq suggested is (approx.): ( {parser.speed_of_sound / (8*np.mean(max_el_length)):.1f}Hz )
"""
        if bem_total_vol > 1e-9:
            BEM_TYPE = "INTERIOR"
            H_sign = -1.0
            log_info += (f"""
             Closed (+) Volume detected: ( {bem_total_vol:.4} L**3 )
             ( i ) Normals point OUTWARDS ==> {BEM_TYPE} analysis expected.\n""")
        elif bem_total_vol < -1e-9:
            BEM_TYPE = "EXTERIOR"
            H_sign = 1.0
            log_info += (f"""
             Closed (-) Volume detected: ( {bem_total_vol:.4} L**3 )
             ( i ) Normals point INWARDS ==> {BEM_TYPE} analysis expected.\n""")
        else:
            log_info += (f"""
 ERROR       Open surface or zero volume detected. CHECK your mesh and normals.
             ( !e! ) BEM element normals must be consistent and pointing AWAY from the acoustic domain.
""")
            print(f"{log_info}")
            with open(log_f, "w") as log:
                log.write(log_top)
                log.write(log_info)
            raise ValueError(f"\n ERROR in PRE-PROCESSING: see '{log_f}' for more details.\n")

        # 6. Setup BCs
        # we get BCs at the ready for the solver
        bc_map, log_bc_info = parser.get_bcs()
        print(bc_map)
        log_info += '\n'+log_bc_info
        print(f"{log_info}")
        if DEBUG:
            print(f"\n DEBUG: see '{parser.model_name}.log' for input BC CHECKs (ELEMENTAL & NODAL) after the solve.")
        # Use 'with' to ensure the file closes even if the app crashes
        with open(log_f, "w") as log:
            log.write(log_top)
            log.write(log_info)
            log.flush() # Forces write to disk so you can tail the log in real-time

        # Add DEBUG lines, e.g. on BCs applied, etc
        log_DEBUG = ''
        # time average counts
        all_t_avr = 0
        all_t_exp = 0
        
        # 7. Solver Frequency Loop with Log File
        #    =====================
        # --- Start Global Timer ---
        global_t0 = time.time()
        # Get job settings at the ready
        min_freq = min(parser.frequencies)
        max_freq = max(parser.frequencies)
        num_freqs = len(parser.frequencies)
        del_freq = parser.frequencies[1] - parser.frequencies[0]
        # This value controls when high-order integration kicks in.
        # i.e. only for contributions when near the BEM element of interest.
        # TODO: test in the future this factor for near distance.
        hi_order_length = np.mean(max_el_length)

        str_CPUs += (f"""
 First Assembly and compile of [G] & [H] matrices takes longer. 
 Hold tight, it gets faster after, see times per Freq table in '{parser.model_name}.log'.
 """)   
        print(f"\n{'-' * 80}")
        print(f" ACOUSTICS job started at:  {time.ctime()}")
        print(f" --- Solving {num_freqs} Frequencies [{min_freq:.1f}Hz --> {max_freq:.1f}Hz | delta_Hz = {del_freq:.2f}] (Steady State Direct) ---{str_CPUs}")
        
        with open(log_f, "a") as log:
            log.write(f"\n{'-' * 80}")
            log.write(f"\n ACOUSTICS job started at: {time.ctime()}")
            log.write(f"\n --- Solving {num_freqs} Frequencies [{min_freq:.1f}Hz --> {max_freq:.1f}Hz | delta_Hz = {del_freq:.2f}] (Steady State Direct) ---\n")
            log.write(str_CPUs)
            log.flush() # Forces writing to disk, so we can tail the .log file in real-time
        
        # Calculate the static terms (diagonal) of the [H] matrix, k=0
        k = 0
        H_static = assemble_static(bem_nodal_coords, bem_centers, bem_areas, bem_normals, k, H_sign, hi_order_length)
        print(H_static)
        
        with open(log_f, "a") as log:
            log.write(f"\n{'=' * 98}")
            log.write(f"""
 {'Freq (Hz)':<9} | {'Assembly':^8} | {'Solve All':>9}: {'BEM':^8} + {'Mics':^8} | {'Matrix':<8} | {'Results file':<20} | {'Status':^6}""")
            log.write(f"\n{'=' * 98}\n")
            log.flush() # Forces writing to disk, so we can tail the .log file in real-time

            # --- Freq loop: The progress bar ---
            pbar = tqdm(parser.frequencies, desc=" Done", ncols=80, unit="Freq", colour="#ddcd3e", mininterval=0.10)
            for f in pbar:
                # Update the bar's suffix so we show the freq value
                pbar.set_postfix({"Freq": f"{f:.1f}Hz"})

                # --- TIMING: ASSEMBLY ---
                t_asm_0 = time.time()
                # 1. Physics: Calculate wave number (k) using speed of sound from parser
                omega = 2.0 * np.pi * f
                k = omega / parser.speed_of_sound
                #    Calculate rho*omega to be used when VELO BC present in solver
                rho_omega = rho * omega

                # 2. Matrix Assembly: Assemble G and H matrices (The heavy math)
                # Using the pre-calculated NumPy arrays from the sorted_bem_ids loop
                G_bem, H_bem = assemble_system(bem_nodal_coords, bem_centers, bem_areas, bem_normals, k, H_sign, max_el_length, hi_order_length, H_static)
                t_assembly = time.time() - t_asm_0

                # --- TIMING: SOLVE ---
                t_slv_0 = time.time()
                # 3. Solve Linear System: Solve the boundary value problem
                try:
                    # 3.1. Solve the BEM Surface.
                    #      Passing sorted_bem_ids ensures BCs match the matrix rows/columns
                    #      cond: Matrix Condition Number; tells us if the mesh is 'broken' or math is unstable.
                    #      use cond sparingly, can take ~20% of solve time.
                    # 3.2. 'derive_surface_vectors()': Reconstructs full P and V vectors for the BEM elements.
                    p_surf, v_surf, cond = solve_bem_system(G_bem, H_bem, bc_map, sorted_bem_ids, rho_omega, log=log)
                    t_slv_1 = time.time()
                
                    # log_DEBUG:
                    if f == parser.frequencies[0]:
                        if DEBUG:
                            log_DEBUG = f"""
{'-' * 80}
*** DEBUG ***
    =====
    k, omega & rho_omega at {f}Hz: {k:.5}, {omega:.5} & {rho_omega:.5}
    
    BC CHECKs (ELEMENTAL) - Only for 1st Freq:"""
                            # Get indices of elements that are part of the inlet surface
                            inlet_indices = [i for i, ids in enumerate(sorted_bem_ids) if ids in parser.elsets['inlet_S2']]
                            outlet_indices = [i for i, ids in enumerate(sorted_bem_ids) if ids in parser.elsets['outlet_S2']]

                            log_DEBUG += f"\n    INLET"
                            for idx in inlet_indices[-4:]: # Just check a few
                                p_val = p_surf[idx]
                                v_val = v_surf[idx]
                                db = 20 * np.log10(max(np.abs(p_val), 2e-30) / P_REF)
                                log_DEBUG += f"\n    Element inp_ID{sorted_bem_ids[idx]}->PV_ID{idx}:\n    {p_val}MPa | {db:.2f}dB | {v_val}mm/s | {(180/np.pi)*(np.angle(p_val)-np.angle(v_val)):.2f}deg V wrt P"

                            log_DEBUG += f"\n    OUTLET"
                            for idx in outlet_indices[-4:]: # Just check a few
                                p_val = p_surf[idx]
                                v_val = v_surf[idx]
                                db = 20 * np.log10(max(np.abs(p_val), 2e-30) / P_REF)
                                log_DEBUG += f"\n    Element inp_ID{sorted_bem_ids[idx]}->PV_ID{idx}:\n    {p_val}MPa | {db:.2f}dB | {v_val}mm/s | {(180/np.pi)*(np.angle(p_val)-np.angle(v_val)):.2f}deg V wrt P"
                            
                    t_slv_2 = time.time()
                    
                    # 3.3. Project to Mics (The Radiation Pass)
                    # Only if mics exist in the model
                    if sorted_mics_els:
                        p_mics = calculate_field_points(
                            sorted_mics_centers,
                            bem_centers, 
                            bem_areas, 
                            bem_normals, 
                            p_surf, 
                            v_surf, 
                            k,
                            rho_omega
                        )
                    t_slv_3 = time.time()
                    
                    t_solve_bem = t_slv_1 - t_slv_0
                    # t_solve_pv = t_slv_2 - t_slv_1
                    t_solve_mics = t_slv_3 - t_slv_2
                    t_solve = t_slv_3 - t_slv_0

                except np.linalg.LinAlgError:
                    log.write(f"      Status: FAILED (Singular Matrix)\n")
                    traceback.print_exc()
                    continue # Skip to next frequency

                # 4. Post-Process & Export: Convert element-center results to nodes for smooth ParaView viewing
                try:
                    t_avg_0 = time.time()
                    # 4.1. Get nodal averages for BEM elements & Mics nodes
                    nodal_p_surf = averaged_at_nodes(sorted_nodes, sorted_bem_els, p_surf, bem_areas,sorted_mics_nodes, p_mics)
                    t_avg_1 = time.time()
                    all_t_avr += t_avg_1-t_avg_0
                    
                    if f == parser.frequencies[0]:
                        if DEBUG:                            
                            inlet_indices = [(i, ids) for i, ids in enumerate(sorted_nodes.keys()) if ids in parser.nsets['inlet']]
                            log_DEBUG += f"\n\n    BC CHECKs (NODAL) - Only for 1st Freq:"
                            for idx, nid in inlet_indices[-4:]: # Just check a few nodes
                                p_val = nodal_p_surf[idx]
                                db = 20 * np.log10(max(np.abs(p_val), 2e-30) / P_REF)
                                log_DEBUG += f"\n    Node inp_ID{nid}->PV_ID{idx}: {p_val}MPa | {db:.2f}dB"
                    if f == parser.frequencies[-1]:
                        # # Checks on assembly matrices
                        # np.matrix.tofile(H_bem[:], 'matrix.txt', sep=' ', format='%s')
                        # np.matrix.tofile(G_bem[:], 'matrix.txt', sep=' ', format='%s')
                        log_DEBUG += f"""

    ASSY MATRIX CHECKS @ {parser.frequencies[-1]:.1f}Hz:
    Sum of Col#1 in [H]: {np.sum(H_bem[:,0])}
    Sum of last Col# in [H]: {np.sum(H_bem[:,-1])}
    Sum of Row#1 in [H]: {np.sum(H_bem[0,:])}
    Sum of last Row# in [H]: {np.sum(H_bem[-1,:])}
    Sum of Col#1 in [G]: {np.sum(G_bem[:,0])}
    Sum of last Col# in [G]: {np.sum(G_bem[:,-1])}
    Sum of Row#1 in [G]: {np.sum(G_bem[0,:])}
    Sum of last Row# in [G]: {np.sum(G_bem[-1,:])}
"""
                    t_exp_0 = time.time()
                    # 4.2. Combine them into one master results array
                    nodal_pressures = nodal_p_surf
                    # OLD method, ignore. Since they are NumPy arrays, adding them merges the results
                    # nodal_pressures = nodal_p_surf + nodal_p_mics
                    
                    # Write the .vtu file for this frequency
                    exporter.write_vtu(f, nodal_pressures, group_ids=group_ids)
                    t_exp_1 = time.time()
                    all_t_exp += t_exp_1-t_exp_0
                    
                    if f == parser.frequencies[-1]:
                        if DEBUG:
                            to_nods_avg_time = all_t_avr / num_freqs if num_freqs > 0 else 0
                            log_DEBUG += f"\n\n    Function 'averaged_at_nodes' took: ( {to_nods_avg_time:.3f}s ) per Freq."
                            exp_avg_time = all_t_exp / num_freqs if num_freqs > 0 else 0
                            log_DEBUG += f"\n    Write / Export of Results into PV format took: ( {exp_avg_time:.3f}s ) per Freq."
                    
                    # Write formatted table header to LOG
                    rslt_f = f'Result_{f:.1f}Hz.vtu'
                    log.write(f" {f:<7.1f}Hz | {t_assembly:^7.3f}s | {t_solve:>7.3f}s : {t_solve_bem:^8.3f} + {t_solve_mics:^8.3f} | {cond:<8} | {rslt_f:<20} | {'OK':^6}\n")
                    log.flush() # Forces write to disk so we can tail the log in real-time

                except Exception as e:
                    # If an error happens, use pbar.write so it doesn't break the progress bar
                    t_solve = time.time() - t_slv_0
                    log.write(f"{f:<7.1f}Hz | {t_assembly:<11.4f}s | {t_solve:<8.4f}s | {cond:<8} | {rslt_f:<20} | FAILED: {str(e)}\n")
                    pbar.write(f"Error at {f}Hz: See '{log_f}' for details.")
                    traceback.print_exc()
            # print(v_surf)
            # print(p_mics)
            # print(nodal_pressures)
            log.write(log_DEBUG)
            # 5. Final results output
            exporter.write_pvd()
            log.write(f"\n{'-' * 80}")
            log.write(f"\n --- All Frequency Results saved ---")
            
            # --- Final Summary Timing ---
            total_elapsed = time.time() - global_t0
            avg_time = total_elapsed / num_freqs if num_freqs > 0 else 0

            def SUMMARY(): return f"""
{"-" * 80}
*** SIMULATION SUMMARY ***
    ==================
    Total Frequencies:       {num_freqs}
    Total Elapsed Time:      {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)
    Avg Time per Freq:       {avg_time:.3f} seconds
    Simulation Finished at:  {time.ctime()}
    Check '{log_f}' for more details.
    Open '{parser.model_name}_Results.pvd' in ParaView.
{"-" * 80}
"""
            print(SUMMARY())
            log.write(SUMMARY())

    except ValueError as e:
        print(f"\n ERROR loading model: [FATAL INPUT ERROR] {e}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()

if __name__ == "__main__":
    start_pybem_app()