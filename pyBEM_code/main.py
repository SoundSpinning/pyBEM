import os, sys
import numpy as np
import traceback
import time
from tqdm import tqdm

# Import our custom modules
from version import __solver__
from pmx_parser import PMXParser
from solver_core import assemble_system, solve_bem_system, derive_surface_vectors, calculate_field_points
from exporter import PVExporter
from utils import calculate_element_properties, calculate_signed_volume, average_to_nodes
from constants import TOP_LOG_LINES, BEM_TYPE, DEBUG, P_REF

def start_pybem_app():
    # 1. Ask the user for the file name
    print(f"{__solver__}")
    
    filename = input("\nEnter PrePoMax *.inp filename: ").strip()

    # 0. Check if the file actually exists before starting
    if not os.path.exists(filename):
        print(f"ERROR: File '{filename}' not found in the current folder.")
        return

    try:
        # 1. Setup Parser and Load Model
        parser = PMXParser(filename)
        parser.load_model()
        parser.print_model_summary()
        # LOG setup
        log_f = f"{parser.project_name}.log"

        # Create the list of IDs in the exact order they appear in the matrices
        # This ensures index 0 is always Element 101, index 1 is 102, etc.
        bem_ids = sorted(parser.elements.keys())
    
        # 2. Geometry Prep (Centers, Areas, Normals)
        bem_centers, bem_areas, bem_normals = [], [], []
        for eid in bem_ids:  # Changed to use sorted IDs for index alignment
            conn = parser.elements[eid]
            c, a, n = calculate_element_properties(parser.nodes, conn)
            bem_centers.append(c)
            bem_areas.append(a)
            bem_normals.append(n)
        
        # Convert to numpy arrays once to speed up the frequency loop
        bem_centers = np.array(bem_centers)
        bem_areas = np.array(bem_areas)
        bem_normals = np.array(bem_normals)

        # Only if mics exist in the model
        if parser.mics_elements:
            mic_ids = sorted(parser.mics_elements.keys()) # Sorted for consistency
            mic_centers, mic_areas, mic_normals = [], [], []
            for mid in mic_ids:
                mic_conn = parser.mics_elements[mid]
                mc, ma, mn = calculate_element_properties(parser.nodes, mic_conn)
                mic_centers.append(mc)
                mic_areas.append(ma)
                mic_normals.append(mn)
            mic_centers = np.array(mic_centers)
            mic_areas = np.array(mic_areas)
            mic_normals = np.array(mic_normals)
        
        group_ids = {}
        # Default everyone to 1 (BEM)
        for node_id in parser.nodes:
            group_ids[node_id] = 1 

        # Identify Mic nodes and tag them as 2
        for conn in parser.mics_elements.values():
            for node_id in conn:
                group_ids[node_id] = 2
        
        # 3. CHECK on BEM volume: is it interior or exterior, 
        #    or check for holes / free edges in mesh.
        total_vol = calculate_signed_volume(bem_centers, bem_areas, bem_normals)
        if total_vol > 1e-9:
            BEM_TYPE = "INTERIOR"
            log_info = (f"""
 GEOMETRY: Closed (+) Volume detected ({total_vol:.4f} L**3). 
           ( i ) Normals point OUTWARDS ==> {BEM_TYPE} analysis expected.\n""")
        elif total_vol < -1e-9:
            BEM_TYPE = "EXTERIOR"
            log_info = (f"""
 GEOMETRY: Closed (-) Volume detected ({total_vol:.4f} L**3). 
           ( i ) Normals point INWARDS ==> {BEM_TYPE} analysis expected.\n""")
        else:
            print(f"\nError in PRE-PROCESSING: [FATAL INPUT ERROR]")
            log_info = (""" GEOMETRY: Open surface or zero volume detected. CHECK your mesh and normals.
 ( i ) BEM element normals must be consistent and pointing AWAY from the acoustic domain.""")
            print(f"{log_info}")
            raise SystemExit
        
        print(f"{log_info}")

        # 4. Setup BCs
        # we get BCs at the ready for the solver
        bc_map, log_bc_info = parser.get_bcs()
        log_info += log_bc_info
        if DEBUG:
            print(f"\n DEBUG: see '{parser.project_name}.log' for BC CHECKs (ELEMENTAL)")
        # Use 'with' to ensure the file closes even if the app crashes
        with open(log_f, "w") as log:
            log.write(__solver__)
            log.write(str(parser.header_comments))
            log.write(str(parser.top_log))
            log.write(log_info)
            log.write(f" BC-PROCESSING: BC Resolution complete. {len(bc_map)} elements have active BCs.\n")
            log.flush() # Forces write to disk so you can tail the log in real-time

        # Add DEBUG lines, e.g. on BCs applied.
        log_DEBUG = ''
        # time average counts
        all_t_avr = 0
        all_t_exp = 0
        
        # 5. Setup Exporter
        # We pass mics_elements to the exporter so it can merge them into the VTU
        exporter = PVExporter(parser.project_name, parser.nodes, parser.elements, parser.mics_elements)
        
        # 4. Solver Frequency Loop with Log File
        # --- Start Global Timer ---
        global_t0 = time.time()

        num_freqs = len(parser.frequencies)
        print(f"\n--- ACOUSTICS job started at:  {time.ctime()} ---")
        print(f"--- Solving {num_freqs} Frequencies (Steady State Direct) ---\n")

        with open(log_f, "a") as log:
            log.write(f"\n{'-' * 80}")
            log.write(f"\n ACOUSICS job started at: {time.ctime()}")
            log.write(f"\n --- Solving {num_freqs} Frequencies ---")
            log.write(f"\n{'=' * 108}")
            log.write(f"""
{'Freq (Hz)':<9} | {'Assembly':^8} | {'Solve All':>9}: {'BEM':^8} + {'Pres&Vels':^9} + {'Mics':^8} | {'Matrix':<8} | {'Results file':<20} | {'Status':^6}""")
            log.write(f"\n{'=' * 108}\n")
            log.flush() # Forces write to disk so you can tail the log in real-time

            # --- The progress bar loop ---
            pbar = tqdm(parser.frequencies, desc="Done", ncols=80, unit="Freq", colour='black')
            for f in pbar:
                # Update the bar's suffix so we show the freq value
                pbar.set_postfix({"Freq": f"{f}Hz"})

                # --- TIMING: ASSEMBLY ---
                t_asm_0 = time.time()
                # 1. Physics: Calculate wave number k using speed of sound from parser
                k = (2.0 * np.pi * f) / parser.speed_of_sound

                # 2. Matrix Assembly: Assemble G and H matrices (The heavy math)
                # Using the pre-calculated NumPy arrays from the BEM_IDS loop
                G_surf, H_surf = assemble_system(bem_centers, bem_areas, bem_normals, k, BEM_TYPE)
                t_assembly = time.time() - t_asm_0

                # --- TIMING: SOLVE ---
                t_slv_0 = time.time()
                # 3. Solve Linear System: Solve the boundary value problem
                try:
                    # 3.1. Solve the BEM Surface
                    # Passing bem_ids ensures BCs match the matrix rows/columns
                    p_unknowns, cond = solve_bem_system(G_surf, H_surf, bc_map, bem_ids, log=log)

                    t_slv_1 = time.time()
                
                    # 3.2. Reconstruct full p and v vectors for the surface
                    # Updated to use bem_ids to ensure p_surf/v_surf order matches the geometry
                    p_surf, v_surf = derive_surface_vectors(p_unknowns, bc_map, bem_ids)

                    # log_DEBUG:
                    if DEBUG:
                        if f == parser.frequencies[0]:
                            log_DEBUG = f"""
{'-' * 80}
*** DEBUG ***
    =====
    BC CHECKs (ELEMENTAL) - Only for 1st Freq:"""
                            # Get indices of elements that are part of the inlet surface
                            inlet_indices = [i for i, ids in enumerate(bem_ids) if ids in parser.elsets     ['Internal-1_inlet_S2']]
                            # inlet_indices = [i for i, ids in enumerate(bem_ids) if ids in parser.elsets   ['Internal-1_outlet_S2']]

                            for idx in inlet_indices[:3]: # Just check a few
                                p_val = p_surf[idx]
                                v_val = v_surf[idx]
                                db = 20 * np.log10(max(np.abs(p_val), 2e-14) / P_REF)
                                log_DEBUG += f"\n    Element ID{bem_ids[idx]}: {p_val}MPa | {db:.2f}dB"
                                log_DEBUG += f"\n                   {v_val}mm/s"
                    
                    t_slv_2 = time.time()
                    
                    # 3.3. Project to Mics (The Radiation Pass)
                    # Only if mics exist in the model
                    if parser.mics_elements:
                        p_mics = calculate_field_points(
                            mic_centers, 
                            bem_centers, 
                            bem_areas, 
                            bem_normals, 
                            p_surf, 
                            v_surf, 
                            k
                        )
                    t_slv_3 = time.time()
                    
                    t_solve_bem = t_slv_1 - t_slv_0
                    t_solve_pv = t_slv_2 - t_slv_1
                    t_solve_mics = t_slv_3 - t_slv_2
                    t_solve = t_slv_3 - t_slv_0

                except np.linalg.LinAlgError:
                    log.write(f"      Status: FAILED (Singular Matrix)\n")
                    traceback.print_exc()
                    continue # Skip to next frequency

                # 4. Post-Process & Export: Convert element-center results to nodes for smooth ParaView viewing
                try:
                    t_avg_0 = time.time()
                    # 4.1. Get nodal averages for BEM & Mics elements
                    nodal_p_surf = average_to_nodes(parser.nodes, parser.elements, p_surf)
                    nodal_p_mics = average_to_nodes(parser.nodes, parser.mics_elements, p_mics)
                    t_avg_1 = time.time()
                    all_t_avr += t_avg_1-t_avg_0
                    
                    if DEBUG:
                        if f == parser.frequencies[-1]:
                            to_nods_avg_time = all_t_avr / num_freqs if num_freqs > 0 else 0
                            log_DEBUG += f"\n\n    Function 'average_to_nodes' per Freq took: {to_nods_avg_time:.3f}s"
                    
                    t_exp_0 = time.time()
                    # 4.2. Combine them into one master results dictionary/array
                    # Since they are NumPy arrays, adding them merges the results
                    nodal_pressures = nodal_p_surf + nodal_p_mics
                    
                    # 4.3. Create the Group ID array for the Exporter
                    group_ids = {}
                    for node_id in parser.nodes:
                        if any(node_id in conn for conn in parser.mics_elements.values()):
                            group_ids[node_id] = 2  # Microphones
                        else:
                            group_ids[node_id] = 1  # BEM Surface nodes
                    
                    # Write the .vtu file for this frequency
                    exporter.write_vtu(f, nodal_pressures, group_ids=group_ids)
                    t_exp_1 = time.time()
                    all_t_exp += t_exp_1-t_exp_0
                    
                    if DEBUG:
                        if f == parser.frequencies[-1]:
                            exp_avg_time = all_t_exp / num_freqs if num_freqs > 0 else 0
                            log_DEBUG += f"\n    Write / Export of Results per Freq, into PV format took: {exp_avg_time:.3f}s"
                    
                    # Write formatted table header to LOG
                    rslt_f = f'Result_{f:.1f}Hz.vtu'
                    log.write(f"{f:<7.1f}Hz | {t_assembly:^7.3f}s | {t_solve:>7.3f}s : {t_solve_bem:^8.3f} + {t_solve_pv:^9.3f} + {t_solve_mics:^8.3f} | {cond:<8.1f} | {rslt_f:<20} | {'OK':^6}\n")
                    log.flush() # Forces write to disk so you can tail the log in real-time

                except Exception as e:
                    # If an error happens, use pbar.write so it doesn't break the bar
                    t_solve = time.time() - t_slv_0
                    log.write(f"{f:<7.1f}Hz | {t_assembly:<11.4f}s | {t_solve:<8.4f}s | {cond:<8} | {rslt_f:<20} | FAILED: {str(e)}\n")
                    pbar.write(f"Error at {f}Hz: See '{log_f}' for details.")
                    traceback.print_exc()
            
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
    Open '{parser.project_name}_Results.pvd' in ParaView.
{"-" * 80}
"""
            print(SUMMARY())
            log.write(SUMMARY())

    except ValueError as e:
        print(f"\nError loading model: [FATAL INPUT ERROR] {e}")
        traceback.print_exc()
        # return # Exit the function gracefully
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()

if __name__ == "__main__":
    start_pybem_app()