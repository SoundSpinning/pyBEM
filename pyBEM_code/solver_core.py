# The heavy math (Numba-accelerated BEM kernels)
import time
import os
import atexit
import numpy as np
from numba import njit, prange
from multiprocessing import shared_memory
from utils import (
    get_ram, pre_high_order, pre_mid_order, averaged_at_nodes
)
import logging
# Module-level loggers (automatically connected to setup_logger from main.py)
logger = logging.getLogger("pyBEM")           # Dual output (Terminal + Log file)
file_logger = logging.getLogger("pyBEM.file_only") # File-only output


# This keeps track of all blocks created in this session
_SHM_REGISTRY = {}

def global_shm_cleanup():
    """The 'Janitor' that runs when the script ends or crashes."""
    for name, shm in list(_SHM_REGISTRY.items()):
        try:
            shm.close()
            shm.unlink()
            # file_logger.debug(f" ( i ) Cleaned up SHM: {name}") # Debug only
        except Exception:
            pass
    _SHM_REGISTRY.clear()

# Register the janitor with the OS
atexit.register(global_shm_cleanup)

def create_shared_array_directly(shape, dtype, name_tag):
    """Allocates a block of shared memory and returns it as a numpy array."""
    pid = os.getpid()
    unique_name = f"pyBEM_{pid}_{name_tag}"
    
    # Calculate bytes
    nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    
    # Allocate once in SHM
    shm = shared_memory.SharedMemory(create=True, size=nbytes, name=unique_name)
    _SHM_REGISTRY[unique_name] = shm
    
    # Create the numpy view
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    arr.fill(0) # or just leave it 'empty' for speed
    
    # We return the array AND the metadata for the workers
    metadata = ('SHM_MARKER', unique_name, shape, dtype)
    return arr, metadata

def promote_to_shm(data_dict):
    pid = os.getpid()
    shm_mapped_dict = {}
    
    for key, value in data_dict.items():
        # Promote large arrays (tweak 1MB threshold as you see fit)
        if isinstance(value, np.ndarray) and value.nbytes > 1_000_000:
            unique_name = f"pyBEM_{pid}_{key}"
            
            try:
                shm = shared_memory.SharedMemory(create=True, size=value.nbytes, name=unique_name)
                shared_array = np.ndarray(value.shape, dtype=value.dtype, buffer=shm.buf)
                shared_array[:] = value[:] # The ONLY time we copy this data
                
                _SHM_REGISTRY[unique_name] = shm
                shm_mapped_dict[key] = ('SHM_MARKER', unique_name, value.shape, value.dtype)
            
            except FileExistsError:
                # Emergency recovery: attach, kill, and recreate
                old_shm = shared_memory.SharedMemory(name=unique_name)
                old_shm.close()
                old_shm.unlink()
                # Recurse once to create fresh
                return promote_to_shm(data_dict)
        else:
            shm_mapped_dict[key] = value
            
    return shm_mapped_dict

# Crucial: prevents garbage collection in workers
_worker_shm_refs = [] 

def rebuild_from_shm(shm_mapped_dict):
    global _worker_shm_refs
    rebuilt_dict = {}
    
    for key, value in shm_mapped_dict.items():
        if isinstance(value, tuple) and value[0] == 'SHM_MARKER':
            _, name, shape, dtype = value
            shm = shared_memory.SharedMemory(name=name)
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            rebuilt_dict[key] = arr
            _worker_shm_refs.append(shm) # Keep handle open
        else:
            rebuilt_dict[key] = value
            
    return rebuilt_dict

# A global container inside the worker process for fast I/O
_worker_context = {}

from utils import setup_logger
def init_worker(shared_data, threads_per_worker, log_file, debug_mode):
    """Runs ONCE per worker process at startup."""
    global _worker_context, np
    import numpy as np
    rebuilt_dict = rebuild_from_shm(shared_data)
    _worker_context = rebuilt_dict
    # Configure the loggers inside the worker process
    setup_logger(log_file, debug_mode=debug_mode)

# ==================================================================
# --- MODULAR COUPLING HELPERS FOR MAINTAINABILITY ---
# ==================================================================
def apply_tie_collocation_columns(A_global, H_local, G_local, local_j, start_row, n_elements, 
                                  eid, rho_omega, slave_lagrange_col_map, W_slave_to_master, W_master_to_slave, p_col):
    """
    Handles column-wise coefficient assembly for tied interface elements (Master or Slave).
    - Native pressure P_eid goes into column 'p_col' multiplied by H_local.
    - Interface flux (Lagrange multiplier lambda_s) goes into trailing columns mapped via G_local.
    """
    ik_rho_c = 1j * rho_omega

    # 1. Native Pressure Unknown (P_eid) ALWAYS receives H_local in its primary column
    A_global[start_row : start_row + n_elements, p_col] += H_local[:, local_j]

    # 2. SLAVE ELEMENT BRANCH
    if eid in slave_lagrange_col_map:
        l_col = slave_lagrange_col_map[eid]
        # Total coverage fraction (\sum W_{S -> M} <= 1.0) calculated in PRE
        eta_s = sum(W_slave_to_master[eid].values())
        # Scale G_local by the actual active interface coverage
        A_global[start_row : start_row + n_elements, l_col] -= G_local[:, local_j] * ik_rho_c * eta_s

    # if eid in slave_lagrange_col_map:
    #     l_col = slave_lagrange_col_map[eid]
    #     # Direct 1:1 mapping of native slave flux unknown
    #     A_global[start_row : start_row + n_elements, l_col] -= G_local[:, local_j] * ik_rho_c

    # 3. MASTER ELEMENT BRANCH
    elif eid in W_master_to_slave:
        for s_eid, w_m2s in W_master_to_slave[eid].items():
            if s_eid in slave_lagrange_col_map:
                l_col = slave_lagrange_col_map[s_eid]
                # Opposite interface normal (+G) scaled by master area fraction (A_intersect / A_master)
                A_global[start_row : start_row + n_elements, l_col] += G_local[:, local_j] * ik_rho_c * w_m2s

def enforce_interface_pressure_continuity(A_global, B_global, N_all, 
                                          eid_global_p_col, W_slave_to_master, slave_lagrange_col_map):
    """
    PHASE 2: Enforces acoustic pressure potential continuity across tied interfaces.
    Row 'tie_row' (N_all ... N_all + N_slave - 1) enforces:
        P_slave[s_eid] - Sum_m ( W_{S -> M}[s_eid][m_eid] * P_master[m_eid] ) = 0
    """
    for s_eid, master_dict in W_slave_to_master.items():
        if s_eid not in slave_lagrange_col_map:
            continue
            
        tie_row = slave_lagrange_col_map[s_eid]
        s_p_col = eid_global_p_col[s_eid]  # Pressure column for slave element
        
        # 1. Slave Pressure Coefficient (+1.0)
        # Sum of weights in W_slave_to_master for a slave is identically 1.0
        A_global[tie_row, s_p_col] = 1.0
        
        # 2. Interpolated Master Pressure Coefficients (-W_{S -> M})
        for m_eid, weight_s2m in master_dict.items():
            m_p_col = eid_global_p_col[m_eid]  # Pressure column for master element
            A_global[tie_row, m_p_col] = -weight_s2m
            
        # 3. Continuity RHS remains zero
        B_global[tie_row] = 0.0

def reconstruct_master_velocity(global_solution, master_eid, W_slave_to_master, W_master_to_slave, slave_lagrange_col_map, eid_area_map):
    """
    Reconstructs master element normal velocity v_master from slave Lagrange multipliers (lambda_s)
    by averaging acoustic flux over the master element area:
    v_master = sum( w_m2s * lambda_s )
    """
    if master_eid not in W_master_to_slave:
        raise ValueError(f"\n [!] ERROR Master element {master_eid} not found in W_master_to_slave.")

    v_master = 0.0 + 0.0j

    # Map slave Lagrange multipliers (lambda_s = v_slave) to master using W_{M -> S}
    for s_eid, w_m2s in W_master_to_slave[master_eid].items():
        if s_eid in slave_lagrange_col_map:
            l_col = slave_lagrange_col_map[s_eid]
            v_slave = global_solution[l_col]  # Correct Lagrange multiplier column for velocity
            
            # w_m2s = A_intersection / A_master
            v_master += v_slave * w_m2s

    return v_master

# ==================================================================
# --- REFACTORED MAIN WORKER FUNCTION ---
# ==================================================================
# NEW worker to handle multi-zone BEM
def frequency_worker(f, bc_map, sorted_bem_ids, threads_per_worker):
    """
    Independent multi-zone worker function. Runs for EVERY frequency.
    TIED pair constraints:
    Enforces interface continuity via Dual-Lagrange coupling:
    - Primary solution vector holds element pressure unknown (p) in columns 0..N_all-1.
    - Trailing columns (N_all..N_all+N_slave-1) contain slave Lagrange multipliers (lambda = v_slave).
    - Trailing rows enforce interface pressure continuity across master-slave patches.
    """
    # 1. Pull data from Shared Context
    static_data = _worker_context
    amps = static_data.get('amplitudes') # Use .get() to avoid KeyErrors
    damping_f = static_data['damping']
    # ZONES method
    zones_mesh = static_data['zones_mesh']
    zone_offsets = static_data['zone_offsets']            # dict: zone_name -> {'start_idx', 'n_elements'}
    # TIED pairs
    W_slave_to_master = static_data['W_slave_to_master']  # dict: seid -> {meid: weight}
    W_master_to_slave = static_data['W_master_to_slave']  # dict: meid -> {seid: weight}
    master_elements = static_data['master_elements']      # list of unique master eids
    slave_elements = static_data['slave_elements']        # list of unique slave eids

    # PRE-processing if any amplitude curves present
    # A. Resolve Damping
    if amps and 'AMP_damp' in damping_f:
        curve = amps[damping_f['AMP_damp']]
        damping_f = np.interp(f, curve[:, 0], curve[:, 1]) * damping_f['value']
    else:
        damping_f = damping_f['value']

    # B. Resolve BCs
    resolved_bcs = {}
    for eid, info in bc_map.items():
        resolved_bcs[eid] = res = {}
        # We only look for the three BC types
        for bc_type in ['PRES', 'VELO', 'IMPE']:
            if bc_type in info:
                base_val = info[bc_type]
                v_real = base_val.real
                v_imag = base_val.imag

                # --- Scale REAL part ---
                amp_r_key = f'{bc_type}_AMP_real'
                if amps and amp_r_key in info:
                    c_name = info[amp_r_key]
                    if c_name in amps:
                        v_real *= np.interp(f, amps[c_name][:, 0], amps[c_name][:, 1])

                # --- Scale IMAGINARY part ---
                amp_i_key = f'{bc_type}_AMP_imag'
                if amps and amp_i_key in info:
                    c_name = info[amp_i_key]
                    if c_name in amps:
                        v_imag *= np.interp(f, amps[c_name][:, 0], amps[c_name][:, 1])

                # Recombine into a complex number for the solver
                res[bc_type] = complex(v_real, v_imag)

    # w = 2PI*freq
    omega = 2.0 * np.pi * f
    t_asm_0 = time.time()

    # ==================================================================
    # DUAL-LAGRANGE SYSTEM MATRIX:
    # Primary block (0..N_all-1): Domain element pressure unknowns (p).
    # Trailing block (N_all..N_all+N_slave-1): Slave Lagrange multipliers (lambda = v_slave).
    # Constraint rows (N_all..N_all+N_slave-1): Interface pressure continuity.
    # ==================================================================
    N_all = sum(info['n_elements'] for info in zone_offsets.values())
    N_slave = len(slave_elements)
    total_matrix_size = N_all + N_slave

    A_global = np.zeros((total_matrix_size, total_matrix_size), dtype=np.complex128)
    B_global = np.zeros((total_matrix_size,), dtype=np.complex128)

    # REUSE THE PRE-CALCULATED GLOBAL MAPS DIRECTLY
    # p_col: The column index in A_global storing the Pressure Unknown for element eid.
    eid_global_p_col = static_data['global_eid_to_col']
    eid_area_map = static_data['global_bem_areas']

    # Maps each slave element ID (s_eid) to its dedicated trailing column (N_all ... End)
    # contains the Slave Interface Pressure Multipliers (lambda)
    slave_lagrange_col_map = {s_eid: N_all + idx for idx, s_eid in enumerate(W_slave_to_master.keys())}

    # Pre-build lookup sets for O(1) compliance checks inside loops
    master_set = set(master_elements)
    slave_set = set(slave_elements)

    # ==================================================================
    # PHASE 1: LOOP PER ZONE - ASSEMBLE COLLOCATION COEFFICIENTS
    # ==================================================================
    for zone_name, z_mesh in zones_mesh.items():
        z_bem = static_data['pre_bem_data'][zone_name]
        alloc = zone_offsets[zone_name]
        
        start_row = alloc['start_idx']
        n_elements = alloc['n_elements']
        
        H_sign = static_data['global_h_signs'][zone_name]
        order_length = static_data['global_order_lengths'][zone_name]
        max_el_length = order_length

        c_zone = static_data['global_c'][zone_name]
        rho_zone = static_data['global_rho'][zone_name]
        k_zone = omega / c_zone
        if damping_f != 0:
            k_zone = k_zone * (1.0 - (1j * damping_f * 0.5))
            
        rho_omega = rho_zone * omega

        G_local, H_local = main_assembly(
            z_bem['gp_per_element'], z_bem['GP_start_idx'],
            z_bem['R_map'], z_bem['G_static_map'], z_bem['H_static_map'],
            z_bem['G_diag_static'], z_bem['H_diag_static'],
            k_zone, H_sign, max_el_length, order_length
        )
        
        for local_j, eid in enumerate(z_mesh['elements'].keys()):
            p_col = eid_global_p_col[eid]  # Native pressure column (0 ... N_all - 1)
            bc = resolved_bcs.get(eid, {})

            # --- Tied Interface Elements (Master or Slave) ---
            if eid in master_set or eid in slave_set:
                apply_tie_collocation_columns(
                    A_global, H_local, G_local, local_j, start_row, n_elements,
                    eid, rho_omega, slave_lagrange_col_map, W_slave_to_master, W_master_to_slave, p_col
                )
            
            # --- Standard Non-Tied Domain Elements ---
            else:
                # Case 1: Velocity is known (Vibrating Wall - Neumann BC)
                if 'VELO' in bc:
                    A_global[start_row : start_row + n_elements, p_col] += H_local[:, local_j]
                    B_global[start_row : start_row + n_elements] -= G_local[:, local_j] * bc['VELO'] * (1j * rho_omega)
                    # Case 1.1 Simultaneous VELO + IMPE (Robin BC)
                    if 'IMPE' in bc:
                        z_val = bc['IMPE'] if abs(bc['IMPE']) > 1e-12 else 1e-12
                        A_global[start_row : start_row + n_elements, p_col] += G_local[:, local_j] * (1j * rho_omega / z_val)
                # Case 2: Pressure is known (Open end / Source) - (Dirichlet BC)
                elif 'PRES' in bc:
                    A_global[start_row : start_row + n_elements, p_col] -= G_local[:, local_j] * (1j * rho_omega)
                    B_global[start_row : start_row + n_elements] -= H_local[:, local_j] * bc['PRES']
                # Case 3: Impedance (Absorbent material)
                elif 'IMPE' in bc and 'VELO' not in bc:
                    z_val = bc['IMPE'] if abs(bc['IMPE']) > 1e-12 else 1e-12
                    A_global[start_row : start_row + n_elements, p_col] += H_local[:, local_j] + (G_local[:, local_j] * (1j * rho_omega / z_val))
                # Case 4: Rigid Wall, v=0 (Default)
                else:
                    A_global[start_row : start_row + n_elements, p_col] += H_local[:, local_j]

    t_assembly = time.time() - t_asm_0
    t_slv_0 = time.time()

    # ==================================================================
    # PHASE 2: ASSEMBLE LAGRANGE CONSTRAINT ROWS [C] (N_all .. N_all+N_slave-1)
    #          ENFORCE INTERFACE PRESSURE CONTINUITY ON TIED PAIRS
    # ==================================================================
    enforce_interface_pressure_continuity(A_global, B_global, N_all, 
                                          eid_global_p_col, W_slave_to_master, slave_lagrange_col_map)

    # DEBUG
    # ==================================================================
    # DIAGNOSTIC: PRINT MATRIX MAP AND COEF VALUES
    # ==================================================================
    file_logger.debug(f"\n{'='*60}\nMATRIX DIAGNOSTIC FOR FREQ: {f} Hz")
    file_logger.debug(f"Matrix Size: {total_matrix_size}x{total_matrix_size} (N_all: {N_all}, N_slave: {N_slave})")
    file_logger.debug(f"Number of Master elements: {len(master_elements)}, Slave elements: {len(slave_elements)}")
    
    # 1. Row and Column Zero Checks
    zero_rows = np.where(~A_global.any(axis=1))[0]
    zero_cols = np.where(~A_global.any(axis=0))[0]
    file_logger.debug(f"Completely empty rows: {list(zero_rows)}")
    file_logger.debug(f"Completely empty columns: {list(zero_cols)}")

    # 2. Text-Based Sparsity Map (Shows where numbers vs zeros are)
    file_logger.debug("\n[A_global Structural Layout Map] (X = Non-zero, . = Pure Zero):")
    for r in range(total_matrix_size):
        row_str = ""
        for c in range(total_matrix_size):
            if abs(A_global[r, c]) > 1e-15:
                row_str += "X "
            else:
                row_str += ". "
        # Highlight the trailing Dual-Lagrange rows/columns visually
        if r == N_all:
            file_logger.debug(f"{'-' * (total_matrix_size * 2)}\n")
        file_logger.debug(f"Row {r:02d} | {row_str}")

    # # 3. Print raw matrices cleanly formatted if you want to see exact values
    # # np.set_printoptions(precision=3, suppress=True, linewidth=200)
    # np.set_printoptions(precision=3, suppress=False, linewidth=400, threshold=1e4)
    # file_logger.debug("\n[A_global Raw Values]:")
    # file_logger.debug(A_global)
    # file_logger.debug("\n[B_global Raw Values]:")
    # file_logger.debug(B_global)
    file_logger.debug(f"{'='*60}\n")
    # DEBUG_end

    # DEBUG
    # ==================================================================
    # DIAGNOSTIC: ISOLATE & INSPECT CONSTRAINT MATRIX [C]
    # ==================================================================
    file_logger.debug(f"\n{'='*60}\nCONSTRAINT BLOCK [C] DIAGNOSTIC (N_all: {N_all}, N_slave: {N_slave})")

    # Slice out the constraint rows: shape is (N_slave, total_matrix_size)
    C_matrix = A_global[N_all:, :]

    # 1. Sparsity / Structure Map of [C] only
    file_logger.debug("\n[C Matrix Layout Map] (X = non-zero, . = zero):")
    for local_r, global_r in enumerate(range(N_all, total_matrix_size)):
        row_str = "".join(["X " if abs(C_matrix[local_r, c]) > 1e-15 else ". " for c in range(total_matrix_size)])
        file_logger.debug(f"Constraint Row {global_r:02d} (Slave local {local_r:02d}) | {row_str}")

    # 2. Detailed Index & Value Extraction
    file_logger.debug("\n[C Matrix Explicit Coupling Pairs]:")
    for local_r, global_r in enumerate(range(N_all, total_matrix_size)):
        non_zero_cols = np.where(np.abs(C_matrix[local_r, :]) > 1e-15)[0]
        couplings = []
        for c in non_zero_cols:
            val = C_matrix[local_r, c]
            # Format complex or real values cleanly
            val_str = f"{val.real:+.3f}{val.imag:+.3f}j" if np.iscomplexobj(val) else f"{val:+.3f}"
            couplings.append(f"col {c:02d}: {val_str}")

        file_logger.debug(f"Row {global_r:02d} -> " + ", ".join(couplings))

    # 3. Sum of Master Weights Verification (Row Check)
    # Check that the sum of master coupling weights for each slave row balances appropriately
    file_logger.debug("\n[C Matrix Row Sum Check]:")
    for local_r, global_r in enumerate(range(N_all, total_matrix_size)):
        row_sum = np.sum(C_matrix[local_r, :])
        file_logger.debug(f"Row {global_r:02d} total row sum: {row_sum:.6f}")

    file_logger.debug(f"{'='*60}\n")
    # DEBUG_end

    # ==================================================================
    # PHASE 3.1: SOLVE the BEM system MATRIX --> Ax = B
    # ==================================================================
    global_solution = np.linalg.solve(A_global, B_global)
    
    p_surf = np.zeros(len(sorted_bem_ids), dtype=np.complex128)
    v_surf = np.zeros(len(sorted_bem_ids), dtype=np.complex128)
    global_areas = np.zeros(len(sorted_bem_ids))
    
    # ==================================================================
    # PHASE 3.2: DISTRIBUTE SOLVED SYSTEM RESULTS BACK TO BEM ELEMENTS
    # ==================================================================
    for zone_name, offset_info in zone_offsets.items():
        start_row = offset_info['start_idx']
        n_elements = offset_info['n_elements']
        
        zone_solved = global_solution[start_row : start_row + n_elements]
        z_mesh = zones_mesh[zone_name]
        z_areas = static_data['pre_bem_data'][zone_name]['areas']
        
        for local_j, eid in enumerate(z_mesh['elements'].keys()):
            solved_val = zone_solved[local_j]
            bc = resolved_bcs.get(eid, {})
            
            if eid in master_set:
                # 1. Master Pressure: Read directly from primary solution column
                p_col = eid_global_p_col[eid]
                p_val = global_solution[p_col]
                # 2. Master Velocity: Reconstruct via mortar flux averaging
                v_val = reconstruct_master_velocity(
                    global_solution, eid, W_slave_to_master, W_master_to_slave, slave_lagrange_col_map, eid_area_map
                )

            elif eid in slave_set:
                # 1. Slave Pressure: Read directly from primary solution column
                p_col = eid_global_p_col[eid]
                p_val = global_solution[p_col]
                # 2. Slave Velocity: Read directly from Lagrange multiplier trailing column
                l_col = slave_lagrange_col_map[eid]
                v_val = global_solution[l_col]

            else:
                if 'VELO' in bc:
                    p_val = solved_val
                    v_val = bc['VELO']
                    if 'IMPE' in bc:
                        z_val = bc['IMPE'] if abs(bc['IMPE']) > 1e-12 else 1e-12
                        v_val += solved_val / z_val
                elif 'PRES' in bc:
                    p_val = bc['PRES']
                    v_val = solved_val
                elif 'IMPE' in bc and 'VELO' not in bc:
                    p_val = solved_val
                    z_val = bc['IMPE'] if abs(bc['IMPE']) > 1e-12 else 1e-12
                    v_val = solved_val / z_val
                else:
                    p_val = solved_val
                    v_val = 0.0 + 0.0j
                    
            if eid in sorted_bem_ids:
                global_idx = eid_global_p_col[eid]
                p_surf[global_idx] = p_val
                v_surf[global_idx] = v_val
                global_areas[global_idx] = z_areas[local_j]

    solve_RAM = get_ram()
    t_slv_1 = time.time()

    # ==================================================================
    # PHASE 4: POST-PROCESS MICS + AVERAGE AT NODES ALL RESULTS
    # ==================================================================
    all_p_mics_list = []
    all_vx_mics_list = []
    all_vy_mics_list = []
    all_vz_mics_list = []
    ordered_mic_tracking = []  
    
    if 'pre_mics_data' in static_data and static_data['pre_mics_data']:
        for zone_name in static_data['pre_mics_data'].keys():
            if zone_name in zones_mesh:
                z_mesh = zones_mesh[zone_name]
                z_mic = static_data['pre_mics_data'][zone_name]
                z_bem = static_data['pre_bem_data'][zone_name]
                
                if z_mic and z_mic.get('num_mics', 0) > 0:
                    c_zone = static_data['global_c'][zone_name]
                    rho_zone = static_data['global_rho'][zone_name]
                    k_zone = omega / c_zone
                    if damping_f != 0:
                        k_zone = k_zone * (1.0 - (1j * damping_f * 0.5))
                    rho_omega_zone = rho_zone * omega
                    H_sign_zone = static_data['global_h_signs'][zone_name]
                    
                    # --- Extract local zone BEM results ---
                    alloc = zone_offsets[zone_name]
                    start_row = alloc['start_idx']
                    n_elements = alloc['n_elements']
                    p_surf_local = p_surf[start_row:start_row+n_elements]
                    v_surf_local = v_surf[start_row:start_row+n_elements]
                    
                    # Compute zone MICS results
                    p_mics_zone, v_mics_x_zone, v_mics_y_zone, v_mics_z_zone = calculate_mics(
                        z_mic['pre_mics_G'], z_mic['pre_mics_H'], z_mic['pre_mics_R'],
                        z_mic['pre_mics_dx'], z_mic['pre_mics_dy'], z_mic['pre_mics_dz'],
                        z_mic['num_mics'], z_bem['areas'], z_bem['normals'],
                        p_surf_local, v_surf_local, k_zone, rho_omega_zone, H_sign_zone
                    )
                    
                    # Accumulate arrays
                    all_p_mics_list.append(p_mics_zone)
                    all_vx_mics_list.append(v_mics_x_zone)
                    all_vy_mics_list.append(v_mics_y_zone)
                    all_vz_mics_list.append(v_mics_z_zone)
                    
                    local_mics_dict = z_mic.get('mics_nodes_dict') if z_mic.get('mics_nodes_dict') else z_mic.get('mics_nodes', {})
                    
                    # Store both the zone name and node ID to ensure absolute row tracking integrity
                    for nid in local_mics_dict.keys():
                        ordered_mic_tracking.append((zone_name, nid))

    p_mics = np.concatenate(all_p_mics_list) if all_p_mics_list else None
    v_mics_x = np.concatenate(all_vx_mics_list) if all_vx_mics_list else None
    v_mics_y = np.concatenate(all_vy_mics_list) if all_vy_mics_list else None
    v_mics_z = np.concatenate(all_vz_mics_list) if all_vz_mics_list else None
    # Extract the raw, sequential node IDs from the tracking list
    ordered_mic_ids = [nid for (zone, nid) in ordered_mic_tracking] if ordered_mic_tracking else None

    t_slv_2 = time.time()
    t_avg_0 = time.time()
    
    nodal_pressures = averaged_at_nodes(
        static_data['sorted_nodes'], static_data['sorted_bem_els'], p_surf, 
        global_areas, eid_global_p_col, ordered_mic_ids, p_mics, static_data['nodal_id_map']
    )

    t_avg_1 = time.time()
    
    metadata = {
        't_assembly': t_assembly,
        't_solve_bem': t_slv_1 - t_slv_0,
        't_solve_mics': t_slv_2 - t_slv_1,
        't_avrg_nodes': t_avg_1 - t_avg_0,
        'solve_RAM': solve_RAM,
        # --- PACKAGE RAW ARRAYS DIRECTLY INTO METADATA FOR THE POWER CALCULATOR ---
        'p_surf': p_surf,
        'v_surf': v_surf,
        'p_mics': p_mics,
        'v_mics_x': v_mics_x,
        'v_mics_y': v_mics_y,
        'v_mics_z': v_mics_z
    }
    
    return f, nodal_pressures, metadata

# @njit(parallel=False, boundscheck=True)  # <-- Change this temporarily to DEBUG
@njit(parallel=True, cache=True)
def pre_assembly(element_nodes, centers, areas, normals):
    """
    Pre-Computes G and H (static, k=0) matrices using the input mesh ZONES:
    - Green's Function Kernel (G-matrix): Gij = 1 / (4PI*r)
    - Derivative Kernel (H-matrix): Hij = Gij / r * r_dot_n
    It also pre-computes fast arrays with all GPs info for fast assembly in the solve.
    element_nodes: BEM elem nodal coords
    centers: (N, 3) array
    areas: (N,) array
    normals: (N, 3) array
    """
    n_els = len(centers)    # number of BEM elements
    inv_4pi = 1.0 / (4.0 * np.pi) # 4*pi is a constant in the Green's function denominator

    # STEP 1: Calculate Total GPs per ZONES
    gp_per_element = np.zeros(n_els, dtype=np.int64)  # Force i64 for numba to work
    for i in range(n_els):
        elem_n_nodes = len(element_nodes[i])
        gp_per_element[i] = 11 if elem_n_nodes == 3 else 14

    total_gps = int(np.sum(gp_per_element))
    
    # STEP 2: Pre-allocate local static arrays with Exact Size for speed
    # These are "Flat Lists" with all GPoints + dtype to save on RAM
    GP_start_idx = np.zeros(n_els, dtype=np.int64)
    offset = 0
    for j in range(n_els):
        GP_start_idx[j] = offset
        offset += gp_per_element[j]

    R_map = np.zeros((n_els, total_gps), dtype=np.float32)
    G_static_map = np.zeros((n_els, total_gps), dtype=np.float32)
    H_static_map = np.zeros((n_els, total_gps), dtype=np.float32)

    # STEP 3: Compute distances, integration orders and basic green function static terms
    for j in prange(n_els):
        nodes = element_nodes[j]
        area = areas[j]
        nj = normals[j]
        n_pts = gp_per_element[j]
        start = GP_start_idx[j]

        pts_coords = np.zeros((n_pts, 3), dtype=np.float64)
        pts_weights = np.zeros(n_pts, dtype=np.float64)
        # Calculate ALL possible GPs(coords, weights) per SOURCE element
        if n_pts == 11: # TRIAs: 1 + 3 + 7 GPs levels
            mid_gp_idx = 4
        elif n_pts == 14: # QUADs: 1 + 4 + 9 GPs levels
            mid_gp_idx = 5

        # Integration options mapping:
        # used later in solver as a function of distance receiver --> source elements:
        # CENTROID (1pnt):
        pts_coords[0] = centers[j]
        pts_weights[0] = 1.0 * area
        # MID-order (3pnts or 4pnts):
        pts, weights = pre_mid_order(nodes, area)
        pts_coords[1:mid_gp_idx] = pts
        pts_weights[1:mid_gp_idx] = weights
        # HIGH-order (7pnts or 9pnts):
        pts, weights = pre_high_order(nodes, area)
        pts_coords[mid_gp_idx:n_pts] = pts
        pts_weights[mid_gp_idx:n_pts] = weights
        # --- Fill the "Flat Lists" ---
        for i in range(n_els): # Loop over RECEIVER elements
            recv_center = centers[i]
            # Explicit loop is "The Numba Way" - no axis issues here
            for p in range(n_pts):
                idx = start + p
                # 1. Calculate the vector from GP to Receiver Center
                rx = recv_center[0] - pts_coords[p, 0]
                ry = recv_center[1] - pts_coords[p, 1]
                rz = recv_center[2] - pts_coords[p, 2]
                # r_vec = recv_center - pts_coords[p]
                # 2. Distance math
                r2 = rx*rx + ry*ry + rz*rz
                # r = np.linalg.norm(r_vec)
                # r2 = r * r
                if r2 < 1e-18: # Safety for self-term
                    r2 = 1e-18
                r = np.sqrt(r2)
                
                R_map[i, idx] = r
                # Fill Static Influence Matrices
                # 3. Bake the G Static Map: G = weight / (4 * pi * r)
                g_base = pts_weights[p] * inv_4pi / r
                G_static_map[i, idx] = g_base
                # 4. Bake the H Static Map: H = G_static * (dot / r^2)
                # dot = np.dot(r_vec, normals[j])
                dot = rx*nj[0] + ry*nj[1] + rz*nj[2]
                H_static_map[i, idx] = g_base * (dot / r2)

    # STEP 4: Calc the static [G] & [H] diagonal self-terms
    G_diag_static = np.zeros(n_els, dtype=np.float64)
    H_diag_static = np.zeros(n_els, dtype=np.float64)
    for i in range(n_els):
        row_sum = 0.0
        G_diag_static[i] = np.sqrt(areas[i] / np.pi) * 2.0 * inv_4pi
        
        for j in range(n_els):
            if i == j:
                continue # Skip the pre-diagonals in their sum
            n_pts = gp_per_element[j]
            if n_pts == 11:   # TRIAs: 1 + 3 + 7
                local_start = GP_start_idx[j] + 4
                n_high_pts = 7
            elif n_pts == 14: # QUADs: 1 + 4 + 9
                local_start = GP_start_idx[j] + 5
                n_high_pts = 9
                
            # SUM ONLY HIGH ORDER from the static pre-baked H values for this source element
            for p in range(n_high_pts):
                p_local = local_start + p
                row_sum += H_static_map[i, p_local]

        # The BEM [H] Diagonal Balance
        H_diag_static[i] = -row_sum

    return gp_per_element, GP_start_idx, R_map, G_static_map, H_static_map, G_diag_static, H_diag_static

@njit(parallel=True, cache=True)
def pre_mics(mics_nodes, bem_centers, bem_normals):
    """
    Calculates influence mesh surface (BEM) properties onto Microphone nodes.
    mics_nodes: mics nodal coords.
    bem_centers: BEM elements CoG coords.
    """
    num_mics = len(mics_nodes)
    num_surf = len(bem_centers)
    pre_mics_G = np.zeros((num_mics, num_surf), dtype=np.float32)
    pre_mics_H = np.zeros((num_mics, num_surf), dtype=np.float32)
    pre_mics_R = np.zeros((num_mics, num_surf), dtype=np.float32)
    # 3D velocity vectors
    pre_mics_dx = np.zeros((num_mics, num_surf), dtype=np.float32)
    pre_mics_dy = np.zeros((num_mics, num_surf), dtype=np.float32)
    pre_mics_dz = np.zeros((num_mics, num_surf), dtype=np.float32)

    inv_4pi = 1.0 / (4.0 * np.pi)

    for i in prange(num_mics):
        for j in range(num_surf):
            r_vec = mics_nodes[i] - bem_centers[j]
            r = np.linalg.norm(r_vec)
            # Unit directional vectors components
            dx = r_vec[0] / r
            dy = r_vec[1] / r
            dz = r_vec[2] / r
            
            pre_mics_dx[i, j] = dx
            pre_mics_dy[i, j] = dy
            pre_mics_dz[i, j] = dz

            pre_mics_R[i, j] = r
            # Green's Function (G) 1/r
            pre_mics_G[i, j] = inv_4pi / r

            # Dot product of (r_vec) and the surface normal / r
            # r_dot_n = np.dot(r_vec, bem_normals[j]) / r
            r_dot_n = (dx * bem_normals[j, 0] + dy * bem_normals[j, 1] + dz * bem_normals[j, 2])
            pre_mics_H[i, j] = r_dot_n

    return pre_mics_G, pre_mics_H, pre_mics_R, pre_mics_dx, pre_mics_dy, pre_mics_dz, num_mics

# NEW main_assembly to handle multi-zone BEM
# @njit(parallel=False, boundscheck=True)  # <-- Uncomment / use this to DEBUG
@njit(parallel=True, cache=True)
def main_assembly(gp_per_element, GP_start_idx, R_map, G_static_map, H_static_map, 
                  G_diag_static, H_diag_static, k, H_sign, max_el_length, order_length):
    """
    Computes G and H matrices PRE static ones + dynamic terms per ZONE:
    - Green's Function Kernel (G-matrix): Gij = exp(jk*r) * G_static
    - Derivative Kernel (H-matrix): Hij = H_sign * exp(jk*r) * (jk*r - 1) * H_static
    k: wave number (complex or float)
    Accounts for INTERIOR (H_sign=-1) or EXTERIOR (H_sign=1)
    order_length: pre-calculated MAX (approx.) edge size per BEM ZONE. 
    Used for local distance search for mid-order, high-order & centroid integration.
    """
    n_els = len(G_diag_static)    # number of BEM elements in ZONE
    G = np.zeros((n_els, n_els), dtype=np.complex128)
    H = np.zeros((n_els, n_els), dtype=np.complex128)

    # Thresholds for mid- or high-order GPs based on distance logic
    limit_high = order_length * 3
    limit_mid = order_length * 5

    for i in prange(n_els):
        for j in range(n_els):  # from source j to receiver i
            n_pts = gp_per_element[j]
            start = GP_start_idx[j]
            # Identify element types by total_GPs count from PRE
            # Used to set limits between different integration orders.
            if n_pts == 11:   # TRIA: 1 (Cent) + 3 (Mid) + 7 (High)
                high_start_idx = 4
            elif n_pts == 14: # QUAD: 1 (Cent) + 4 (Mid) + 9 (High)
                high_start_idx = 5
            # We use the centroid (the first point in the array) for the distance check
            r_centroid = R_map[i, start]

            if i == j:
                G[i, j] = G_diag_static[j]
                H[i, j] = H_diag_static[j]
            else:
                if r_centroid < limit_high:
                    p_start, p_end = high_start_idx, n_pts
                elif r_centroid < limit_mid:
                    p_start, p_end = 1, high_start_idx
                else:
                    p_start, p_end = 0, 1

                g_temp = 0.0 + 0.0j
                h_temp = 0.0 + 0.0j
                
                for p_offset in range(p_start, p_end):
                    p_idx = start + p_offset
                    r_p = R_map[i, p_idx]
                    exp_jkr = np.exp(1j * k * r_p)

                    # Fetch pre-baked static terms using index
                    gs = G_static_map[i, p_idx]
                    hs = H_static_map[i, p_idx]

                    g_temp += exp_jkr * gs
                    h_temp += H_sign * exp_jkr * (1j * k * r_p - 1.0) * hs

                G[i, j] = g_temp
                H[i, j] = h_temp

    return G, H

# NEW calculate_mics to handle multi-zone BEM & total power calcs
@njit(parallel=True, cache=True)
def calculate_mics(pre_mics_G, pre_mics_H, pre_mics_R, pre_mics_dx, pre_mics_dy, pre_mics_dz, 
                   num_mics, bem_areas, bem_normals, p_surf, v_surf, k, rho_omega, H_sign):
    """
    Projects solved Element (BEM) results onto Microphone nodes.
    Kirchhoff-Helmholtz boundary integral.
    """
    num_surf = len(bem_areas)
    p_mics = np.zeros(num_mics, dtype=np.complex128)
    
    # Component arrays for nodal velocity vectors
    v_mics_x = np.zeros(num_mics, dtype=np.complex128)
    v_mics_y = np.zeros(num_mics, dtype=np.complex128)
    v_mics_z = np.zeros(num_mics, dtype=np.complex128)
    
    # Scaling factor for velocities
    v_surf = v_surf * (1j * rho_omega)
    
    # Pre-compute wave number squares and coefficients for efficiency
    k2 = k * k
    inv_rho_omega = -1.0 / (1j * rho_omega)

    for i in prange(num_mics):
        sum_p = 0.0 + 0.0j
        sum_grad_px = 0.0 + 0.0j
        sum_grad_py = 0.0 + 0.0j
        sum_grad_pz = 0.0 + 0.0j
        
        for j in range(num_surf):
            r = pre_mics_R[i, j]
            inv_r = 1.0 / r
            inv_r2 = inv_r * inv_r
            
            dx = pre_mics_dx[i, j]
            dy = pre_mics_dy[i, j]
            dz = pre_mics_dz[i, j]
            
            r_dot_n = pre_mics_H[i, j]
            nx = bem_normals[j, 0]
            ny = bem_normals[j, 1]
            nz = bem_normals[j, 2]

            # Base Green's Function (G) value
            g_val = np.exp(1j * k * r) * pre_mics_G[i, j]

            # Common spatial terms
            jk_minus_inv_r = 1j * k - inv_r
            
            # Pressure Kernel: Derivative of Green's Function (H)
            h_val = H_sign * g_val * jk_minus_inv_r * r_dot_n
            
            # Integrate Pressure
            sum_p += (g_val * v_surf[j] + h_val * p_surf[j]) * bem_areas[j]
            
            # --- Velocity Kernel Calculations ---
            # Monopole analytical gradient components
            dg_dx = g_val * jk_minus_inv_r * dx
            dg_dy = g_val * jk_minus_inv_r * dy
            dg_dz = g_val * jk_minus_inv_r * dz
            
            # Dipole analytical gradient components (Quadrupole terms)
            quad_term = (-k2 - 3.0 * 1j * k * inv_r + 3.0 * inv_r2) * r_dot_n
            
            dh_dx = H_sign * g_val * (quad_term * dx + jk_minus_inv_r * nx * inv_r)
            dh_dy = H_sign * g_val * (quad_term * dy + jk_minus_inv_r * ny * inv_r)
            dh_dz = H_sign * g_val * (quad_term * dz + jk_minus_inv_r * nz * inv_r)
            
            # Integrate Pressure Gradient components over the element area
            sum_grad_px += (dg_dx * v_surf[j] + dh_dx * p_surf[j]) * bem_areas[j]
            sum_grad_py += (dg_dy * v_surf[j] + dh_dy * p_surf[j]) * bem_areas[j]
            sum_grad_pz += (dg_dz * v_surf[j] + dh_dz * p_surf[j]) * bem_areas[j]
            
        p_mics[i] = sum_p
        
        # Apply Euler's relationship to convert pressure gradient to particle velocity
        v_mics_x[i] = sum_grad_px * inv_rho_omega
        v_mics_y[i] = sum_grad_py * inv_rho_omega
        v_mics_z[i] = sum_grad_pz * inv_rho_omega

    return p_mics, v_mics_x, v_mics_y, v_mics_z
