# The heavy math (Numba-accelerated BEM kernels)

import numpy as np
import time
import os
import atexit
from multiprocessing import shared_memory
from numba import njit, prange, set_num_threads
from utils import get_ram, pre_high_order, pre_mid_order, set_hardware_limits, averaged_at_nodes

# This keeps track of all blocks created in this session
_SHM_REGISTRY = {}

def global_shm_cleanup():
    """The 'Janitor' that runs when the script ends or crashes."""
    for name, shm in list(_SHM_REGISTRY.items()):
        try:
            shm.close()
            shm.unlink()
            # print(f" ( i ) Cleaned up SHM: {name}") # Debug only
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

def init_worker(shared_data, threads_per_worker):
    """Runs ONCE per worker process at startup."""
    # from numba import set_num_threads
    # set_num_threads(threads_per_worker)
    # set_hardware_limits(threads_per_worker)
    global _worker_context, np
    import numpy as np
    rebuilt_dict = rebuild_from_shm(shared_data)
    _worker_context = rebuilt_dict

###
### This is for parallel solve of Freqs, based on machine resources; i.e. automated.
###
# @njit(cache=True) do NOT use here, overkill; keep inside functions with njit as they are
def frequency_worker(f, bc_map, sorted_bem_ids, threads_per_worker):
    """
    Independent worker function. Runs for EVERY frequency.
    static_data: dict of all BEM/Physics constants
    """
    # import numpy as np
    # Pull the data from the worker's local 'memory vault'
    # No pickling or data transfer happens here!
    static_data = _worker_context
    amps = static_data.get('amplitudes') # Use .get() to avoid KeyErrors
    damping_f = static_data['damping']

    # PRE-processing if any amplitude curves present
    # A. Resolve Damping
    if amps and 'AMP_damp' in damping_f:
        curve = amps[damping_f['AMP_damp']]
        damping_f = np.interp(f, curve[:, 0], curve[:, 1]) * damping_f['value']
    else:
        damping_f = damping_f['value']

    # B. Resolve BCs (The "Clean & Scale" Pass)
    resolved_bcs = {}
    
    for eid, info in bc_map.items():
        # Create a local numeric-only dict for this element
        resolved_bcs[eid] = res = {}
        
        # We only look for the three core types
        for bc_type in ['PRES', 'VELO', 'IMPE']:
            if bc_type in info:
                # Get base value: e.g., (2e-11 + 1e-11j)
                base_val = info[bc_type]
                v_real = base_val.real
                v_imag = base_val.imag

                # --- Scale REAL channel ---
                amp_r_key = f'{bc_type}_AMP_real'
                if amps and amp_r_key in info:
                    c_name = info[amp_r_key]
                    if c_name in amps:
                        scale_r = np.interp(f, amps[c_name][:, 0], amps[c_name][:, 1])
                        v_real *= scale_r

                # --- Scale IMAGINARY channel ---
                amp_i_key = f'{bc_type}_AMP_imag'
                if amps and amp_i_key in info:
                    c_name = info[amp_i_key]
                    if c_name in amps:
                        scale_i = np.interp(f, amps[c_name][:, 0], amps[c_name][:, 1])
                        v_imag *= scale_i

                # Recombine into a complex number for the solver
                res[bc_type] = complex(v_real, v_imag)

    # 1. Physics setup
    omega = 2.0 * np.pi * f
    k = omega / static_data['c']
    if damping_f != 0:
        k = k * (1 - (1j * damping_f * 0.5))
    rho_omega = static_data['rho'] * omega

    # 2. Assembly (The heavy lifting)
    # Using main_assembly function inside the worker
    t_asm_0 = time.time()
    # set_hardware_limits(threads_per_worker)
    # set_num_threads(threads_per_worker)
    G_bem, H_bem = main_assembly(
        static_data['gp_per_element'], static_data['GP_start_idx'],
        static_data['R_map'], static_data['G_static_map'], 
        static_data['H_static_map'], static_data['G_diag_static'], 
        static_data['H_diag_static'], k, static_data['H_sign'], 
        static_data['max_el_length'], static_data['order_length']
    )
    t_assembly = time.time() - t_asm_0
    # 3. Solve
    t_slv_0 = time.time()
    # p_surf, v_surf, cond = solve_bem_system(G_bem, H_bem, resolved_bcs, sorted_bem_ids, rho_omega)
    bem_solution, cond = solve_bem_system(G_bem, H_bem, resolved_bcs, sorted_bem_ids, rho_omega)
    # Build both PRESS & VELO array results, before Mics calcs & averaged_at_nodes for PV
    p_surf, v_surf = derive_surface_vectors(bem_solution, bc_map, sorted_bem_ids, rho_omega)
    solve_RAM = get_ram()
    t_slv_1 = time.time()
    # 4. Post-Process (Mics + Averaging)
    # MICS calcs
    if static_data['sorted_mics_els']:
        # set_num_threads(threads_per_worker)
        p_mics = calculate_mics(
            static_data['pre_mics_G'], static_data['pre_mics_H'], static_data['pre_mics_R'], 
            static_data['num_mics'], static_data['bem_areas'], 
            p_surf, v_surf, k, rho_omega, static_data['H_sign']
        )
        mics_nodes = static_data['sorted_mics_nodes']
    else:
        mics_nodes = None
        p_mics = None
    t_slv_2 = time.time()
    
    # This ensures we return the final nodal_pressures array
    t_avg_0 = time.time()
    nodal_pressures = averaged_at_nodes(
        static_data['sorted_nodes'], static_data['sorted_bem_els'], p_surf, 
        static_data['bem_areas'], mics_nodes, p_mics) 
    t_avg_1 = time.time()
    
    metadata = {
        't_assembly': t_assembly,
        't_solve_bem': t_slv_1 - t_slv_0,
        't_solve_mics': t_slv_2 - t_slv_1,
        't_avrg_nodes': t_avg_1 - t_avg_0,
        'solve_RAM': solve_RAM
    }
    
    return f, nodal_pressures, metadata

@njit(parallel=True, cache=True)
def pre_assembly(element_nodes, centers, areas, normals, R_map, G_static_map, H_static_map, total_gps):
    """
    It takes pre-allocated Shared Memory large arrays to avoid (RAM) duplication crashes.
    Pre-Computes G and H (static, k=0) matrices using the input mesh:
    - Green's Function Kernel (G-matrix): Gij = 1 / (4PI*r)
    - Derivative Kernel (H-matrix): Hij = Gij / r * r_dot_n
    It also pre-computes fast arrays with all GPs info for fast assembly in the solve.
    element_nodes: BEM elem nodal coords
    centers: (N, 3) array
    areas: (N,) array
    normals: (N, 3) array
    """
    n_els = len(centers)    # number of BEM elements
    inv_4pi = 1 / (4*np.pi) # 4*pi is a constant in the Green's function denominator

    # --- STEP 1: Calculate Total GPs ---
    total_gps = 0
    gp_per_element = np.zeros(n_els, dtype=np.int64)  # Force i64 for numba to work
    # Here we aim at all possible GPs for centroid, mid- and high-order integration options.
    for i in range(n_els):
        elem_n_nodes = len(element_nodes[i])
        num_gps = 11 if elem_n_nodes == 3 else 14
        gp_per_element[i] = num_gps
        total_gps += num_gps

    print(f""" For ( {n_els} ) BEM elements, found a total of ( {total_gps} ) possible Integration / Gauss Points.""")
    
    # --- STEP 2: Pre-allocate with Exact Size for speed ---
    # These are "Flat Lists" with all GPoints + dtype to save on RAM
    GP_start_idx = np.zeros(n_els, dtype=np.int64)
    offset = 0
    for j in range(n_els):
        GP_start_idx[j] = offset
        offset += gp_per_element[j]

    cursor = 0
    for j in prange(n_els):      # Parallel Loop over SOURCE elements
        n_pts = gp_per_element[j]
        cursor = GP_start_idx[j]
        nj = normals[j]          # THIS element normal
        nodes = element_nodes[j] # Node coords for THIS element
        area = areas[j]          # Area for THIS element
        # temp arrays of Flat Lists
        pts_coords = np.zeros((n_pts, 3), dtype=np.float64)
        pts_weights = np.zeros(n_pts, dtype=np.float64)
        # Calculate ALL possible GPs(coords, weights) per SOURCE element
        if n_pts == 11: # TRIAs: 1 + 3 + 7 GPs levels
            mid_gp_idx = 3 + 1
        elif n_pts == 14: # QUADs: 1 + 4 + 9 GPs levels
            mid_gp_idx = 4 + 1
        
        # Integration options overal mapping:
        # used later in solver as a function of distance receiver --> source elements:
        # CENTROID (1pnt):
        pts_coords[0] = centers[j]
        pts_weights[0] = 1 * area
        # MID-order (3pnts or 4pnts):
        pts, weights = pre_mid_order(nodes, area)
        pts_coords[1:mid_gp_idx] = pts
        pts_weights[1:mid_gp_idx] = weights
        # HIGH-order (7pnts or 9pnts):
        pts, weights = pre_high_order(nodes, area)
        pts_coords[mid_gp_idx:] = pts
        pts_weights[mid_gp_idx:] = weights

        # --- STEP 3: Fill the "Flat Lists" ---
        for i in range(n_els): # Loop over RECEIVER elements
            r_pt = centers[i]
            # Explicit loop is "The Numba Way" - no axis issues here
            for p in range(n_pts):
                # 1. Calculate the vector from GP to Receiver Center
                rx = r_pt[0] - pts_coords[p, 0]
                ry = r_pt[1] - pts_coords[p, 1]
                rz = r_pt[2] - pts_coords[p, 2]
                # 2. Distance math
                r2 = rx*rx + ry*ry + rz*rz
                if r2 < 1e-18: # Safety for self-term
                    continue
                r = np.sqrt(r2)
                idx = cursor + p
                R_map[i, idx] = r

                # Fill Static Influence Matrices
                # 3. Bake the G Static Map: G = weight / (4 * pi * r)
                g_base = pts_weights[p] * inv_4pi / r
                G_static_map[i, idx] = g_base

                # 4. Bake the H Static Map: H = G_static * (dot / r^2)
                dot = rx*nj[0] + ry*nj[1] + rz*nj[2]
                H_static_map[i, idx] = g_base * (dot / r2)

    # Calc the static [H] diagonal self terms
    H_diag_static = np.zeros(n_els, dtype=np.float64)
    G_diag_static = np.zeros(n_els, dtype=np.float64)
    for i in range(n_els):
        row_sum = 0.0
        G_diag_static[i] = np.sqrt(areas[i] / np.pi) * 2.0 * inv_4pi
        for j in range(n_els):
            if i == j: 
                continue # Skip the pre-diagonals in ther sum
            
            n_pts = gp_per_element[j]
            if n_pts == 11:   # TRIAs: 1 + 3 + 7
                start = GP_start_idx[j] + 4
                n_pts = 7
            elif n_pts == 14: # QUADs: 1 + 4 + 9
                start = GP_start_idx[j] + 5
                n_pts = 9
            # SUM ONLY HIGH ORDER from the static pre-baked H values for this source element
            for p in range(start, start + n_pts):
                row_sum += H_static_map[i, p]

        # The BEM Diagonal Balance
        H_diag_static[i] = -row_sum

    return gp_per_element, GP_start_idx, G_diag_static, H_diag_static
    # return gp_per_element, GP_start_idx, R_map, G_static_map, H_static_map, G_diag_static, H_diag_static

# OLD before Shared Memory approach, without the need for a large pagefile.
# @njit(parallel=True, cache=True)
# def pre_assembly(element_nodes, centers, areas, normals):
#     """
#     Pre-Computes G and H (static, k=0) matrices using the input mesh:
#     - Green's Function Kernel (G-matrix): Gij = 1 / (4PI*r)
#     - Derivative Kernel (H-matrix): Hij = Gij / r * r_dot_n
#     It also pre-computes fast arrays with all GPs info for fast assembly in the solve.
#     element_nodes: BEM elem nodal coords
#     centers: (N, 3) array
#     areas: (N,) array
#     normals: (N, 3) array
#     """
#     n_els = len(centers)    # number of BEM elements
#     inv_4pi = 1 / (4*np.pi) # 4*pi is a constant in the Green's function denominator

#     # --- STEP 1: Calculate Total GPs ---
#     total_gps = 0
#     gp_per_element = np.zeros(n_els, dtype=np.int64)  # Force i64 for numba to work
#     # Here we aim at all possible GPs for centroid, mid- and high-order integration options.
#     for i in range(n_els):
#         elem_n_nodes = len(element_nodes[i])
#         if elem_n_nodes == 3:
#             num_gps = 1+3+7
#         else:
#             num_gps = 1+4+9
#         gp_per_element[i] = num_gps
#         total_gps += num_gps

#     print(f""" For ( {n_els} ) BEM elements, found a total of ( {total_gps} ) possible Integration / Gauss Points.""")
    
#     # --- STEP 2: Pre-allocate with Exact Size for speed ---
#     # These are "Flat Lists" with all GPoints + dtype to save on RAM
#     GP_start_idx = np.zeros(n_els, dtype=np.int64)
#     offset = 0
#     for j in range(n_els):
#         GP_start_idx[j] = offset
#         offset += gp_per_element[j]

#     # Static Influence Matrices
#     # Rows = Centroids (n_els), Cols = Every single GP in the system
#     G_static_map = np.zeros((n_els, total_gps), dtype=np.float32)
#     H_static_map = np.zeros((n_els, total_gps), dtype=np.float32)
#     R_map = np.zeros((n_els, total_gps), dtype=np.float32)

#     cursor = 0
#     for j in prange(n_els):      # Parallel Loop over SOURCE elements
#         n_pts = gp_per_element[j]
#         cursor = GP_start_idx[j]
#         nj = normals[j]          # THIS element normal
#         nodes = element_nodes[j] # Node coords for THIS element
#         area = areas[j]          # Area for THIS element
#         # temp arrays of Flat Lists
#         pts_coords = np.zeros((n_pts, 3), dtype=np.float64)
#         pts_weights = np.zeros(n_pts, dtype=np.float64)
#         # Calculate ALL possible GPs(coords, weights) per SOURCE element
#         if n_pts == 11: # TRIAs: 1 + 3 + 7 GPs levels
#             mid_gp_idx = 3 + 1
#         elif n_pts == 14: # QUADs: 1 + 4 + 9 GPs levels
#             mid_gp_idx = 4 + 1
        
#         # Integration options overal mapping:
#         # used later in solver as a function of distance receiver --> source elements:
#         # CENTROID (1pnt):
#         pts_coords[0] = centers[j]
#         pts_weights[0] = 1 * area
#         # MID-order (3pnts or 4pnts):
#         pts, weights = pre_mid_order(nodes, area)
#         pts_coords[1:mid_gp_idx] = pts
#         pts_weights[1:mid_gp_idx] = weights
#         # HIGH-order (7pnts or 9pnts):
#         pts, weights = pre_high_order(nodes, area)
#         pts_coords[mid_gp_idx:] = pts
#         pts_weights[mid_gp_idx:] = weights

#         # --- STEP 3: Fill the "Flat Lists" ---
#         for i in range(n_els): # Loop over RECEIVER elements
#             r_pt = centers[i]
#             # Explicit loop is "The Numba Way" - no axis issues here
#             for p in range(n_pts):
#                 # 1. Calculate the vector from GP to Receiver Center
#                 rx = r_pt[0] - pts_coords[p, 0]
#                 ry = r_pt[1] - pts_coords[p, 1]
#                 rz = r_pt[2] - pts_coords[p, 2]
#                 # 2. Distance math
#                 r2 = rx*rx + ry*ry + rz*rz
#                 if r2 < 1e-18: # Safety for self-term
#                     continue
#                 r = np.sqrt(r2)
#                 idx = cursor + p
#                 R_map[i, idx] = r

#                 # Fill Static Influence Matrices
#                 # 3. Bake the G Static Map: G = weight / (4 * pi * r)
#                 g_base = pts_weights[p] * inv_4pi / r
#                 G_static_map[i, idx] = g_base

#                 # 4. Bake the H Static Map: H = G_static * (dot / r^2)
#                 dot = rx*nj[0] + ry*nj[1] + rz*nj[2]
#                 H_static_map[i, idx] = g_base * (dot / r2)

#     # Calc the static [H] diagonal self terms
#     H_diag_static = np.zeros(n_els, dtype=np.float64)
#     G_diag_static = np.zeros(n_els, dtype=np.float64)
#     for i in range(n_els):
#         row_sum = 0.0
#         G_diag_static[i] = np.sqrt(areas[i] / np.pi) * 2.0 * inv_4pi
#         for j in range(n_els):
#             if i == j: 
#                 continue # Skip the pre-diagonals in ther sum
            
#             n_pts = gp_per_element[j]
#             if n_pts == 11:   # TRIAs: 1 + 3 + 7
#                 start = GP_start_idx[j] + 4
#                 n_pts = 7
#             elif n_pts == 14: # QUADs: 1 + 4 + 9
#                 start = GP_start_idx[j] + 5
#                 n_pts = 9
#             # SUM ONLY HIGH ORDER from the static pre-baked H values for this source element
#             for p in range(start, start + n_pts):
#                 row_sum += H_static_map[i, p]

#         # The BEM Diagonal Balance
#         H_diag_static[i] = -row_sum

#     return gp_per_element, GP_start_idx, R_map, G_static_map, H_static_map, G_diag_static, H_diag_static

# @njit(parallel=True, cache=True, nogil=True)
@njit(parallel=True, cache=True)
def main_assembly(gp_per_element, GP_start_idx, R_map, G_static_map, H_static_map, G_diag_static, H_diag_static, k, H_sign, max_el_length, order_length):
    """
    Computes G and H matrices PRE static ones + dynamic terms:
    - Green's Function Kernel (G-matrix): Gij = exp(jk*r) * G_static
    - Derivative Kernel (H-matrix): Hij = H_sign * exp(jk*r) * (jk*r - 1) * H_static
    k: wave number (complex or float)
    Accounts for INTERIOR (H_sign=-1) or EXTERIOR (H_sign=1)
    max_el_length: (N,) array with MAX edge size per BEM element. Used for local size high-order integration.
    """
    n_els = len(H_diag_static)    # number of BEM elements

    # Build 2D complex matrices
    G = np.zeros((n_els, n_els), dtype=np.complex128)
    H = np.zeros((n_els, n_els), dtype=np.complex128)

    for i in prange(n_els):
        for j in range(n_els):  # from source j to receiver i
            start = GP_start_idx[j]
            # We use the centroid (the first point in the block) for the distance check
            r_centroid = R_map[i, start]

            # Identify element types by total_GPs count from PRE
            # Used to set limits between different integration orders.
            total_gps = gp_per_element[j]
            is_tria = (total_gps == 11) # TRIA: 1 (Cent) + 3 (Mid) + 7 (High)
            # This is where the High-Order slice starts
            high_start_idx = 4 if is_tria else 5   # QUAD: 1 (Cent) + 4 (Mid) + 9 (High)

            # Thresholds for mid- or high-order GPs based on distance logic
            # Local thresholds to avoid parallel race conditions
            limit_high = order_length * 3
            limit_mid = order_length * 5

            if i == j:
                G[i, j] = G_diag_static[j]
                H[i, j] = H_diag_static[j]
            
            else:
                # Set the integration slice
                if r_centroid < limit_high:
                    p_start, p_end = high_start_idx, total_gps
                elif r_centroid < limit_mid:
                    p_start, p_end = 1, high_start_idx
                else:
                    p_start, p_end = 0, 1

                g_temp = 0.0 + 0.0j
                h_temp = 0.0 + 0.0j
            
                for p_offset in range(p_start, p_end):
                    p_global = start + p_offset

                    r_p = R_map[i, p_global]
                    exp_jkr = np.exp(1j * k * r_p)

                    # Fetch pre-baked static terms using global index
                    gs = G_static_map[i, p_global]
                    hs = H_static_map[i, p_global]

                    g_temp += exp_jkr * gs
                    # Using (1 - jkr) convention to match standard exterior/interior derivatives
                    h_temp += H_sign * exp_jkr * (1j * k * r_p - 1.0) * hs

                G[i, j] = g_temp
                H[i, j] = h_temp
    return G, H

### OLD 1st solve, do NOT use, it worked, but very slow vs above ones.
# @njit(parallel=True, cache=True)
# def assemble_static(element_nodes, centers, areas, normals, k, H_sign, max_el_length, order_length):
#     """
#     Computes G and H matrices using element normals:
#     - Green's Function Kernel (G-matrix): Gij = exp(jk*r) / 4PI*r
#     - Derivative Kernel (H-matrix): Hij = (exp(jk*r) / 4PI*r^2) * (jk*r - 1) * r_dot_n
#       Hij = (Gij / r) * (jk*r - 1) * r_dot_n
#     centers: (N, 3) array
#     areas: (N,) array
#     normals: (N, 3) array
#     k: wave number (complex or float)
#     Accounts for INTERIOR (H_sign=-1) or EXTERIOR (H_sign=1)
#     max_el_length: (N,) array with MAX edge size per BEM element. Used for local size high-order integration.
#     """
#     n_els = len(centers)    # number of BEM elements
#     inv_4pi = 1 / (4*np.pi) # 4*pi is a constant in the Green's function denominator
#     # Build 2D complex matrices
#     G = np.zeros((n_els, n_els), dtype=np.complex128)
#     H = np.zeros((n_els, n_els), dtype=np.complex128)

#     for i in prange(n_els):
#         r_pt = centers[i]
#         for j in range(n_els):
#             # Vector from source j to receiver i
#             r_vec = r_pt - centers[j]
#             r = np.linalg.norm(r_vec)
#             if max_el_length[j] > order_length:
#                 order_length = max_el_length[j]
            
#             if i == j: # Analytical self-term approximations skipped
#                 continue
#             # All off-diagonal terms benefit from quadrature, especially near-field neighbours.
#             elif r < order_length * 3:
#                 # high-order
#                 g_val, h_val = compute_high_order_contribution(
#                     r_pt, 
#                     element_nodes[j], # Array of actual nodal coordinates
#                     normals[j], 
#                     areas[j], 
#                     k, H_sign, inv_4pi
#                 )
#                 G[i, j] = g_val
#                 H[i, j] = h_val
#             elif r < order_length * 5:
#                 # mid-order
#                 g_val, h_val = compute_mid_order_contribution(
#                     r_pt, 
#                     element_nodes[j], # Array of actual nodal coordinates
#                     normals[j], 
#                     areas[j], 
#                     k, H_sign, inv_4pi
#                 )
#                 G[i, j] = g_val
#                 H[i, j] = h_val
#             else:
#                 # for k = 0 all exp == 1, so remove?
#                 exp_jkr = np.exp(1j * k * r)
#                 # (G-matrix): Free-space 3D Helmholtz Green's function
#                 g_val = exp_jkr * inv_4pi / r
#                 G[i, j] = g_val * areas[j]
                
#                 # (H-matrix) Double Layer: Derivative of G with respect to normal n_j
#                 r_dot_n = np.dot(r_vec, normals[j]) / r
#                 H[i, j] = H_sign * (G[i, j]) * ((1j * k) - 1.0 / r) * r_dot_n
    
#     # In Numpy axis=0 ==> cols | axis=1 ==> rows
#     H_static = np.real(-np.sum(H, axis=1, dtype=np.complex128))
#     return H_static

# @njit(parallel=True, cache=True)
# def assemble_system(element_nodes, centers, areas, normals, k, H_sign, max_el_length, order_length, H_static):
#     """
#     Computes G and H matrices using element normals:
#     - Green's Function Kernel (G-matrix): Gij = exp(jk*r) / 4PI*r
#     - Derivative Kernel (H-matrix): Hij = (exp(jk*r) / 4PI*r^2) * (jk*r - 1) * r_dot_n
#       Hij = (Gij / r) * (jk*r - 1) * r_dot_n
#     centers: (N, 3) array
#     areas: (N,) array
#     normals: (N, 3) array
#     k: wave number (complex or float)
#     Accounts for INTERIOR (H_sign=-1) or EXTERIOR (H_sign=1)
#     max_el_length: (N,) array with MAX edge size per BEM element. Used for local size high-order integration.
#     """
#     n_els = len(centers)    # number of BEM elements
#     inv_4pi = 1 / (4*np.pi) # 4*pi is a constant in the Green's function denominator
#     # Build 2D complex matrices
#     G = np.zeros((n_els, n_els), dtype=np.complex128)
#     H = np.zeros((n_els, n_els), dtype=np.complex128)

#     for i in prange(n_els):
#         r_pt = centers[i]
#         for j in range(n_els):
#             # Vector from source j to receiver i
#             r_vec = r_pt - centers[j]
#             r = np.linalg.norm(r_vec)
#             if max_el_length[j] > order_length:
#                 order_length = max_el_length[j]
            
#             if i == j: # Analytical self-term approximations for diagonal terms
#                 G[i, j] = np.sqrt(areas[j] / np.pi) * 2.0 * inv_4pi
#                 # H[i, j] = 0.5  # Jump term for smooth surfaces
#                 H[i, j] = H_static[j]
#             # All off-diagonal terms benefit from quadrature, especially near-field neighbours.
#             elif r < order_length * 3:
#                 # high-order
#                 g_val, h_val = compute_high_order_contribution(
#                     r_pt, 
#                     element_nodes[j], # Array of actual nodal coordinates
#                     normals[j], 
#                     areas[j], 
#                     k, H_sign, inv_4pi
#                 )
#                 G[i, j] = g_val
#                 H[i, j] = h_val
#             elif r < order_length * 5:
#                 # mid-order
#                 g_val, h_val = compute_mid_order_contribution(
#                     r_pt, 
#                     element_nodes[j], # Array of actual nodal coordinates
#                     normals[j], 
#                     areas[j], 
#                     k, H_sign, inv_4pi
#                 )
#                 G[i, j] = g_val
#                 H[i, j] = h_val
#             else:
#                 exp_jkr = np.exp(1j * k * r)
#                 # (G-matrix): Free-space 3D Helmholtz Green's function
#                 g_val = exp_jkr * inv_4pi / r
#                 G[i, j] = g_val * areas[j]
                
#                 # (H-matrix) Double Layer: Derivative of G with respect to normal n_j
#                 r_dot_n = np.dot(r_vec, normals[j]) / r
#                 H[i, j] = H_sign * (G[i, j]) * ((1j * k) - 1.0 / r) * r_dot_n
    
#     return G, H

# @njit(parallel=True)   # it crashes Numba if dictionaries present
# @njit(nogil=True)
def solve_bem_system(G, H, bc_map, sorted_bem_ids, rho_omega, log=None):
    """
    Re-arranges the BEM system (H*p = G*v) based on Boundary Conditions.
    Solves Ax=B.
    Returns the complex pressure for every element.
    sorted_bem_ids: sorted list of element IDs to ensure index alignment
    """
    num_elements = len(sorted_bem_ids)
    A = np.zeros((num_elements, num_elements), dtype=np.complex128)
    B = np.zeros(num_elements, dtype=np.complex128)

    # We MUST use the same index 'j' that corresponds to the matrix columns (0, 1, 2...)
    # eid is the ACTUAL ID from the .inp (1, 101, 500...)
    for j, eid in enumerate(sorted_bem_ids):
        bc = bc_map.get(eid, {})

        # Case 1: Velocity is known (Vibrating Wall - Neumann BC)
        if 'VELO' in bc:
            # Velocity is known (v), Pressure (p) is unknown
            A[:, j] += H[:, j]
            B -= G[:, j] * bc['VELO'] * (1j * rho_omega)
            # Case 1.1 Simultaneous VELO + IMPE (Robin BC)
            if 'IMPE' in bc:
                # Admittance logic: v = p / Z. 
                # Term: (H - G/Z)*p = 0 -> A_col = H - G/Z, B = 0
                # Safety check for division by zero
                z_val = bc['IMPE'] if abs(bc['IMPE']) > 1e-12 else 1e-12
                # Units must match: H is unitless, G is [L], so we need [1/L]
                # (i * omega * rho) / Z  has units of [1/L]
                A[:, j] += G[:, j] * (1j * rho_omega / z_val)
        
        # Case 2: Pressure is known (Open end / Source) - (Dirichlet BC)
        elif 'PRES' in bc:
            # Pressure is known (p), Velocity (v) is unknown
            A[:, j] -= G[:, j] * (1j * rho_omega)
            B -= H[:, j] * bc['PRES']
        
        # Case 3: Impedance (Absorbent material)
        elif 'IMPE' in bc and 'VELO' not in bc:
            # Admittance logic: v = p / Z. 
            # Term: (H - G/Z)*p = 0 -> A_col = H - G/Z, B = 0
            # Safety check for division by zero
            z_val = bc['IMPE'] if abs(bc['IMPE']) > 1e-12 else 1e-12
            # Units must match: H is unitless, G is [L], so we need [1/L]
            # (i * omega * rho) / Z  has units of [1/L]
            A[:, j] += H[:, j] + (G[:, j] * (1j * rho_omega / z_val))
        
        # Case 4: Rigid Wall, v=0 (Default)
        else:
            A[:, j] += H[:, j]

    # --- Solver Feedback ---
    # Check Matrix Condition Number (good UX)
    # If this number is too high, the mesh might be bad
    # if log:
    #     # Matrix Condition Number tells us if the mesh is 'broken' or math is unstable
    #     # Note: This can be slow for huge matrices, use sparingly!
    #     cond = np.linalg.cond(A)
    #     if cond > 1e12:
    #         log.write("      WARNING: Matrix is ill-conditioned. Check for overlapping elements!\n")
    cond = 'N/A'   # turned off as expensive.
    
    # Solve the linear system Ax = B for the unknown surface values (usually Pressure)
    bem_solution = np.linalg.solve(A, B)
    
    # # Build both PRESS & VELO array results, before Mics calcs & averaged_at_nodes for PV
    # p_final, v_final = derive_surface_vectors(bem_solution, bc_map, sorted_bem_ids, rho_omega)

    # del G, H, A, B, bem_solution
    
    return (bem_solution, cond)
    # return (p_final, v_final, cond)

# @njit   # it crashes Numba if dictionaries present
def derive_surface_vectors(p_sol, bc_map, sorted_bem_ids, rho_omega):
    """
    Reconstructs the full pressure and velocity vectors for the surface.
    p_sol: The unknown values returned by the solver.
    bc_map: The dictionary of boundary conditions.
    bem_ids: The sorted list of element IDs used in the matrix.
    """
    num_elements = len(sorted_bem_ids)
    p_final = np.zeros(num_elements, dtype=np.complex128)
    v_final = np.zeros(num_elements, dtype=np.complex128)

    for j, eid in enumerate(sorted_bem_ids):
        bc = bc_map.get(eid, {})
        
        if 'VELO' in bc:
            # We knew Velocity, p_sol gave us Pressure
            p_final[j] = p_sol[j]
            v_final[j] = bc['VELO']
            # * (1j * rho_omega)   # Do NOT use it here, it's in solve_bem_system.
            if 'IMPE' in bc:
                # We solved for Pressure, Velocity is p/Z
                z_val = bc['IMPE'] if abs(bc['IMPE']) > 1e-12 else 1e-12
                v_final[j] += p_sol[j] / z_val
            
        elif 'PRES' in bc:
            # We knew Pressure, p_sol gave us Velocity
            p_final[j] = bc['PRES']
            v_final[j] = p_sol[j]
            
        elif 'IMPE' in bc and 'VELO' not in bc:
            # We solved for Pressure, Velocity is p/Z
            p_final[j] = p_sol[j]
            z_val = bc['IMPE'] if abs(bc['IMPE']) > 1e-12 else 1e-12
            v_final[j] = p_sol[j] / z_val
            
        else:
            # Rigid wall: v=0, p_sol gave us Pressure
            p_final[j] = p_sol[j]
            v_final[j] = 0.0 + 0.0j
            
    return p_final, v_final

@njit(parallel=True, cache=True)
def pre_mics(mic_centers, bem_centers, bem_normals):
    """
    Pass 2: Projects solved surface (BEM) results onto Microphone points.
    mic_centers: mics nodal coords.
    bem_centers: BEM elements CoG coords.
    """
    num_mics = len(mic_centers)
    num_surf = len(bem_centers)
    pre_mics_G = np.zeros((num_mics, num_surf), dtype=np.float64)
    pre_mics_H = np.zeros((num_mics, num_surf), dtype=np.float64)
    pre_mics_R = np.zeros((num_mics, num_surf), dtype=np.float32)
    inv_4pi = 1.0 / (4.0 * np.pi)

    for i in prange(num_mics):
        for j in range(num_surf):
            r_vec = mic_centers[i] - bem_centers[j]
            r = np.linalg.norm(r_vec)
            pre_mics_R[i, j] = r

            # Green's Function (G) 1/r
            pre_mics_G[i, j] = inv_4pi / r

            # Dot product of (r_vec) and the surface normal / r
            r_dot_n = np.dot(r_vec, bem_normals[j]) / r
            pre_mics_H[i, j] = r_dot_n

    return pre_mics_G, pre_mics_H, pre_mics_R, num_mics

@njit(parallel=True, cache=True)
def calculate_mics(pre_mics_G, pre_mics_H, pre_mics_R, num_mics, bem_areas, p_surf, v_surf, k, rho_omega, H_sign):
    """
    Pass 2: Projects solved surface (BEM) results onto Microphone points.
    mic_centers: mics nodal coords.
    bem_centers: BEM elements CoG coords.
    """
    # num_mics = len(mic_centers)
    num_surf = len(bem_areas)
    p_mics = np.zeros(num_mics, dtype=np.complex128)
    # inv_4pi = 1.0 / (4.0 * np.pi)
    # TODO: when input VELO present, we have 1 too many '* (1j * rho_omega)' on those els.
    #       to be reviewed in the future.
    v_surf = v_surf * (1j * rho_omega)

    for i in prange(num_mics):
        sum_p = 0.0 + 0.0j
        for j in range(num_surf):
            # r_vec = mic_centers[i] - bem_centers[j]
            # r = np.linalg.norm(r_vec)

            # Green's Function (G)
            g_val = np.exp(1j * k * pre_mics_R[i, j]) * pre_mics_G[i, j]

            # Derivative of Green's Function (H)
            # Dot product of (r_vec) and the surface normal / r
            # r_dot_n = np.dot(r_vec, bem_normals[j]) / r
            h_val = H_sign * g_val * (1j * k - 1.0 / pre_mics_R[i, j]) * pre_mics_H[i, j]
            
            # p_mic = G*v + H*p (integrated over area)
            # TODO: check the sign for interior vs exterior when the time comes.
            sum_p += (g_val * v_surf[j] + h_val * p_surf[j]) * bem_areas[j]
        p_mics[i] = sum_p

    return p_mics
