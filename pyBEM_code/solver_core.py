# The heavy math (Numba-accelerated BEM kernels)

import numpy as np
from numba import njit, prange
from utils import pre_high_order, pre_mid_order, compute_mid_order_contribution, compute_high_order_contribution

@njit(parallel=True, cache=True)
def pre_assembly(element_nodes, centers, areas, normals):
    """
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
        if elem_n_nodes == 3:
            num_gps = 1+3+7
        else:
            num_gps = 1+4+9
        gp_per_element[i] = num_gps
        total_gps += num_gps

    print(f"""
 For ( {n_els} ) BEM elements, found a total of ( {total_gps} ) possible Integration / Gauss Points.""")
    
    # --- STEP 2: Pre-allocate with Exact Size for speed ---
    # These are "Flat Lists" with all GPoints + dtype to save on RAM
    GP_start_idx = np.zeros(n_els, dtype=np.int64)
    offset = 0
    for j in range(n_els):
        GP_start_idx[j] = offset
        offset += gp_per_element[j]

    # Static Influence Matrices
    # Rows = Centroids (n_els), Cols = Every single GP in the system
    G_static_map = np.zeros((n_els, total_gps), dtype=np.float32)
    H_static_map = np.zeros((n_els, total_gps), dtype=np.float32)
    R_map = np.zeros((n_els, total_gps), dtype=np.float32)

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
        
        # Integration options:
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

    return gp_per_element, GP_start_idx, R_map, G_static_map, H_static_map, G_diag_static, H_diag_static

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

            # Thresholds for mid- or high-order GPs based on distance logic
            if max_el_length[j] > order_length:
                order_length = max_el_length[j]
            limit_high = order_length * 3
            limit_mid = order_length * 5

            # Identify element types by total_GPs count from PRE
            # Used to set limits between different integration orders.
            total_gps = gp_per_element[j]
            is_tria = (total_gps == 11) # TRIA: 1 (Cent) + 3 (Mid) + 7 (High)
            # This is where the High-Order slice starts
            high_start_idx = 4 if is_tria else 5   # QUAD: 1 (Cent) + 4 (Mid) + 9 (High)

            if i == j:
                G[i, j] = G_diag_static[j]
                H[i, j] = H_diag_static[j]
            
            elif r_centroid < limit_high:
                # --- HIGH ORDER (Full Integration) ---
                # Sum all GPs for this order (e.g., indices 0 to 13)
                g_sum = 0.0 + 0.0j
                h_sum = 0.0 + 0.0j
                p_start, p_end = high_start_idx, total_gps

                for p_offset in range(p_start, p_end):
                    p_global = start + p_offset # Shift to the correct element's data
                    r_p = R_map[i, p_global]
                    exp_jkr = np.exp(1j * k * r_p)
                    # G logic
                    g_sum += exp_jkr * G_static_map[i, p_offset]
                    # H logic: H_static already contains r_dot_n/r^2
                    h_sum += H_sign * exp_jkr * (1j * k * r_p - 1.0) * H_static_map[i, p_global]

                G[i, j] = g_sum
                H[i, j] = h_sum

            elif r_centroid < limit_mid:
                # --- MID ORDER (Reduced Integration) ---
                g_sum = 0.0 + 0.0j
                h_sum = 0.0 + 0.0j
                p_start, p_end = 1, high_start_idx

                for p_offset in range(p_start, p_end):
                    p_global = start + p_offset # Shift to the correct element's data
                    r_p = R_map[i, p_offset]
                    exp_jkr = np.exp(1j * k * r_p)
                    g_sum += exp_jkr * G_static_map[i, p_offset]
                    h_sum += H_sign * exp_jkr * (1j * k * r_p - 1.0) * H_static_map[i, p_offset]

                G[i, j] = g_sum
                H[i, j] = h_sum
            
            else:
                # --- FAR FIELD (Centroid Only) ---
                # 1. Get the Correct Index for the Centroid (the first GP in the block)
                r_p = R_map[i, start]
                # 2. Phase term
                exp_jkr = np.exp(1j * k * r_p)
                # 3. G: exp(jkr) * static_G (which is weight/4pi*r)
                G[i, j] = exp_jkr * G_static_map[i, start]
                # 4. H
                H[i, j] = H_sign * exp_jkr * (1j * k * r_p - 1.0) * H_static_map[i, start]
    
    return G, H

@njit(parallel=True, cache=True)
def assemble_static(element_nodes, centers, areas, normals, k, H_sign, max_el_length, order_length):
    """
    Computes G and H matrices using element normals:
    - Green's Function Kernel (G-matrix): Gij = exp(jk*r) / 4PI*r
    - Derivative Kernel (H-matrix): Hij = (exp(jk*r) / 4PI*r^2) * (jk*r - 1) * r_dot_n
      Hij = (Gij / r) * (jk*r - 1) * r_dot_n
    centers: (N, 3) array
    areas: (N,) array
    normals: (N, 3) array
    k: wave number (complex or float)
    Accounts for INTERIOR (H_sign=-1) or EXTERIOR (H_sign=1)
    max_el_length: (N,) array with MAX edge size per BEM element. Used for local size high-order integration.
    """
    n_els = len(centers)    # number of BEM elements
    inv_4pi = 1 / (4*np.pi) # 4*pi is a constant in the Green's function denominator
    # Build 2D complex matrices
    G = np.zeros((n_els, n_els), dtype=np.complex128)
    H = np.zeros((n_els, n_els), dtype=np.complex128)

    for i in prange(n_els):
        r_pt = centers[i]
        for j in range(n_els):
            # Vector from source j to receiver i
            r_vec = r_pt - centers[j]
            r = np.linalg.norm(r_vec)
            if max_el_length[j] > order_length:
                order_length = max_el_length[j]
            
            if i == j: # Analytical self-term approximations skipped
                continue
            # All off-diagonal terms benefit from quadrature, especially near-field neighbours.
            elif r < order_length * 3:
                # high-order
                g_val, h_val = compute_high_order_contribution(
                    r_pt, 
                    element_nodes[j], # Array of actual nodal coordinates
                    normals[j], 
                    areas[j], 
                    k, H_sign, inv_4pi
                )
                G[i, j] = g_val
                H[i, j] = h_val
            elif r < order_length * 5:
                # mid-order
                g_val, h_val = compute_mid_order_contribution(
                    r_pt, 
                    element_nodes[j], # Array of actual nodal coordinates
                    normals[j], 
                    areas[j], 
                    k, H_sign, inv_4pi
                )
                G[i, j] = g_val
                H[i, j] = h_val
            else:
                # for k = 0 all exp == 1, so remove?
                exp_jkr = np.exp(1j * k * r)
                # (G-matrix): Free-space 3D Helmholtz Green's function
                g_val = exp_jkr * inv_4pi / r
                G[i, j] = g_val * areas[j]
                
                # (H-matrix) Double Layer: Derivative of G with respect to normal n_j
                r_dot_n = np.dot(r_vec, normals[j]) / r
                H[i, j] = H_sign * (G[i, j]) * ((1j * k) - 1.0 / r) * r_dot_n
    
    # In Numpy axis=0 ==> cols | axis=1 ==> rows
    H_static = np.real(-np.sum(H, axis=1, dtype=np.complex128))
    return H_static

@njit(parallel=True, cache=True)
def assemble_system(element_nodes, centers, areas, normals, k, H_sign, max_el_length, order_length, H_static):
    """
    Computes G and H matrices using element normals:
    - Green's Function Kernel (G-matrix): Gij = exp(jk*r) / 4PI*r
    - Derivative Kernel (H-matrix): Hij = (exp(jk*r) / 4PI*r^2) * (jk*r - 1) * r_dot_n
      Hij = (Gij / r) * (jk*r - 1) * r_dot_n
    centers: (N, 3) array
    areas: (N,) array
    normals: (N, 3) array
    k: wave number (complex or float)
    Accounts for INTERIOR (H_sign=-1) or EXTERIOR (H_sign=1)
    max_el_length: (N,) array with MAX edge size per BEM element. Used for local size high-order integration.
    """
    n_els = len(centers)    # number of BEM elements
    inv_4pi = 1 / (4*np.pi) # 4*pi is a constant in the Green's function denominator
    # Build 2D complex matrices
    G = np.zeros((n_els, n_els), dtype=np.complex128)
    H = np.zeros((n_els, n_els), dtype=np.complex128)

    for i in prange(n_els):
        r_pt = centers[i]
        for j in range(n_els):
            # Vector from source j to receiver i
            r_vec = r_pt - centers[j]
            r = np.linalg.norm(r_vec)
            if max_el_length[j] > order_length:
                order_length = max_el_length[j]
            
            if i == j: # Analytical self-term approximations for diagonal terms
                G[i, j] = np.sqrt(areas[j] / np.pi) * 2.0 * inv_4pi
                # H[i, j] = 0.5  # Jump term for smooth surfaces
                H[i, j] = H_static[j]
            # All off-diagonal terms benefit from quadrature, especially near-field neighbours.
            elif r < order_length * 3:
                # high-order
                g_val, h_val = compute_high_order_contribution(
                    r_pt, 
                    element_nodes[j], # Array of actual nodal coordinates
                    normals[j], 
                    areas[j], 
                    k, H_sign, inv_4pi
                )
                G[i, j] = g_val
                H[i, j] = h_val
            elif r < order_length * 5:
                # mid-order
                g_val, h_val = compute_mid_order_contribution(
                    r_pt, 
                    element_nodes[j], # Array of actual nodal coordinates
                    normals[j], 
                    areas[j], 
                    k, H_sign, inv_4pi
                )
                G[i, j] = g_val
                H[i, j] = h_val
            else:
                exp_jkr = np.exp(1j * k * r)
                # (G-matrix): Free-space 3D Helmholtz Green's function
                g_val = exp_jkr * inv_4pi / r
                G[i, j] = g_val * areas[j]
                
                # (H-matrix) Double Layer: Derivative of G with respect to normal n_j
                r_dot_n = np.dot(r_vec, normals[j]) / r
                H[i, j] = H_sign * (G[i, j]) * ((1j * k) - 1.0 / r) * r_dot_n
    
    return G, H

# @njit(parallel=True)   # it crashes Numba if dictionaries present
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
    
    # Solve the linear system Ax = B
    # Solve for the unknown surface values (usually Pressure)
    bem_solution = np.linalg.solve(A, B)
    
    # Build both PRESS & VELO array results, before Mics calcs & averaged_at_nodes for PV
    p_final, v_final = derive_surface_vectors(bem_solution, bc_map, sorted_bem_ids, rho_omega)
    
    return (p_final, v_final, cond)

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

# Superseeded with the above function for speed
# @njit(parallel=True)
# def calculate_mics(mic_centers, bem_centers, bem_areas, bem_normals, p_surf, v_surf, k, rho_omega, H_sign):
#     """
#     Pass 2: Projects solved surface (BEM) results onto Microphone points.
#     mic_centers: mics nodal coords.
#     bem_centers: BEM elements CoG coords.
#     """
#     num_mics = len(mic_centers)
#     num_surf = len(bem_centers)
#     p_mics = np.zeros(num_mics, dtype=np.complex128)
#     inv_4pi = 1.0 / (4.0 * np.pi)
#     # TODO: when input VELO present, we have 1 too many '* (1j * rho_omega)' on those els.
#     #       to be reviewed in the future.
#     v_surf = v_surf * (1j * rho_omega)

#     for i in prange(num_mics):
#         sum_p = 0.0 + 0.0j
#         for j in range(num_surf):
#             r_vec = mic_centers[i] - bem_centers[j]
#             r = np.linalg.norm(r_vec)

#             # if r < order_length * 0.5:
#             #     # high-order
#             #     g_val, h_val = compute_high_order_contribution(
#             #         mic_centers[i], 
#             #         element_nodes[j], # Array of actual nodal coordinates
#             #         bem_normals[j], 
#             #         bem_areas[j], 
#             #         k, H_sign, inv_4pi
#             #     )
#             # else:
#             # Green's Function (G)
#             g_val = np.exp(1j * k * r) * inv_4pi / r

#             # Derivative of Green's Function (H)
#             # Dot product of (r_vec) and the surface normal / r
#             r_dot_n = np.dot(r_vec, bem_normals[j]) / r
#             h_val = H_sign * g_val * (1j * k - 1.0 / r) * r_dot_n
            
#             # p_mic = G*v + H*p (integrated over area)
#             # TODO: check the sign for interior vs exterior when the time comes.
#             sum_p += (g_val * v_surf[j] + h_val * p_surf[j]) * bem_areas[j]
#         p_mics[i] = sum_p

#     return p_mics

###
### 1st ASSEMBLY implementation for BEM element centers only, no gaussian points;
### to get things off the ground on the complete workflow.
# # @njit(fastmath=True)   # 2 x slower vs parallel=True; both together is same 2 x slower.
# @njit(parallel=True)
# def assemble_system(centers, areas, normals, k, H_sign):
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
#     """
#     n_els = len(centers)   # number of BEM elements
#     # Build 2D complex matrices
#     G = np.zeros((n_els, n_els), dtype=np.complex128)
#     H = np.zeros((n_els, n_els), dtype=np.complex128)
    
#     # 4*pi is a constant in the Green's function denominator
#     inv_4pi = 1 / (4*np.pi)
#     # Factor required to get the resonant freqs & levels right 
#     # for interior analysis; e.g. coarse 1m pipe with changing BCs.
#     # TODO: find out more about kernels, and why this factor is required.
#     TMP_FACTOR = 1.061**0.5
#     # TMP_FACTOR = 1.0
#     # TMP_FACTOR = (10./(3.0*np.pi))**0.5
#     # total_area = np.sum(areas)
#     # TMP_FACTOR = np.exp(6.3e-5 * (n_els - 1))
#     # TMP_FACTOR = np.exp(9.22e-8 * total_area)
#     # # set the sign in Hij depending if exterior or interior BEM
#     # if BEM_TYPE == "INTERIOR": H_sign = -1.0
#     # elif BEM_TYPE == "EXTERIOR": H_sign = 1.0

#     for i in prange(n_els):
#         for j in range(n_els):
#             if i == j: # Analytical self-term approximations for diagonal terms
#                 G[i, j] = inv_4pi * np.sqrt(areas[j])   # length units == G units off diagonal
#                 H[i, j] = 0.5  # Jump term for smooth surfaces
#             else:
#                 # Vector from source j to receiver i
#                 r_vec = centers[i] - centers[j]
#                 r = np.linalg.norm(r_vec)
#                 exp_jkr = np.exp(1j * k * r)
                
#                 # (G-matrix): Free-space 3D Helmholtz Green's function
#                 g_val = TMP_FACTOR * exp_jkr * inv_4pi / r
#                 G[i, j] = g_val * areas[j]
                
#                 # (H-matrix) Double Layer: Derivative of G with respect to normal n_j
#                 r_dot_n = np.dot(r_vec, normals[j]) / r
#                 H[i, j] = H_sign * TMP_FACTOR * (G[i, j]) * ((1j * k) - 1.0 / r) * r_dot_n
#     return G, H

### 
### 1st solve implementation before combined BCs + Amplitude curves support
### 
# # @njit(parallel=True)   # it crashes Numba if dictionaries present
# def solve_bem_system(G, H, bc_map, sorted_bem_ids, rho_omega, log=None):
#     """
#     Re-arranges the BEM system (H*p = G*v) based on Boundary Conditions.
#     Solves Ax=B.
#     Returns the complex pressure for every element.
#     sorted_bem_ids: sorted list of element IDs to ensure index alignment
#     """
#     num_elements = len(sorted_bem_ids)
#     A = np.zeros((num_elements, num_elements), dtype=np.complex128)
#     B = np.zeros(num_elements, dtype=np.complex128)

#     # We MUST use the same index 'j' that corresponds to the matrix columns (0, 1, 2...)
#     # eid is the ACTUAL ID from the .inp (1, 101, 500...)
#     for j, eid in enumerate(sorted_bem_ids):
#         bc = bc_map.get(eid, {})

#         # Case 1: Velocity is known (Vibrating Wall)
#         if 'VELO' in bc:
#             # Velocity is known (v), Pressure (p) is unknown
#             A[:, j] = H[:, j]
#             B += G[:, j] * bc['VELO'] * (1j * rho_omega)
        
#         # Case 2: Pressure is known (Open end / Source)
#         elif 'PRES' in bc:
#             # Pressure is known (p), Velocity (v) is unknown
#             A[:, j] = -G[:, j]
#             # * (1j * rho_omega)
#             B -= H[:, j] * bc['PRES']
        
#         # Case 3: Impedance (Absorbent material)
#         elif 'IMPE' in bc:
#             # Admittance logic: v = p / Z. 
#             # Term: (H - G/Z)*p = 0 -> A_col = H - G/Z, B = 0
#             # Safety check for division by zero
#             z_val = bc['IMPE'] if abs(bc['IMPE']) > 1e-12 else 1e-12
#             A[:, j] = H[:, j] - (G[:, j] / z_val)
#             # A[:, j] = H[:, j] + (G[:, j] / z_val)
        
#         # Case 4: Rigid Wall, v=0 (Default)
#         else:
#             A[:, j] = H[:, j]

#     # --- Solver Feedback ---
#     # Check Matrix Condition Number (good UX)
#     # If this number is too high, the mesh might be bad
#     # if log:
#     #     # Matrix Condition Number tells us if the mesh is 'broken' or math is unstable
#     #     # Note: This can be slow for huge matrices, use sparingly!
#     #     cond = np.linalg.cond(A)
#     #     if cond > 1e12:
#     #         log.write("      WARNING: Matrix is ill-conditioned. Check for overlapping elements!\n")
#     cond = 'N/A'   # turned off as expensive.
    
#     # Solve the linear system Ax = B
#     # Solve for the unknown surface values (usually Pressure)
#     bem_solution = np.linalg.solve(A, B)
    
#     # Build both PRESS & VELO array results, before Mics calcs & averaged_at_nodes for PV
#     p_final, v_final = derive_surface_vectors(bem_solution, bc_map, sorted_bem_ids, rho_omega)
    
#     return (p_final, v_final, cond)