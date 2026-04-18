# The heavy math (Numba-accelerated BEM kernels)

import numpy as np
from numba import njit, prange
from utils import compute_mid_order_contribution, compute_high_order_contribution

# @njit(parallel=True)
def assemble_static(element_nodes, centers, areas, normals, k, H_sign, hi_order_length):
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

    for i in range(n_els):
        r_pt = centers[i]
        for j in range(n_els):
            # Vector from source j to receiver i
            r_vec = r_pt - centers[j]
            r = np.linalg.norm(r_vec)
            if i == j: # Analytical self-term approximations for diagonal terms
                continue
            #     G[i, j] = np.sqrt(areas[j] / np.pi) * 2.0 * inv_4pi
            #     H[i, j] = 0.5  # Jump term for smooth surfaces
            # All off-diagonal terms benefit from quadrature, especially near-field neighbours.
            # elif r < max_el_length[j] * 3:
            elif r < hi_order_length * 3:
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
            # elif r < max_el_length[j] * 5:
            elif r < hi_order_length * 5:
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
    H_static = np.real(-np.sum(H, axis=1))
    return H_static

@njit(parallel=True)
def assemble_system(element_nodes, centers, areas, normals, k, H_sign, max_el_length, hi_order_length, H_static):
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
            if i == j: # Analytical self-term approximations for diagonal terms
                G[i, j] = np.sqrt(areas[j] / np.pi) * 2.0 * inv_4pi
                # H[i, j] = 0.5  # Jump term for smooth surfaces
                H[i, j] = H_static[j]
            # All off-diagonal terms benefit from quadrature, especially near-field neighbours.
            # elif r < max_el_length[j] * 3:
            elif r < hi_order_length * 3:
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
            # elif r < max_el_length[j] * 5:
            elif r < hi_order_length * 5:
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
            A[:, j] = H[:, j]
            B -= G[:, j] * bc['VELO'] * (1j * rho_omega)
            # 1.1 Simultaneous VELO + IMPE (Robin BC)
            if 'IMPE' in bc:
                # Admittance logic: v = p / Z. 
                # Term: (H - G/Z)*p = 0 -> A_col = H - G/Z, B = 0
                # Safety check for division by zero
                z_val = bc['IMPE'] if abs(bc['IMPE']) > 1e-12 else 1e-12
                # Units must match: H is unitless, G is [L], so we need [1/L]
                # (i * omega * rho) / Z  has units of [1/L]
                A[:, j] -= G[:, j] * (1j * rho_omega / z_val)
                # A[:, j] += G[:, j] * (1j * rho_omega / z_val)
        
        # Case 2: Pressure is known (Open end / Source) - (Dirichlet BC)
        elif 'PRES' in bc:
            # Pressure is known (p), Velocity (v) is unknown
            A[:, j] = -G[:, j] * (1j * rho_omega)
            B -= H[:, j] * bc['PRES']
        
        # Case 3: Impedance (Absorbent material)
        elif 'IMPE' in bc and 'VELO' not in bc:
            # Admittance logic: v = p / Z. 
            # Term: (H - G/Z)*p = 0 -> A_col = H - G/Z, B = 0
            # Safety check for division by zero
            z_val = bc['IMPE'] if abs(bc['IMPE']) > 1e-12 else 1e-12
            # Units must match: H is unitless, G is [L], so we need [1/L]
            # (i * omega * rho) / Z  has units of [1/L]
            A[:, j] = H[:, j] - (G[:, j] * (1j * rho_omega / z_val))
            # A[j, j] = 0.5 - (G[:, j] * (1j * rho_omega / z_val))
        
        # Case 4: Rigid Wall, v=0 (Default)
        else:
            A[:, j] = H[:, j]

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

@njit(parallel=True)
def calculate_field_points(mic_centers, bem_centers, bem_areas, bem_normals, p_surf, v_surf, k, rho_omega):
    """
    Pass 2: Projects solved surface (BEM) results onto Microphone points.
    mic_centers: mics nodal coords.
    bem_centers: BEM elements CoG coords.
    """
    num_mics = len(mic_centers)
    num_surf = len(bem_centers)
    p_mics = np.zeros(num_mics, dtype=np.complex128)
    inv_4pi = 1.0 / (4.0 * np.pi)
    # TODO: when input VELO present, we have 1 too many '* (1j * rho_omega)' on those els.
    #       to be reviewed in the future.
    v_surf = v_surf * (1j * rho_omega)

    for i in prange(num_mics):
        sum_p = 0.0 + 0.0j
        for j in range(num_surf):
            r_vec = mic_centers[i] - bem_centers[j]
            r = np.linalg.norm(r_vec)
            
            # Green's Function (G)
            g_val = np.exp(1j * k * r) * inv_4pi / r
            
            # Derivative of Green's Function (H)
            # Dot product of (r_vec) and the surface normal / r
            r_dot_n = np.dot(r_vec, bem_normals[j]) / r
            h_val = g_val * (1j * k - 1.0 / r) * r_dot_n
            
            # p_mic = G*v + H*p (integrated over area)
            # TODO: check the sign for interior vs exterior when the time comes.
            sum_p += (g_val * v_surf[j] + h_val * p_surf[j]) * bem_areas[j]
            
        p_mics[i] = sum_p
    return p_mics

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