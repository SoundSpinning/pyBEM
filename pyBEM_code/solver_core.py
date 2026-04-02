# The heavy math (Numba-accelerated BEM kernels)

import numpy as np
from numba import njit, prange
from constants import BEM_TYPE

# @njit(fastmath=True)   # 2 x slower vs parallel=True; both together is same 2 x slower.
@njit(parallel=True)
def assemble_system(centers, areas, normals, k, BEM_TYPE):
    """
    The Engine: Computes G and H matrices using element normals.
    centers: (N, 3) array
    areas: (N,) array
    normals: (N, 3) array
    k: wave number (complex or float)
    Checks if BEM_TYPE = INTERIOR or EXTERIOR, and adapts maths accordingly (Hij sign).
    """
    n_els = len(centers)   # number of BEM elements
    # Build 2D complex matrices
    G = np.zeros((n_els, n_els), dtype=np.complex128)
    H = np.zeros((n_els, n_els), dtype=np.complex128)
    
    # 4*pi is a constant in the Green's function denominator
    inv_4pi = 1.0 / (4.0 * np.pi)
    # set the sign in Hij depending if exterior or interior BEM
    if BEM_TYPE == "INTERIOR": H_sign = -1.0
    elif BEM_TYPE == "EXTERIOR": H_sign = 1.0

    for i in prange(n_els):
        for j in range(n_els):
            if i == j: # Analytical self-term approximations for diagonal terms
                # G[i, j] = inv_4pi * np.sqrt(areas[j] / np.pi)
                G[i, j] = inv_4pi * areas[j]
                H[i, j] = 0.5  # Jump term for smooth surfaces
            else:
                # Vector from source j to receiver i
                r_vec = centers[i] - centers[j]
                r = np.linalg.norm(r_vec)
                
                # Free-space 3D Helmholtz Green's function
                g_val = np.exp(1j * k * r) * inv_4pi / r
                G[i, j] = g_val * areas[j]
                
                # Double Layer (H): Derivative of G with respect to normal n_j
                # grad_g = g_val * (1j * k - 1/r) * (r_vec / r)
                dot_prod = np.dot(r_vec, normals[j]) / r
                H[i, j] = H_sign * g_val/r * ((1j * k * r) - 1.0) * dot_prod * areas[j]
                # H[i, j] = 1.06 * H_sign * g_val * (1j * k - 1.0 / r) * dot_prod * areas[j]
    return G, H

# @njit(parallel=True)   # it crashes Numba
# @njit(fastmath=True)   # it crashes Numba
def solve_bem_system(G, H, bc_map, elements_list, rho_omega, log=None):
    """
    Re-arranges the BEM system (H*p = G*v) based on Boundary Conditions.
    Solves Ax=B.
    Returns the complex pressure for every element.
    elements_list: sorted list of element IDs to ensure index alignment
    """
    num_elements = len(elements_list)
    A = np.zeros((num_elements, num_elements), dtype=np.complex128)
    B = np.zeros(num_elements, dtype=np.complex128)

    # We MUST use the same index 'j' that corresponds to the matrix columns (0, 1, 2...)
    # eid is the ACTUAL ID from the .inp (1, 101, 500...)
    for j, eid in enumerate(elements_list):
        bc = bc_map.get(eid, {})

        # Case 1: Velocity is known (Vibrating Wall)
        if 'VELO' in bc:
            # Velocity is known (v), Pressure (p) is unknown
            A[:, j] = H[:, j]
            B += G[:, j] * bc['VELO'] * (-1j * rho_omega)
        
        # Case 2: Pressure is known (Open end / Source)
        elif 'PRES' in bc:
            # Pressure is known (p), Velocity (v) is unknown
            A[:, j] = -G[:, j]
            B -= H[:, j] * bc['PRES']
        
        # Case 3: Impedance (Absorbent material)
        elif 'IMPE' in bc:
            # Admittance logic: v = p / Z. 
            # Term: (H - G/Z)*p = 0 -> A_col = H - G/Z, B = 0
            # Safety check for division by zero
            z_val = bc['IMPE'] if abs(bc['IMPE']) > 1e-12 else 1e-12
            A[:, j] = H[:, j] - (G[:, j] / z_val)
        
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
    cond = 'N/A'
    
    # Solve the linear system Ax = B
    # Solve for the unknown surface values (usually Pressure)
    bem_solution = np.linalg.solve(A, B)
    return (bem_solution, cond)

def derive_surface_vectors(p_sol, bc_map, bem_ids):
    """
    Reconstructs the full pressure and velocity vectors for the surface.
    p_sol: The unknown values returned by the solver.
    bc_map: The dictionary of boundary conditions.
    bem_ids: The sorted list of element IDs used in the matrix.
    """
    num_elements = len(bem_ids)
    p_final = np.zeros(num_elements, dtype=np.complex128)
    v_final = np.zeros(num_elements, dtype=np.complex128)

    for j, eid in enumerate(bem_ids):
        bc = bc_map.get(eid, {})
        
        if 'VELO' in bc:
            # We knew Velocity, p_sol gave us Pressure
            p_final[j] = p_sol[j]
            v_final[j] = bc['VELO']
            
        elif 'PRES' in bc:
            # We knew Pressure, p_sol gave us Velocity
            p_final[j] = bc['PRES']
            v_final[j] = p_sol[j]
            
        elif 'IMPE' in bc:
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
def calculate_field_points(mic_centers, bem_centers, bem_areas, bem_normals, p_surf, v_surf, k):
    """
    Pass 2: Projects solved surface (BEM) results onto Microphone points.
    """
    num_mics = len(mic_centers)
    num_surf = len(bem_centers)
    p_mics = np.zeros(num_mics, dtype=np.complex128)
    inv_4pi = 1.0 / (4.0 * np.pi)

    for i in prange(num_mics):
        sum_p = 0.0 + 0.0j
        for j in range(num_surf):
            r_vec = mic_centers[i] - bem_centers[j]
            r = np.linalg.norm(r_vec)
            
            # Green's Function (G)
            g_val = np.exp(1j * k * r) * inv_4pi / r
            
            # Derivative of Green's Function (H)
            # Dot product of (r_vec/r) and the surface normal
            cos_theta = np.dot(r_vec, bem_normals[j]) / r
            h_val = g_val * (1j * k - 1.0 / r) * cos_theta
            
            # p_mic = G*v - H*p (integrated over area)
            sum_p += (g_val * v_surf[j] - h_val * p_surf[j]) * bem_areas[j]
            
        p_mics[i] = sum_p
    return p_mics

# def get_greens(r, k):
#     """3D Helmholtz Green's Function."""
#     if r < 1e-9: return 0.0 + 0.0j
#     return np.exp(1j * k * r) / (4.0 * np.pi * r)