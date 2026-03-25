# The heavy math (Numba-accelerated BEM kernels)

import numpy as np
from numba import njit, prange
from utils import calculate_element_properties
from constants import BEM_TYPE

def prepare_geometry(nodes, elements):
    """
    Loops through all elements and gets their physics properties 
    using the utility functions.
    """
    centers = []
    areas = []
    
    for eid, conn in elements.items():
        # We call the function we imported from utils.py
        c, a = calculate_element_properties(nodes, conn)
        centers.append(c)
        areas.append(a)
        
    return np.array(centers), np.array(areas)

def solve_bem_system(G, H, bc_map, elements_list, log=None):
    """
    Rearranges the BEM system (H*p = G*v) based on Boundary Conditions.
    Solves Ax=B.
    Returns the complex pressure for every element.
    elements_list: sorted list of element IDs to ensure index alignment
    Logs the process.
    """
    num_elements = len(elements_list)
    A = np.zeros((num_elements, num_elements), dtype=np.complex128)
    B = np.zeros(num_elements, dtype=np.complex128)

    # We MUST use the same index 'j' that corresponds to the matrix columns
    # j is the matrix column (0, 1, 2...)
    # eid is the ACTUAL ID from the .inp (1, 101, 500...)
    for j, eid in enumerate(elements_list):
        bc = bc_map.get(eid, {})
        
        # Case 1: Velocity is known (Vibrating Wall)
        if 'VELO' in bc:
            # Velocity is known (v), Pressure (p) is unknown
            A[:, j] = H[:, j]
            B += G[:, j] * bc['VELO']
        # Case 2: Pressure is known (Open end / Source)
        elif 'PRES' in bc:
            # Pressure is known (p), Velocity (v) is unknown
            A[:, j] = -G[:, j]
            B -= H[:, j] * bc['PRES']
        # Case 3: Impedance (Absorbent material)
        elif 'IMPE' in bc:
            # Relation: v = p / Z. 
            # Term: (H - G/Z)*p = 0 -> A_col = H - G/Z, B = 0
            # Safety check for division by zero
            z_val = bc['IMPE'] if abs(bc['IMPE']) > 1e-12 else 1e-12
            # Admittance logic: v = p/Z. Term becomes (H - G/Z)p
            A[:, j] = H[:, j] - (G[:, j] / z_val)
        # Case 4: Rigid Wall, v=0 (Default)
        else:
            A[:, j] = H[:, j]

    # # --- Solver Feedback ---
    # # Check Matrix Condition Number (good UX)
    # # If this number is too high, the mesh might be bad
    # if log:
    #     # Condition number tells us if the mesh is 'broken' or math is unstable
    #     # Note: This can be slow for huge matrices, use sparingly!
    #     cond = np.linalg.cond(A)
    #     # log.write(f"      Matrix Condition Number: {cond:.2e}\n")
    #     if cond > 1e12:
    #         log.write("      WARNING: Matrix is ill-conditioned. Check for overlapping elements!\n")
    cond='N/A'
    # Solve the linear system Ax = B
    # Solve for the unknown surface values (usually Pressure)
    surface_solution = np.linalg.solve(A, B)
    return (surface_solution, cond)

# @njit(fastmath=True)
# def get_greens(r, k):
#     """3D Helmholtz Green's Function."""
#     if r < 1e-9: return 0.0 + 0.0j
#     return np.exp(1j * k * r) / (4.0 * np.pi * r)

@njit(fastmath=True)
# @njit(parallel=True)
def assemble_system(centers, areas, normals, k, BEM_TYPE):
    """
    The Engine: Computes G and H matrices using element normals.
    centers: (N, 3) array
    areas: (N,) array
    normals: (N, 3) array
    k: wave number (complex or float)
    Checks if BEM_TYPE = INTERIOR or EXTERIOR, and adapts maths accordingly.
    """
    n = len(centers)
    G = np.zeros((n, n), dtype=np.complex128)
    H = np.zeros((n, n), dtype=np.complex128)
    
    # 4*pi is a constant in the Green's function denominator
    inv_4pi = 1.0 / (4.0 * np.pi)
    # set the diagonal terms in H depending if exterior or interior BEM
    if BEM_TYPE == "INTERIOR":
        H_sign = -1.0
    elif BEM_TYPE == "EXTERIOR":
        H_sign = 1.0

    for i in prange(n):
        for j in range(n):
            if i == j:
                # Analytical self-term approximations for diagonal terms
                G[i, j] = inv_4pi * np.sqrt(areas[i] / np.pi)
                H[i, j] = H_sign*0.5  # Jump term for smooth surfaces
            else:
                # Vector from source j to receiver i
                r_vec = centers[i] - centers[j]
                r = np.linalg.norm(r_vec)
                
                # Free-space 3D Helmholtz Green's function
                g_val = np.exp(1j * k * r) * inv_4pi / r
                G[i, j] = g_val * areas[j]
                
                # Double Layer (H): Derivative of G with respect to normal n_j
                # grad_g = g_val * (1j * k - 1/r) * (r_vec / r)
                dot_prod = np.dot(r_vec / r, normals[j])
                H[i, j] = g_val * (1j * k - 1.0 / r) * dot_prod * areas[j]
    return G, H

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

def calculate_field_points(mic_centers, surf_centers, surf_areas, surf_normals, p_surf, v_surf, k):
    """
    Pass 2: Projects solved surface data onto Microphone points.
    """
    num_mics = len(mic_centers)
    num_surf = len(surf_centers)
    p_mics = np.zeros(num_mics, dtype=np.complex128)
    inv_4pi = 1.0 / (4.0 * np.pi)

    for i in prange(num_mics):
        sum_p = 0.0 + 0.0j
        for j in range(num_surf):
            r_vec = mic_centers[i] - surf_centers[j]
            r = np.linalg.norm(r_vec)
            
            # Green's Function (G)
            g_val = np.exp(1j * k * r) * inv_4pi / r
            
            # Derivative of Green's Function (H)
            # Dot product of (r_vec/r) and the surface normal
            cos_theta = np.dot(r_vec / r, surf_normals[j])
            h_val = g_val * (1j * k - 1.0 / r) * cos_theta
            
            # p_mic = G*v - H*p (integrated over area)
            sum_p += (g_val * v_surf[j] - h_val * p_surf[j]) * surf_areas[j]
            
        p_mics[i] = sum_p
    return p_mics