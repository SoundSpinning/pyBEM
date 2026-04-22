import numpy as np
from numba import njit

def calculate_element_properties(nodes, connectivity):
    """
    Calculates normals, the center (centroid) and surface area of elements.
    Supports S3 (3 nodes) and S4 (4 nodes).
    """
    # Get the coordinates for each node ID in the connectivity list
    pts = np.array([nodes[nid] for nid in connectivity])
    # The center is the average of all corner node's coordinates
    center = np.mean(pts, axis=0)

    # Get elem edges, lengths, area & normal.
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[1]
    v3 = pts[0] - pts[2]
    v1_len = np.linalg.norm(v1)
    v2_len = np.linalg.norm(v2)
    v3_len = np.linalg.norm(v3)
    max_len = max(v1_len, v2_len, v3_len)
    ratio = max_len / min(v1_len, v2_len, v3_len)
    
    # Cross product gives the normal vector
    cross_prod = np.cross(v1, v2)
    area_total = 0.5 * np.linalg.norm(cross_prod)
    
    if len(connectivity) == 4: # S4 Quad
        v3 = pts[3] - pts[2]
        v4 = pts[0] - pts[3]
        v3_len = np.linalg.norm(v3)
        v4_len = np.linalg.norm(v4)
        max_len = max(v1_len, v2_len, v3_len, v4_len)
        ratio = max_len / min(v1_len, v2_len, v3_len, v4_len)
        area_total += 0.5 * np.linalg.norm(np.cross(v3, v4))
        
    # Normalized unit normal
    unit_normal = cross_prod / np.linalg.norm(cross_prod)
    
    return pts, center, area_total, unit_normal, ratio, max_len

def prepare_geometry(nodes, elements):
    """
    Loops through all elements and gets their geometry properties 
    using the utility functions.
    """
    nodal_coords = []
    centers = []
    areas = []
    unit_normals = []
    ratios = []
    lengths = []
    
    for eid, conn in elements.items():
        e_n_coords, c, a, n, max_ratio, max_len = calculate_element_properties(nodes, conn)
        nodal_coords.append(e_n_coords)
        centers.append(c)
        areas.append(a)
        unit_normals.append(n)
        ratios.append(max_ratio)
        lengths.append(max_len)
    return nodal_coords, np.array(centers), np.array(areas), np.array(unit_normals), np.array(ratios), np.array(lengths)

def calculate_signed_volume(centers, areas, normals):
    """
    Calculates from the BEM elements: volume, total area & CoG.
    """
    # The sum of (Center dot Normal) * Area for volume & CoG calcs
    # 1.0/3.0 because it's a 3D volume integral
    volume = np.sum([np.dot(c, n) * a for c, a, n in zip(centers, areas, normals)]) / 3.0
    area = np.sum([areas])
    CoGx = np.sum([(centers[i,0]) * areas[i] for i in range(len(centers))]) / area 
    CoGy = np.sum([(centers[i,1]) * areas[i] for i in range(len(centers))]) / area 
    CoGz = np.sum([(centers[i,2]) * areas[i] for i in range(len(centers))]) / area 
    CoG = [CoGx, CoGy, CoGz]
    return volume, area, CoG

# ---------------------------------
###
### Gauss Points for QUADS & TRIAS
###
# ---------------------------------
###
### TERMINOLOGY
###
# ---------------------------------
# ξ (Xi) and η (Eta) are the standard Greek letters used to represent the local (or "parent") coordinate system of an element.
# ξ (Xi - pronounced "Ksee" or "Zai"): This represents the horizontal axis of the local square.
# η (Eta - pronounced "Ay-tuh"): This represents the vertical axis of the local square.
# In short: ξ is "Local X" and η is "Local Y." We use Greek letters just to make sure we don't accidentally confuse a point on the element with a point in the room!

# QUAD4 Gauss points and weights for 2x2 quadrature
# ---------------------------------
# Local coordinates: -/+ 1/sqrt(3): [-0.5773502691896257, 0.5773502691896257]
Q_GP = 1./3.**0.5
# NOTE: The logic here takes into account that BEM normals point AWAY from the fluid.
#       This is against usual RH rules with (+) normals into the material, e.g. in FEA.
#       As a result, it is not that trivial which order / signs the GPoints must follow.
QUAD_GP = np.array([
    [-Q_GP, -Q_GP],   # Bottom-Left
    [-Q_GP,  Q_GP],   # Top-Left
    [ Q_GP,  Q_GP],   # Top-Right
    [ Q_GP, -Q_GP],   # Bottom-Right
])
# # test influence of order of GPoints 
# QUAD_GP = np.array([
#     [-Q_GP, -Q_GP],   # Bottom-Left
#     [ Q_GP,  Q_GP],   # Top-Right
#     [ Q_GP, -Q_GP],   # Bottom-Right
#     [-Q_GP,  Q_GP],   # Top-Left
# ])
# Standard 2x2 Weights sum to 4.0
QUAD_GW = np.array([1.0, 1.0, 1.0, 1.0])

@njit
def get_quad_points(v1, v2, v3, v4):
    """
    Computes 4 spatial points on a quad4 element surface defined by its 4 vertices.
    2x2 Gauss Quadrature: Instead of calculating the kernel once at the center, we sample it at 4 specific locations (Gauss points) and take a weighted average. 
    For a standard quad element, these points are located at ±0.577 in local coordinates.
    """
    points = np.zeros((4, 3))
    weigths = QUAD_GW
    for i in range(4):
        xi, eta = QUAD_GP[i]
        # Bilinear interpolation of the surface; Ni are the shape functions
        n1 = 0.25 * (1-xi) * (1-eta)  # Bottom-Left
        n2 = 0.25 * (1-xi) * (1+eta)  # Top-Left
        n3 = 0.25 * (1+xi) * (1+eta)  # Top-Right
        n4 = 0.25 * (1+xi) * (1-eta)  # Bottom-Right
        points[i] = n1*v1 + n2*v2 + n3*v3 + n4*v4
        
    return points, weigths

# TRI3 (3) Gauss points in Barycentric coordinates (L1, L2, L3)
# ---------------------------------
# These points are at (2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)
# TRI_GP = np.array([
#     [0.666666666, 0.166666666, 0.166666666],
#     [0.166666666, 0.666666666, 0.166666666],
#     [0.166666666, 0.166666666, 0.666666666]
# ])
TRI_GP = np.array([
    [2/3., 1/6., 1/6.],
    [1/6., 2/3., 1/6.],
    [1/6., 1/6., 2/3.]
])
TRI_GW = np.array([1/3., 1/3., 1/3.]) # Weights sum to 1.0

@njit
def get_tri_points(v1, v2, v3):
    """
    For a triangle with a 3-point quadrature rule. 
    The points are located at the midpoints of the TRIA edges connecting the nodes.
    """
    points = np.zeros((3, 3))
    weights = TRI_GW
    for i in range(3):
        # Linear interpolation using barycentric coordinates
        points[i] = TRI_GP[i,0]*v1 + TRI_GP[i,1]*v2 + TRI_GP[i,2]*v3
    return points, weights

# QUAD4 Gauss points and weights for 3x3 quadrature
# ---------------------------------
# GPoints: -sqrt(0.6), 0, +sqrt(0.6)
# GP3 = np.array([-0.7745966692, 0.0, 0.7745966692])
Q_GP3 = 0.6**0.5
# QUAD_GP3 = np.array([-Q_GP3, 0.0, Q_GP3])

# NOTE: The logic here takes into account that BEM normals point AWAY from the fluid.
#       This is against usual RH rules with (+) normals into the material, e.g. in FEA.
#       As a result, it is not that trivial which order / signs the GPoints must follow.
# 1.- Local Coords:
    # Define the 9 local (xi, eta) pairs explicitly to match theBEM winding.
    # We order them row-by-row, but keep the signs aligned with the 'Away' from fluid normal.
QUAD_GP3 = np.array([
    [-Q_GP3, -Q_GP3], [-Q_GP3, 0],      [-Q_GP3, Q_GP3], # Bottom row
    [ 0, Q_GP3],      [ 0,  0],         [ Q_GP3, Q_GP3], # Middle row (Point 4 is CoG)
    [ Q_GP3, 0],      [ Q_GP3, -Q_GP3], [ 0, -Q_GP3]     # Top row
])
# Standard ordering with (+) normals, NOT for BEM
# QUAD_GP3 = np.array([
#     [-Q_GP3, -Q_GP3], [ 0, -Q_GP3], [ Q_GP3, -Q_GP3], # Bottom row
#     [-Q_GP3,  0], [ 0,  0], [ Q_GP3,  0],             # Middle row (Point 4 is CoG)
#     [-Q_GP3,  Q_GP3], [ 0,  Q_GP3], [ Q_GP3,  Q_GP3]  # Top row
# ])

# 2.- Local Weights:
# Weights: 5/9, 8/9, 5/9
# GW3 = np.array([0.5555555556, 0.8888888889, 0.5555555556])
# Q_GW3 = np.array([5/9., 8/9., 5/9.])
Q_GW3_side = 5/9.
Q_GW3_mid = 8/9.
# Define corresponding weights (matching the i, j index of GW3)
QUAD_GW3 = np.array([
    Q_GW3_side*Q_GW3_side, Q_GW3_side*Q_GW3_mid,  Q_GW3_side*Q_GW3_side,
    Q_GW3_side*Q_GW3_mid,  Q_GW3_mid*Q_GW3_mid,   Q_GW3_side*Q_GW3_side,
    Q_GW3_side*Q_GW3_mid,  Q_GW3_side*Q_GW3_side, Q_GW3_side*Q_GW3_mid
])

@njit
def get_quad_points_3x3(v1, v2, v3, v4):
    """
    Returns 9 points and 9 weights for a 3x3 integration rule.
    """
    points = np.zeros((9, 3))
    weights = np.zeros(9)

    for i in range(9):
        xi, eta = QUAD_GP3[i]
            
        # Shape functions for bilinear quad
        n1 = 0.25 * (1-xi) * (1-eta)  # Bottom-Left
        n2 = 0.25 * (1-xi) * (1+eta)  # Top-Left
        n3 = 0.25 * (1+xi) * (1+eta)  # Top-Right
        n4 = 0.25 * (1+xi) * (1-eta)  # Bottom-Right
        
        points[i] = n1*v1 + n2*v2 + n3*v3 + n4*v4
        weights[i] = QUAD_GW3[i]

    return points, weights

# 7-point rule for Triangles (Barycentric coordinates L1, L2, L3)
# ---------------------------------
# Format: [L1, L2, L3, Weight]
TRI_7P = np.array([
    [0.3333333333, 0.3333333333, 0.3333333333, 0.2250000000], # Centroid
    [0.7974269853, 0.1012865073, 0.1012865073, 0.1259391805], # Near Vertex 1
    [0.1012865073, 0.7974269853, 0.1012865073, 0.1259391805], # Near Vertex 2
    [0.1012865073, 0.1012865073, 0.7974269853, 0.1259391805], # Near Vertex 3
    [0.4701420641, 0.4701420641, 0.0597158717, 0.1323941527], # Near Mid-edge 1
    [0.0597158717, 0.4701420641, 0.4701420641, 0.1323941527], # Near Mid-edge 2
    [0.4701420641, 0.0597158717, 0.4701420641, 0.1323941527]  # Near Mid-edge 3
])
# old (wrong) implementation
# TRI_7P = np.array([
#     [0.3333333333, 0.3333333333, 0.3333333333, 0.2250000000], # Centroid
#     [0.7974269853, 0.1012865073, 0.1012865073, 0.1259391805], # Near Vertex 1
#     [0.1012865073, 0.7974269853, 0.1012865073, 0.1259391805], # Near Vertex 2
#     [0.1012865073, 0.1012865073, 0.7974269853, 0.1259391805], # Near Vertex 3
#     [0.0597158717, 0.4701420641, 0.4701420641, 0.1323941527], # Near Mid-edge 1
#     [0.4701420641, 0.0597158717, 0.4701420641, 0.1323941527], # Near Mid-edge 2
#     [0.4701420641, 0.4701420641, 0.0597158717, 0.1323941527]  # Near Mid-edge 3
# ])

@njit
def get_tri_points_7p(v1, v2, v3):
    """
    Returns 7 spatial points and weights for a high-order triangle rule.
    """
    points = np.zeros((7, 3))
    weights = np.zeros(7)
    for i in range(7):
        # Linear interpolation using barycentric coordinates
        l1, l2, l3, w = TRI_7P[i]
        points[i] = l1*v1 + l2*v2 + l3*v3
        weights[i] = w
    return points, weights

@njit
def compute_mid_order_contribution(receiver_pt, element_vertices, element_normal, element_area, k, H_sign, inv_4pi):
    """Integrates G and H kernels over one element using 'TRIA_3p & QUAD_4p' quadrature."""
    n_nodes = len(element_vertices)
    
    # Initialize sums
    g_sum = 0.0 + 0j
    h_sum = 0.0 + 0j
    sum_w = 0.0
    
    # 1. Get Integration Points & Weights
    if n_nodes == 3: # TRIA3
        pts, wts = get_tri_points(element_vertices[0], element_vertices[1], element_vertices[2])
        n_pts = 3
    else: # QUAD4
        pts, wts = get_quad_points(element_vertices[0], element_vertices[1], element_vertices[2], element_vertices[3])
        n_pts = 4
    # print("\n", receiver_pt)
    # print(pts, wts)
    sum_w = np.sum(wts)

    # 2. Sum Contributions
    for p_idx in range(n_pts):
        r_vec = receiver_pt - pts[p_idx]
        r = np.linalg.norm(r_vec)
        
        # Kernel math & Integration weights
        # G Kernel
        exp_jkr = np.exp(1j * k * r)
        g_val = exp_jkr * inv_4pi / r
        
        w_eff = (wts[p_idx] / sum_w) * element_area
        g_sum += g_val * w_eff
        
        # H Kernel
        dot_prod = np.dot(r_vec, element_normal) / r
        h_sum += H_sign * g_val * (1j * k - 1.0/r) * dot_prod * w_eff
        
    return g_sum, h_sum

@njit
def compute_high_order_contribution(receiver_pt, vertices, normal, area, k, H_sign, inv_4pi):
    """Integrates G and H kernels over one element using 'TRIA_7p & QUAD_9p' quadrature."""
    n_nodes = len(vertices)
    
    # Initialize sums
    g_sum = 0.0 + 0j
    h_sum = 0.0 + 0j
    sum_w = 0.0
    
    # 1. Get Integration Points & Weights
    if n_nodes == 3:
        pts, wts = get_tri_points_7p(vertices[0], vertices[1], vertices[2])
        n_pts = 7
    else:
        pts, wts = get_quad_points_3x3(vertices[0], vertices[1], vertices[2], vertices[3])
        n_pts = 9
    # print("\n", receiver_pt)
    # print(pts, wts)
    sum_w = np.sum(wts)

    # 2. Sum Contributions
    for p in range(n_pts):
        r_vec = receiver_pt - pts[p]
        r = np.linalg.norm(r_vec)
        
        # G Kernel
        exp_jkr = np.exp(1j * k * r)
        g_val = exp_jkr * inv_4pi / r
        
        # H Kernel derivative
        dot_prod = np.dot(r_vec, normal) / r
        h_val = H_sign * g_val * (1j * k - 1.0/r) * dot_prod
        
        # Integration weight
        w_eff = (wts[p] / sum_w) * area
            
        g_sum += g_val * w_eff
        h_sum += h_val * w_eff
        
    return g_sum, h_sum

def averaged_at_nodes(nodes, elements, P_bem, bem_areas, mic_nodes=None, P_mics=None):
    """
    Averages element-centered results to nodes.
    nodes: dict {nid: [x, y, z]}
    elements: dict {eid: [n1, n2, ...]}
    P_bem: array of complex values (one per element)
    mic_nodes: dict {nid: [x, y, z]}
    P_mics: array of complex values (one per node)
    nodal_pressures: array of complex values (one per node)
    """
    # 1. Determine the size needed for the array
    # We use max_id + 1 so we can index directly by node_id
    max_node_id = max(nodes.keys())
    
    # Initialize buffers
    # node_sums: stores the accumulated elemental pressure
    # count: stores how many elements share a node (for averaging)
    node_sums = np.zeros(max_node_id + 1, dtype=np.complex128)
    area_sums = np.zeros(max_node_id + 1, dtype=np.complex128)
    # count = np.zeros(max_node_id + 1, dtype=np.float32)
    nodal_pressures = np.zeros(max_node_id + 1, dtype=np.complex128)

    # 2. Map element results to their constituent nodes
    # We iterate through the elements provided for BEM
    for i, (eid, conn) in enumerate(elements.items()):
        area = bem_areas[i]
        val = P_bem[i] * area
        for nid in conn:
            node_sums[nid] += val
            area_sums[nid] += area
            # count[nid] += 1
    # We iterate through the nodes for Mics, if provided
    if mic_nodes is not None:
        for i, (nid, coords) in enumerate(mic_nodes.items()):
            val = P_mics[i]
            node_sums[nid] += val
            area_sums[nid] = 1
            # count[nid] = 1

    # 3. Perform the average
    # We only divide where area > 0 to avoid DivisionByZero
    # This also gets the nodal results in the right index order for PV output
    # mask = count > 0
    mask = area_sums > 0
    # nodal_pressures[mask] = node_sums[mask] / count[mask]
    nodal_pressures[mask] = node_sums[mask] / area_sums[mask]

    return nodal_pressures


# NOTE: it does same as above but 5 x slower, do not use.
# np.arrays win!
# def averaged_at_nodes(nodes, elements, element_values):
#     """
#     Averages element-centered results to nodes.
#     nodes: dict {nid: [x, y, z]}
#     elements: dict {eid: [n1, n2, ...]}
#     element_values: array of complex values (one per element)
#     nodal_pressures: array of complex values (one per node)
#     """
#     # 1. Determine the size needed for the array
#     # We use num_nodes + 1 so we can index directly by node_id
#     num_nodes = len(nodes.keys())
#     node_ids = nodes.keys()
#     node_sums = {}
#     count = {}
#     nodal_pressures = {}
    
#     # Initialize buffers
#     # node_sums: stores the accumulated elemental pressure
#     # count: stores how many elements share that node (for averaging)
#     for nid in node_ids:
#         node_sums[nid] = np.zeros(1, dtype=np.complex128)
#         count[nid] = np.zeros(1, dtype=np.float32)
#         nodal_pressures[nid] = np.zeros(1, dtype=np.complex128)

#     # 2. Map element results to their constituent nodes
#     # We iterate through the elements provided (could be BEM or Mics)
#     for i, (eid, conn) in enumerate(elements.items()):
#         val = element_values[i]
#         for nid in conn:
#             node_sums[nid] += val
#             count[nid] += 1

#     # 3. Perform the average
#     # We only divide where count > 0 to avoid DivisionByZero
#     # mask = count > 0
#     for nid in node_ids:
#         if count[nid][0] > 0:
#             nodal_pressures[nid] = node_sums[nid][0] / count[nid][0]
#         # nodal_pressures[i] = node_sums[i] / 1

#     return nodal_pressures