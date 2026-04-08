import numpy as np

def calculate_element_properties(nodes, connectivity):
    """
    Calculates normals, the center (centroid) and surface area of elements.
    Supports S3 (3 nodes) and S4 (4 nodes).
    """
    # Get the coordinates for each node ID in the connectivity list
    pts = np.array([nodes[nid] for nid in connectivity])
    # The center is the average of all corner node's coordinates
    center = np.mean(pts, axis=0)

    # Vector 1 (Node 0 to 1) and Vector 2 (Node 0 to 2)
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    
    # Cross product gives the normal vector
    cross_prod = np.cross(v1, v2)
    area_total = 0.5 * np.linalg.norm(cross_prod)
    
    if len(connectivity) == 4: # S4 Quad
        v3 = pts[3] - pts[0]
        area_total += 0.5 * np.linalg.norm(np.cross(v2, v3))
        
    # Normalized unit normal
    unit_normal = cross_prod / np.linalg.norm(cross_prod)
    
    return center, area_total, unit_normal

def prepare_geometry(nodes, elements):
    """
    Loops through all elements and gets their geometry properties 
    using the utility functions.
    """
    centers = []
    areas = []
    unit_normals = []
    
    for eid, conn in elements.items():
        c, a, n = calculate_element_properties(nodes, conn)
        centers.append(c)
        areas.append(a)
        unit_normals.append(n)
    return np.array(centers), np.array(areas), np.array(unit_normals)

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

def averaged_at_nodes(nodes, elements, P_bem, bem_areas, mic_nodes, P_mics):
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
    # We iterate through the nodes provided for Mics
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