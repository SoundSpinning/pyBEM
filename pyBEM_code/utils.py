import numpy as np

def calculate_element_properties(nodes, connectivity):
    """
    Calculates normals, the center (centroid) and surface area of elements.
    Supports S3 (3 nodes) and S4 (4 nodes).
    """
    # Get the coordinates for each node ID in the connectivity list
    pts = np.array([nodes[nid] for nid in connectivity])
    # The center is the average of all corner coordinates
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
    normal = cross_prod / np.linalg.norm(cross_prod)
    
    return center, area_total, normal

def calculate_signed_volume(centers, areas, normals):
    # The sum of (Center dot Normal) * Area
    # We use 1.0/3.0 because it's a 3D volume integral
    volume = np.sum([np.dot(c, n) * a for c, a, n in zip(centers, areas, normals)]) / 3.0
    return volume

def average_to_nodes(nodes, elements, element_values):
    """
    Averages element-centered results to nodes.
    nodes: dict {nid: [x, y, z]}
    elements: dict {eid: [n1, n2, ...]}
    element_values: array of complex values (one per element)
    """
    # 1. Determine the size needed for the array
    # We use max_id + 1 so we can index directly by node_id
    max_node_id = max(nodes.keys())
    
    # Initialize buffers
    # node_sums: stores the accumulated elemental pressure
    # count: stores how many elements share that node (for averaging)
    node_sums = np.zeros(max_node_id + 1, dtype=np.complex128)
    count = np.zeros(max_node_id + 1, dtype=np.float32)
    result = np.zeros(max_node_id + 1, dtype=np.complex128)

    # 2. Map element results to their constituent nodes
    # We iterate through the elements provided (could be BEM OR Mics)
    for i, (eid, conn) in enumerate(elements.items()):
        val = element_values[i]
        for nid in conn:
            node_sums[nid] += val
            count[nid] += 1

    # 3. Perform the average
    # We only divide where count > 0 to avoid DivisionByZero
    mask = count > 0
    result[mask] = node_sums[mask] / count[mask]

    return result