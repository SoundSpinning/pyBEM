from collections import defaultdict
import psutil
import os

def get_cpus():
    # CPUs
    cpus = psutil.cpu_count(logical=False)
    # Threads
    cores = psutil.cpu_count(logical=True)
    # RAM on machine
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    return cpus, cores, available_ram_gb

def get_ram():
    process = psutil.Process(os.getpid())
    # rss is the Resident Set Size (actual RAM used) in bytes
    # Convert to MB with an adjust to Win10 task manager tests
    return (process.memory_info().rss / (1024**2)) * 1.14

# PARALLEL env settings
# Force single-threading for the underlying math libraries
# This must happen BEFORE numpy/scipy are imported
def set_hardware_limits(threads_per_worker):
    t_str = str(threads_per_worker)
    
    # Generic OpenMP (Used by many libraries)
    os.environ["OMP_NUM_THREADS"] = t_str
    
    # Intel MKL Specifics
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["MKL_NUM_THREADS"] = t_str
    
    # OpenBLAS Specifics
    os.environ["OPENBLAS_NUM_THREADS"] = t_str
    
    # macOS/Accelerate Specifics
    os.environ["VECLIB_MAXIMUM_THREADS"] = t_str
    
    # Numexpr Specifics (often used in Pandas/Numpy)
    os.environ["NUMEXPR_NUM_THREADS"] = t_str
    # Below Controls the thread pool for Numba's parallel=True & prange loops.
    # However, see in main code we take control of this part via set_num_threads()
    # os.environ["NUMBA_NUM_THREADS"] = t_str

import numpy as np
from numba import njit

def validate_and_log_zones(zone_mesh_data, sorted_nodes, parser, log_f, log_top):
    """
    Diagnostic Mesh Checks & Logging for Water-Tight Zones.
    Checks on BEM golden rules across every zone.
    It also auto-checks if a zone is interior or exterior (normals) and sets H_sign accordingly.
    BEM normals must be consistent in a zone and point AWAY from the acoustic fluid.
    """
    
    log_info = ""
    global_h_signs = {}
    global_order_lengths = {}
    
    log_info += f"""
    ==============================
    *** INPUT MESH DIAGNOSTICS ***
    ==============================
    Found a total of ( {len(parser.zone_to_elsets)} ) Acoustic ZONES as follows:
    """
    
    for zone_name, info in zone_mesh_data.items():
        z_elements = info['elements']
        mat_name = info.get('material_name', zone_name)
        c_local = parser.materials[mat_name]['c']
        
        # 1. Local Geometry Preps (Centers, Areas, Normals)
        nodal_coords, bem_centers, bem_areas, bem_normals, elem_ratio, max_el_length = prepare_geometry(sorted_nodes, z_elements)
        
        # Compute order characteristic distance length (95th percentile)
        order_length = np.percentile(max_el_length, 95)
        global_order_lengths[zone_name] = order_length
        
        # 2. Run Single-Domain checks (Volume, Free Edges, Conflicts) via Divergence Theorem
        bem_total_vol, bem_total_area, bem_CoG, conflicts, free_edges = get_geo_info(z_elements, bem_centers, bem_areas, bem_normals)
        
        log_info += f"\n--> ZONE: [ {zone_name} ]"
        log_info += f"\n    BEM Surface Area:           ( {bem_total_area:.4f} L**2 )"
        log_info += f"\n    CoG of BEM zone:            [ {bem_CoG[0]:.2f}, {bem_CoG[1]:.2f}, {bem_CoG[2]:.2f} ] L"
        log_info += f"\n    Max Element Aspect Ratio:   ( {np.max(elem_ratio):.2f} )"
        log_info += f"\n    Element Size (approx):      ( {order_length:.2f} L )"
        log_info += f"\n    Suggested Max Freq:         ( {c_local / (8 * order_length):.1f} Hz )"
        
        # --- STRICT GOLDEN RULES ENFORCEMENT ---
        # Rule 1: Every single zone must be water-tight
        if free_edges > 0:
            log_info += f"\n\n    [!] FATAL MESH ERROR in Zone '{zone_name}': {free_edges} open/free edges detected. Mesh is not water-tight!\n"
            _write_and_fail(log_f, log_top, log_info)
            
        # Rule 2: Normals must be aligned consistently
        if conflicts > 0:
            log_info += f"\n\n    [!] FATAL MESH ERROR in Zone '{zone_name}': {conflicts} unaligned element normals found.\n        Please check all normals are consistent per zone.\n"
            _write_and_fail(log_f, log_top, log_info)
            
        # Rule 3: Detect Interior vs Exterior via Signed Volume orientation
        if bem_total_vol > 1e-9:
            global_h_signs[zone_name] = -1.0
            log_info += f"""
    Closed (+) Volume detected: ( {bem_total_vol:.4f} L**3 )
    -> Normals point OUTWARDS ==> Assuming INTERIOR Analysis
"""
        elif bem_total_vol < -1e-9:
            global_h_signs[zone_name] = 1.0
            log_info += f"""
    Closed (-) Volume detected: ( {bem_total_vol:.4f} L**3 )
    -> Normals point INWARDS ==> Assuming EXTERIOR Analysis
"""
        else:
            # If volume is zero and it's supposed to be a closed zone, it's a flat/corrupted shell
            log_info += f"\n\n    [!] FATAL MESH ERROR in Zone '{zone_name}': Invalid topology. Zero volume found in closed sub-domain.\n"
            _write_and_fail(log_f, log_top, log_info)
                
    return log_info, global_h_signs, global_order_lengths

def _write_and_fail(log_f, log_top, log_info):
    with open(log_f, "w") as log:
        log.write(log_top + log_info)
    raise RuntimeError(f"\n [pyBEM] PRE-PROCESSING GEOMETRY VALIDATION FAILED. See log file '{log_f}' for further details.\n")

def get_zone_data(parser, sorted_nodes):
    """
    Groups BEM elements and MICS nodes by material names per zone.
    If no multi-zones exist (single zone & material), it bundles everything into a single domain.
    It handles the case where some or all zones have no MICS elements defined.
    """
    zone_data = {}
    
    # Fallback to single zone mode if zone_to_elsets wasn't populated
    zones_to_process = parser.zone_to_elsets if getattr(parser, 'zone_to_elsets', None) else {"Single_Domain": None}

    # MICS check: does the input model have any microphones?
    has_global_mics = bool(getattr(parser, 'mics_elements', None))
    
    for zone_name, elset_names in zones_to_process.items():
        zone_data[zone_name] = {
            'elements': {},
            'mics_elements': {},
            'mics_nodes': None,
            'n_mics': 0
        }
        
        # 1. Extract BEM elements belonging to this zone
        if elset_names is None: # Single Domain Fallback
            zone_data[zone_name]['elements'] = dict(sorted(parser.elements.items()))
        else:
            # Multi-zone: Collect elements only matching this zone's elsets
            for elset in elset_names:
                if elset in parser.elsets:
                    for eid in parser.elsets[elset]:
                        if eid in parser.elements:
                            zone_data[zone_name]['elements'][eid] = parser.elements[eid]
            zone_data[zone_name]['elements'] = dict(sorted(zone_data[zone_name]['elements'].items()))
            
        # 2. Extract MICS elements assigned to this material zone
        #    (ONLY if global mics exist)
        if has_global_mics:
            if getattr(parser, 'element_to_zone', None):
                # Multi-zone mapping
                for eid, conn in parser.mics_elements.items():
                    if parser.element_to_zone.get(eid) == zone_name:
                        zone_data[zone_name]['mics_elements'][eid] = conn
                zone_data[zone_name]['mics_elements'] = dict(sorted(zone_data[zone_name]['mics_elements'].items()))
            else:
                # Single domain fallback
                zone_data[zone_name]['mics_elements'] = dict(sorted(parser.mics_elements.items()))
        
        # 3. Resolve localized Microphone Node Coordinates (Unique, Unsorted Sequence)
        if zone_data[zone_name]['mics_elements']:
            mics_nodes_track = {}
            
            # Loop through elements sequentially as parsed to preserve the mesh flow layout
            for conn in zone_data[zone_name]['mics_elements'].values():
                for nid in conn:
                    if nid in sorted_nodes:
                        # Dict tracking naturally keeps unique keys in order of insertion
                        mics_nodes_track[nid] = sorted_nodes[nid]
            mics_nodes_track = dict(sorted(mics_nodes_track.items()))
            
            # Extract coordinates directly from our ordered dictionary as float32
            zone_data[zone_name]['mics_nodes'] = np.array(list(mics_nodes_track.values()), dtype=np.float32)
            zone_data[zone_name]['n_mics'] = len(mics_nodes_track)
            
            # Keep the dictionary matching the array rows 1:1 for the post-processor
            zone_data[zone_name]['mics_nodes_dict'] = mics_nodes_track
        else:
            # Explicit safe empty defaults if this zone has no mics
            zone_data[zone_name]['mics_nodes'] = np.empty((0, 3), dtype=np.float32)
            zone_data[zone_name]['n_mics'] = 0
            zone_data[zone_name]['mics_nodes_dict'] = {}
        
        # # DEBUG
        # print(f"=== DIAGNOSTIC 2: GEOMETRY EXTRACTION FOR ZONE [ {zone_name} ] ===")
        # local_mic_elements = zone_data[zone_name].get('mics_elements', {})
        # print(f"  -> Number of MICS elements filtered into this zone: {len(local_mic_elements)}")

        # if 'mics_nodes_dict' in zone_data[zone_name]:
        #     print(f"  -> Number of unique MICS nodes in mics_nodes_dict: {len(zone_data[zone_name]['mics_nodes_dict'])}\n")
        # elif 'mics_nodes' in zone_data[zone_name]:
        #     print(f"  -> Shape of mics_nodes array: {zone_data[zone_name]['mics_centers'].shape}\n")
        # # DEBUG_end
            
    return zone_data

from scipy.spatial import cKDTree
def resolve_tie_interfaces(parser, zones_mesh, sorted_nodes, default_tolerance=1e-3):
    """
    Normal-Projection Gap Resolver for Constant BEM.
    Pairs Element IDs at TIED pairs between zones by projecting the distance vector
    onto the slave element normal to determine the true physical GAP.
    """
    tie_registry = {}
    if not getattr(parser, 'ties', None):
        return tie_registry

    for tie_dict in parser.ties:
        tie_name = tie_dict.get('name', 'Unnamed_Tie')
        slave_surf_name = tie_dict['slave']
        master_surf_name = tie_dict['master']
        tolerance = tie_dict.get('tolerance', default_tolerance)
        
        # Grab PrePoMax Surface to Element Set links safely
        slave_surface = parser.surfaces.get(slave_surf_name)
        master_surface = parser.surfaces.get(master_surf_name)
        
        if not slave_surface or not master_surface:
            print(f" [ ! ] Warning: Tie '{tie_name}' references missing surfaces.")
            continue
            
        slave_elset = slave_surface.get('elset', slave_surf_name)
        master_elset = master_surface.get('elset', master_surf_name)
        
        slave_eids = parser.elsets.get(slave_elset, [])
        master_eids = parser.elsets.get(master_elset, [])
        n_input_slave_els = len(slave_eids)
        n_input_master_els = len(master_eids)

        # Identify Zone Ownership, Centroids, and Normals
        slave_zone, master_zone = None, None
        slave_data = {}   # stores {eid: (centroid, normal)}
        master_data = {}  # stores {eid: centroid}

        for zone_name, z_mesh in zones_mesh.items():
            # Use existing framework to pull geometry properties
            _, z_centers, _, z_normals, _, _ = prepare_geometry(sorted_nodes, z_mesh['elements'])
            
            # Build lookups for the elements in this zone
            local_elements = list(z_mesh['elements'].keys())
            for idx, eid in enumerate(local_elements):
                if eid in slave_eids:
                    slave_zone = zone_name
                    slave_data[eid] = (z_centers[idx], z_normals[idx])
                if eid in master_eids:
                    master_zone = zone_name
                    master_data[eid] = z_centers[idx]

        if not slave_zone or not master_zone:
            print(f" [ ! ] Warning: Skipping Tie '{tie_name}' - Zone ownership unresolved.")
            continue

        # Spatial Gap Matching Pass
        element_pairs = []
        for seid, (s_center, s_normal) in slave_data.items():
            best_meid = None
            min_lateral_dist = float('inf')
            best_gap = float('inf')

            # Phase 1: Project gap along the normal, filter by tolerance,
            # and map to the closest spatial neighbour to catch dissimilar overlap.
            for meid, m_center in master_data.items():
                # 1. Vector from slave centroid to master centroid
                vec = m_center - s_center
                
                # 2. Calculate true physical GAP via dot product with slave normal
                gap = abs(np.dot(vec, s_normal))
                
                # 3. Check if the element face falls within the gap tolerance
                if gap <= tolerance:
                    # Calculate lateral/sideways distance component to find the overlapping face
                    lateral_vec = vec - np.dot(vec, s_normal) * s_normal
                    lateral_dist = np.linalg.norm(lateral_vec)
                    
                    if lateral_dist < min_lateral_dist:
                        min_lateral_dist = lateral_dist
                        best_meid = meid
                        best_gap = gap

            if best_meid is not None:
                element_pairs.append((best_meid, seid))  # (Master Element ID, Slave Element ID)

        if len(element_pairs) > 0:
            tie_registry[tie_name] = {
                'slave_zone': slave_zone,
                'master_zone': master_zone,
                'element_pairs': element_pairs,
                'tolerance_used': tolerance,
                'n_input_slave_els': n_input_slave_els,
                'n_input_master_els': n_input_master_els
            }

    if len(tie_registry) == 0:
        raise RuntimeError(" [ ! ] 0 tie connections matched. Verify surface normal orientations.")
    return tie_registry

def compute_tie_projection_matrix(tie_registry, zones_mesh, sorted_nodes):
    """
    TIED pair PRE-processing phase geometric projection calculator.
    Builds the mapping matrix [W] for dissimilar interface meshes.
    Returns:
      W_mapping: dict mapping slave_eid -> {master_eid: weight, ...}
      master_eids_set: unique set of all master elements active in the tie
      slave_eids_set: unique set of all slave elements active in the tie
    """
    W_mapping = {}
    master_eids_set = set()
    slave_eids_set = set()
    
    for tie_name, tie_info in tie_registry.items():
        s_zone = tie_info['slave_zone']
        m_zone = tie_info['master_zone']
        tolerance = tie_info['tolerance_used']
        
        # Pull geometric parameters for both interface zones safely
        _, s_centers, _, s_normals, _, _ = prepare_geometry(sorted_nodes, zones_mesh[s_zone]['elements'])
        _, m_centers, _, _, _, _ = prepare_geometry(sorted_nodes, zones_mesh[m_zone]['elements'])
        
        s_eids_local = list(zones_mesh[s_zone]['elements'].keys())
        m_eids_local = list(zones_mesh[m_zone]['elements'].keys())
        
        # Isolate targeted element IDs specified by the tie surface cards
        target_s_eids = {pair[1] for pair in tie_info['element_pairs']}
        target_m_eids = {pair[0] for pair in tie_info['element_pairs']}
        
        for s_idx, seid in enumerate(s_eids_local):
            if seid not in target_s_eids:
                continue
                
            s_center = s_centers[s_idx]
            s_normal = s_normals[s_idx]
            
            overlapping_masters = []
            weights_raw = []
            
            for m_idx, meid in enumerate(m_eids_local):
                if meid not in target_m_eids:
                    continue
                    
                vec = m_centers[m_idx] - s_center
                gap = abs(np.dot(vec, s_normal))
                
                # Check if master element falls within normal proximity tolerance bubble
                if gap <= tolerance:
                    lateral_vec = vec - np.dot(vec, s_normal) * s_normal
                    lateral_dist = np.linalg.norm(lateral_vec)
                    
                    # Prevent division by zero if centroids align perfectly
                    if lateral_dist < 1e-10:
                        lateral_dist = 1e-10
                        
                    overlapping_masters.append(meid)
                    weights_raw.append(1.0 / (lateral_dist ** 2))
            
            if overlapping_masters:
                slave_eids_set.add(seid)
                sum_weights = sum(weights_raw)
                W_mapping[seid] = {}
                
                # Normalize weights to ensure partition of unity (sum of weights = 1.0)
                for m_eid, w_raw in zip(overlapping_masters, weights_raw):
                    norm_w = w_raw / sum_weights
                    W_mapping[seid][m_eid] = norm_w
                    master_eids_set.add(m_eid)
                    
    return W_mapping, list(master_eids_set), list(slave_eids_set)

def get_global_offsets(zones_mesh, tie_registry):
    """
    Computes global row and column matrix offsets for multi-zone Constant BEM allocation.
    Maps localized 0-indexed element offsets per zone, and registers the additional
    equation rows required for *Tie BEM surface TIED pair constraints.
    
    Returns:
        dict: zone_offsets containing {'start_idx': int, 'n_elements': int, 'eid_to_local': dict}
        int: total_matrix_size
    """
    zone_offsets = {}
    current_offset = 0
    
    # 1. Map offsets for standard BEM element variables (Zone by Zone)
    for zone_name, info in zones_mesh.items():
        # Get the true, sorted list of Element IDs local to this zone mesh
        zone_elements = sorted(list(info['elements'].keys()))
        n_elements = len(zone_elements)
        
        # Map input file Element ID -> Local 0-based index inside this zone slice
        eid_to_local = {eid: idx for idx, eid in enumerate(zone_elements)}
        
        zone_offsets[zone_name] = {
            'start_idx': current_offset,
            'n_elements': n_elements,      
            'eid_to_local': eid_to_local  
        }
            # 'local_to_eid': zone_elements  
        
        # In Constant BEM, each zone demands (n_elements) degrees of freedom 
        # for its linear equations block (collocation at element centroids).
        current_offset += n_elements

    # 2. Add extra rows for the *Tie element-face constraints
    total_tie_equations = 0
    for tie_name, info in tie_registry.items():
        # Each matched element pair generates 2 additional coupling equations:
        # 1 for Pressure Continuity, 1 for Flux/Velocity Equilibrium
        total_tie_equations += len(info['element_pairs']) * 2
        
    total_matrix_size = current_offset + total_tie_equations
    
    return zone_offsets, total_matrix_size

def get_element_properties(nodes, connectivity):
    """
    Calculates normal, center (CoG) and surface area of a single element.
    Supports S3 (3 nodes) and S4 (4 nodes).
    It also calcs MAX length edges and aspect ratio per element.
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
    Loops through all elements and gathers their PRE geometry properties into arrays:
    - these to be used in main maths at pre_assembly & pre_mics before the solve.
    - also, other useful info for sanity checks & LOG / print writing.
    """
    nodal_coords = []
    centers = []
    areas = []
    unit_normals = []
    ratios = []
    lengths = []
    
    for eid, conn in elements.items():
        e_n_coords, c, a, n, max_ratio, max_len = get_element_properties(nodes, conn)
        nodal_coords.append(e_n_coords)
        centers.append(c)
        areas.append(a)
        unit_normals.append(n)
        ratios.append(max_ratio)
        lengths.append(max_len)
    return nodal_coords, np.array(centers), np.array(areas), np.array(unit_normals, dtype=np.float64), np.array(ratios), np.array(lengths)

def get_geo_info(elements, centers, areas, normals):
    """
    Calculates mesh volume, total_area, CoG & checks for issues and normal consistency.
    Args:
        elements: dict {el_id: [node_ids]}
        centers:  (N, 3) array of element centroids
        areas:    (N,)   array of element areas
        normals:  (N, 3) array of element unit normals
    """
    
    # --- 1. PHYSICAL CHARACTERISTICS ---
    # np.einsum is an efficient way to do (centers * normals).sum(axis=1)
    # Centroids dot Normals weighted by Area
    dots = np.einsum('ij,ij->i', centers, normals)
    # Total BEM VOLUME
    # Vectorized Volume: V = 1/3 * sum((C_i dot N_i) * A_i)
    volume = np.sum(dots * areas) / 3.0
    # Total BEM AREA
    total_area = np.sum(areas)
    # CoG: Weighted average of centers by area
    # (N,3) * (N,1) summed over N, then divided by scalar total_area
    CoG = np.sum(centers * areas[:, np.newaxis], axis=0) / total_area

    # --- 2. TOPOLOGICAL AUDIT (Edge / Normal Consistency) ---
    # We check if every shared edge by 2 elements is ordered in opposite directions
    edge_map = defaultdict(list)
    conflicts = 0
    
    # We iterate over the values of the dictionary [node_ids]
    for nodes in elements.values():
        num_nodes = len(nodes)
        
        # Build the closed loop of edges for this element
        # TRI: (0,1), (1,2), (2,0)
        # QUAD: (0,1), (1,2), (2,3), (3,0)
        edges = [(nodes[j], nodes[(j + 1) % num_nodes]) for j in range(num_nodes)]
        
        for e in edges:
            # Sorted tuple is our unique edge ID (e.g., (10, 25))
            edge_key = tuple(sorted(e))
            
            # Check if this edge has been visited by another element
            for existing_edge in edge_map[edge_key]:
                # If the existing edge was traversed in the SAME direction, 
                # then one element is 'winding' clockwise and the other counter-clockwise.
                # This is a Right-Hand Rule violation!
                if existing_edge == e:
                    conflicts += 1
            
            edge_map[edge_key].append(e)

    # --- 3. MANIFOLD CHECK (Water-tightness) ---
    # In a closed BEM mesh, every edge MUST be shared by exactly 2 elements
    # If len is 1, there is a hole. If len > 2, it's non-manifold (invalid)
    open_edges = sum(1 for e_list in edge_map.values() if len(e_list) != 2)

    return volume, total_area, CoG, conflicts, open_edges

def get_total_gps(element_nodes):
    total_gps = 0
    for nodes in element_nodes:
        total_gps += (11 if len(nodes) == 3 else 14)
    return total_gps


# ---------------------------------
# ---------------------------------
###
### Gauss Points for QUADS & TRIAS
###
# ---------------------------------
###
### TERMINOLOGY
###
# ---------------------------------
# ξ (Xi) and η (Eta) are the standard Greek letters,
# used to represent the local (or "parent") coordinate system of an element.
# ξ (Xi - pronounced "Ksee" or "Zai"): This represents the horizontal axis of the local square.
# η (Eta - pronounced "Ay-tuh"): This represents the vertical axis of the local square.
# In short: ξ is "Local X" and η is "Local Y." 
# We use Greek letters just to make sure we don't accidentally confuse a gauss point on the element with an input node!

# QUAD4 Gauss points and weights for 2x2 quadrature
# ---------------------------------
# Local coordinates: -/+ 1/sqrt(3): [-0.5773502691896257, 0.5773502691896257]
Q_GP = 1./3.**0.5
# NOTE: The logic here takes into account that BEM normals point AWAY from the fluid.
#       This is against usual RH rules with (+) normals into the material, e.g. in FEA.
#       As a result, it is not trivial which order / signs the GPoints must follow.
QUAD_GP = np.array([
    [-Q_GP, -Q_GP],   # Bottom-Left
    [-Q_GP,  Q_GP],   # Top-Left
    [ Q_GP,  Q_GP],   # Top-Right
    [ Q_GP, -Q_GP],   # Bottom-Right
])
## test influence of order of GPoints 
# QUAD_GP = np.array([
#     [-Q_GP, -Q_GP],   # Bottom-Left
#     [ Q_GP, -Q_GP],   # Bottom-Right
#     [ Q_GP,  Q_GP],   # Top-Right
#     [-Q_GP,  Q_GP],   # Top-Left
# ])

# Standard 2x2 Weights sum to 4.0
QUAD_GW = np.array([1.0, 1.0, 1.0, 1.0])

@njit
def get_quad_points(v1, v2, v3, v4):
    """
    Computes 4 spatial points on a quad4 element surface defined by its 4 nodes.
    2x2 Gauss Quadrature: Instead of calculating the kernel once at the center, we sample it at 4 specific locations (Gauss points) and take a weighted average. 
    For a standard quad element, these points are located at ±0.57735 in local coordinates.
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
#       As a result, it is not trivial which order / signs the GPoints must follow.
# 1.- Local Coords:
    # Define the 9 local (xi, eta) pairs explicitly to match the BEM winding; i.e. node order.
    # We order them row-by-row, but keep the signs aligned with the 'Away' from fluid normal rule.
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

# PRE-processing: we need 1st to know all GPoints we'll have in total.
# This is during pre-processing, so that we make use of fast numpy & numba methods
@njit
def pre_mid_order(element_nodes, element_area):
    """
    PRE-processing step to calculate total number of GPoints & weights.
    'TRIA_3p & QUAD_4p' quadrature.
    """
    n_nodes = len(element_nodes)
    
    # 1. Get number of Integration Points
    if n_nodes == 3: # TRIA3
        pts, wts = get_tri_points(element_nodes[0], element_nodes[1], element_nodes[2])
    else: # QUAD4
        pts, wts = get_quad_points(element_nodes[0], element_nodes[1], element_nodes[2], element_nodes[3])
    sum_w = np.sum(wts)
    
    return pts, wts * element_area / sum_w

@njit
def pre_high_order(element_nodes, element_area):
    """
    PRE-processing step to calculate total number of GPoints & weights.
    'TRIA_7p & QUAD_9p' quadrature.
    """
    n_nodes = len(element_nodes)
    
    # 1. Get Integration Points & Weights
    if n_nodes == 3: # TRIA3
        pts, wts = get_tri_points_7p(element_nodes[0], element_nodes[1], element_nodes[2])
    else: # QUAD4
        pts, wts = get_quad_points_3x3(element_nodes[0], element_nodes[1], element_nodes[2], element_nodes[3])
    sum_w = np.sum(wts)
    
    return pts, wts * element_area / sum_w

# NEW for multi-zone capa
def averaged_at_nodes(nodes, elements, P_bem, bem_areas, elem_id_map, ordered_mic_ids=None, P_mics=None, nodal_id_map=None):
    """
    Averages BEM element results to nodes, plus adds MICS nodes results.
    nodes: dict {nid: [x, y, z]}
    elements: dict {eid: [n1, n2, ...]}
    P_bem: array of complex values (one per element)
    bem_areas: array of BEM areas (one per element)
    elem_id_map: BEM global {eid: index}
    ordered_mic_ids: list of microphone node IDs
    P_mics: array of complex values (one per microphone node)
    nodal_id_map: All nodes global {nid: index}
    nodal_pressures: output array of complex values
    """
    # 1. Size the array exactly to the number of nodes tracked by ParaView
    num_global_nodes = len(nodal_id_map) if nodal_id_map is not None else max(nodes.keys()) + 1
    
    # Initialize buffers
    node_sums = np.zeros(num_global_nodes, dtype=np.complex128)
    area_sums = np.zeros(num_global_nodes, dtype=np.float64)  
    nodal_pressures = np.zeros(num_global_nodes, dtype=np.complex128)

    # 2. Map BEM element results to their constituent nodes
    for i, (eid, conn) in enumerate(elements.items()):
        if i >= len(bem_areas) or i >= len(P_bem):
            continue
        el_idx = elem_id_map[eid]
        area = bem_areas[el_idx]
        val = P_bem[el_idx] * area
        for nid in conn:
            if nodal_id_map is not None:
                if nid in nodal_id_map:
                    vtk_idx = nodal_id_map[nid]
                    node_sums[vtk_idx] += val
                    area_sums[vtk_idx] += area
            else:
                if nid < num_global_nodes:
                    node_sums[nid] += val
                    area_sums[nid] += area

    # 3. Perform the weighted area average for BEM elements
    mask = area_sums > 1e-14
    nodal_pressures[mask] = node_sums[mask] / area_sums[mask]

    # 4. Map microphone results using unified identity mapping
    if ordered_mic_ids is not None and P_mics is not None:
        for i, nid in enumerate(ordered_mic_ids):
            if i < len(P_mics):
                if nodal_id_map is not None:
                    if nid in nodal_id_map:
                        vtk_idx = nodal_id_map[nid]
                        # Direct assignment to the precise ParaView cell row
                        nodal_pressures[vtk_idx] = P_mics[i]
                else:
                    if nid < num_global_nodes:
                        nodal_pressures[nid] = P_mics[i]

    return nodal_pressures
