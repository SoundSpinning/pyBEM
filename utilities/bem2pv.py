import os
import re
import time
import numpy as np
import meshio
from datetime import timedelta

# --- CONFIGURATION ---
PREF_MPA = 1e-11  # 0 dB Reference (MPa)
OUTPUT_DIR = "PV"
SCALE_POS = 1e3   # m -> mm
SCALE_P = 1e-6    # Pa -> MPa
SCALE_V = 1e3     # m/s -> mm/s
# ---------------------

def get_frequencies(nc_out_path):
    """Parses NC.out for Hz values using precise matching."""
    freqs = []
    if os.path.exists(nc_out_path):
        with open(nc_out_path, 'r') as f:
            for line in f:
                match = re.search(r"Frequency\s*=\s*(\d+\.?\d*)", line, re.IGNORECASE)
                if match: freqs.append(float(match.group(1)))
    return freqs

def parse_elements(filename, node_id_map):
    """Maps NumCalc IDs to row indices, handling 7-field Trias and 8-field Quads."""
    trias, quads, tri_grp, qd_grp = [], [], [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('#', '!', '/', '*')): continue
            p = [int(x) for x in line.split()]
            try:
                # Use map to find the correct row index for each original Node ID
                if len(p) == 8: # QUAD4: ID N1 N2 N3 N4 TYPE 0 0
                    quads.append([node_id_map[p[1]], node_id_map[p[2]], 
                                 node_id_map[p[3]], node_id_map[p[4]]])
                    qd_grp.append(p[5])
                elif len(p) == 7: # TRIA3: ID N1 N2 N3 TYPE 0 0
                    trias.append([node_id_map[p[1]], node_id_map[p[2]], node_id_map[p[3]]])
                    tri_grp.append(p[4])
            except KeyError: continue # Skip if element references a missing node
    return (np.array(trias), np.array(tri_grp)), (np.array(quads), np.array(qd_grp))

def load_numcalc_res(filepath):
    if not os.path.exists(filepath): return None
    data = np.loadtxt(filepath, skiprows=3)
    return data.reshape(1, -1) if data.ndim == 1 else data

def draw_progress_bar(percent, bar_len=40):
    filled_len = int(bar_len * percent)
    bar = '█' * filled_len + '-' * (bar_len - filled_len)
    return f"|{bar}| {percent:3.0%}"

def run_conversion():
    start_time_total = time.time()
    root_name = os.path.basename(os.getcwd())
    pvd_name = f"{root_name}_Results.pvd"
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print(f"--- PROJECT: {root_name.upper()} ---")
    
    # 1. Mesh Loading & ID Mapping
    print("Reading Mesh and Mapping IDs...", end="\r")
    raw_nodes = np.loadtxt("all_Nodes.inc", skiprows=1)
    nodes = raw_nodes[:, 1:] * SCALE_POS
    # Create the map: {Original_ID: Array_Index}
    node_id_map = {int(node_id): i for i, node_id in enumerate(raw_nodes[:, 0])}
    
    (trias, tri_grp), (quads, qd_grp) = parse_elements("all_Elements.inc", node_id_map)
    cells, cell_groups = [], []
    if len(trias) > 0: cells.append(("triangle", trias)); cell_groups.append(tri_grp)
    if len(quads) > 0: cells.append(("quad", quads)); cell_groups.append(qd_grp)
    print(f"Mesh Loaded: {len(nodes)} nodes, {len(trias)+len(quads)} elements.   ")

    # 2. Results Prep
    freqs = get_frequencies("NC.out")
    res_base = "be.out"
    sub_folders = sorted([f for f in os.listdir(res_base) if os.path.isdir(os.path.join(res_base, f))], 
                         key=lambda x: int(x.split('.')[-1]) if '.' in x else 0)
    
    total_y = len(sub_folders)
    print(f"Frequencies to translate: {total_y}\n")
    vtu_list = []

    # 3. Processing Loop
    for i, folder in enumerate(sub_folders):
        start_time_freq = time.time()
        f_val = freqs[i] if i < len(freqs) else 0.0
        path = os.path.join(res_base, folder)
        
        num_n = len(nodes)
        p_data = np.zeros((num_n, 2)) 
        v_data = np.zeros((num_n, 6))

        for res_file, is_vel in [("pBoundary",0), ("pEvalGrid",0), ("vBoundary",1), ("vEvalGrid",1)]:
            data = load_numcalc_res(os.path.join(path, res_file))
            if data is None: continue
            for row in data:
                n_idx = node_id_map.get(int(row[0])) # Use map for results too!
                if n_idx is not None:
                    if not is_vel: p_data[n_idx] = row[1:3] * SCALE_P
                    else: 
                        cols = min(len(row)-1, 6)
                        v_data[n_idx, :cols] = row[1:cols+1] * SCALE_V

        p_mag = np.sqrt(p_data[:, 0]**2 + p_data[:, 1]**2)
        spl_db = 20 * np.log10(np.maximum(p_mag, 1e-18) / PREF_MPA)

        point_data = {
            "Original_Node_ID": raw_nodes[:, 0].astype(int), # The actual IDs from all_Nodes.inc
            "Pressure_Complex_MPa": np.column_stack([p_data, np.zeros(num_n)]),
            "SPL (dB)": spl_db,
            "Vel_X_Complex_mm_s": np.column_stack([v_data[:, 0:2], np.zeros(num_n)]),
            "Vel_Y_Complex_mm_s": np.column_stack([v_data[:, 2:4], np.zeros(num_n)]),
            "Vel_Z_Complex_mm_s": np.column_stack([v_data[:, 4:6], np.zeros(num_n)])
        }

        mesh = meshio.Mesh(points=nodes, cells=cells, point_data=point_data, cell_data={"GroupID": cell_groups})
        # Use .2f for frequency to match your NC.out exactly
        vtu_filename = f"results_{f_val:.2f}Hz.vtu"
        mesh.write(os.path.join(OUTPUT_DIR, vtu_filename))
        vtu_list.append((f_val, vtu_filename))

        # UX Progress
        elapsed_freq = time.time() - start_time_freq
        print(f"{draw_progress_bar((i+1)/total_y)} | {f_val:8.2f} Hz | Time: {elapsed_freq:5.2f}s", end="\r")

    # 4. Finalise
    with open(pvd_name, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n <Collection>\n')
        for f_val, name in vtu_list:
            f.write(f'  <DataSet timestep="{f_val}" file="{OUTPUT_DIR}/{name}"/>\n')
        f.write(' </Collection>\n</VTKFile>')
    
    total_elapsed = str(timedelta(seconds=round(time.time() - start_time_total)))
    print(f"\n\nFINISHED! Total Time: {total_elapsed} | Master File: {pvd_name}")

if __name__ == "__main__":
    run_conversion()
