import os
import numpy as np
import pyvista as pv
import shutil
from constants import P_REF

class PVExporter:
    def __init__(self, model_name, nodes, sorted_node_ids, nodal_id_map, all_elements, sorted_el_ids, group_ids):
        self.model_name = model_name
        self.sorted_node_ids = sorted_node_ids
        self.pv_dir = f"PV_{model_name}"
        self.P_REF = P_REF
        
        # 1. Clean and create the directory
        self._prepare_directory()

        # 2. Build the Mesh Geometry (The static part)
        points = np.array([nodes[nid] for nid in sorted_node_ids], dtype=np.float32)
        
        cells = []
        cell_types = []
        for eid in sorted_el_ids:
            nodes_in_elem = [nodal_id_map[nid] for nid in all_elements[eid]]
            cells.append(len(nodes_in_elem))
            cells.extend(nodes_in_elem)
            cell_types.append(5 if len(nodes_in_elem) == 3 else 9)

        # Create the master mesh object
        self.mesh = pv.UnstructuredGrid(np.array(cells), np.array(cell_types), points)
        
        # Attach static Group IDs (BEM vs MICS)
        gids = np.array([group_ids.get(nid, 1) for nid in sorted_node_ids], dtype=np.int32)
        self.mesh.point_data["Groups_ID"] = gids

        # 3. Store snapshots in a dictionary keyed by frequency
        self.block_collection = {} 

    def _prepare_directory(self):
        """Wipes old results and creates a fresh folder."""
        if os.path.exists(self.pv_dir):
            shutil.rmtree(self.pv_dir)
        os.makedirs(self.pv_dir)

    def add_frequency_step(self, freq, nodal_pressures):
        """
        nodal_pressures: Already sorted via nodal_id_map
        """
        # 1. Create a copy of the geometry for this frequency
        snapshot = self.mesh.copy()
        num_mesh_pts = snapshot.n_points

        # 2. Calculate SPL
        p_mag = np.maximum(np.abs(nodal_pressures), 2e-30)
        spl = 20 * np.log10(p_mag / self.P_REF)

        # 3. Build the Pressure Vector (Real, Imag, 0.0)
        # Ensure we use float32 for ParaView compatibility/speed
        pressure_vec = np.zeros((num_mesh_pts, 3), dtype=np.float32)
        pressure_vec[:, 0] = nodal_pressures.real.astype(np.float32)
        pressure_vec[:, 1] = nodal_pressures.imag.astype(np.float32)
        
        # 4. Attach Data
        # PyVista will now see (N) points and (N) data rows. Perfect match.
        snapshot.point_data["Pressure"] = pressure_vec
        snapshot.point_data["SPL_dB"] = spl.astype(np.float32)
        
        # Store in the collection
        self.block_collection[freq] = snapshot

    def finalise(self):
        """
        Version-safe export. Writes individual binary .vtu files 
        and manually generates the .pvd header.
        """
        pvd_path = f"{self.model_name}_Results.pvd"
        
        # We'll use a basic XML builder for the PVD since PVDWriter is missing
        import xml.etree.ElementTree as ET
        from xml.dom import minidom

        root = ET.Element("VTKFile", type="Collection", version="0.1")
        collection = ET.SubElement(root, "Collection")

        print(f"\n --> Writing {len(self.block_collection)} binary frequency steps")
        
        for freq in sorted(self.block_collection.keys()):
            filename = f"Freq_{freq:.1f}.vtu"
            filepath = os.path.join(self.pv_dir, filename)
            
            # Save the mesh as a high-speed binary VTU
            # 'binary=True' is the default in PyVista save
            self.block_collection[freq].save(filepath)

            # Add entry to the PVD collection
            # Note: ParaView uses 'timestep', and the file path is relative to the .pvd
            ET.SubElement(collection, "DataSet", timestep=str(freq), group="", part="0", file=filepath)

        # Pretty-print and save the PVD
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        with open(pvd_path, "w") as f:
            f.write(xml_str)
            
        print(f"     Export Complete. Results file written to: ( '{pvd_path}' )")
