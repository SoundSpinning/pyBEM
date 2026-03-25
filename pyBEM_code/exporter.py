import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil # Standard Python library for high-level file operations

class PVExporter:
    def __init__(self, project_name, nodes, elements, mics_elements=None):
        self.project_name = project_name
        self.nodes = nodes
        # Store both sets of elements
        self.elements = elements
        self.mics_elements = mics_elements if mics_elements else {}

        # Using the standard reference for air (20 microPa in mm units is 2e-11)
        self.P_REF = 2e-11
        self.pv_dir = "PV"
        self.results_list = [] # Keeps track of (freq, filename) for the .pvd

        self._prepare_directory()

    def _prepare_directory(self):
        """Wipes old results and creates a fresh PV directory."""
        if os.path.exists(self.pv_dir):
            print(f"\n--- Cleaning old results in '{self.pv_dir}' folder ---")
            shutil.rmtree(self.pv_dir) # Deletes the folder and everything in it
        os.makedirs(self.pv_dir)
        
    def calculate_spl(self, pressure_complex):
        """Calculates Sound Pressure Level in dB."""
        p_mag = np.abs(pressure_complex)
        # Avoid log(0)
        p_mag = np.maximum(p_mag, 1e-15)
        return 20 * np.log10(p_mag / self.P_REF)

    def write_vtu(self, freq, nodal_pressures, group_ids=None, velocity_nodal=None):
        """
        Writes a single .vtu file for a specific frequency.
        nodal_pressures: dict or array of complex values
        group_ids: dict or array of integers (1 for BEM, 2 for Mics)
        """
        filename = f"Result_{freq:.1f}Hz.vtu"
        filepath = os.path.join(self.pv_dir, filename)
        
        # 1. Create ID Map (Abaqus ID -> 0-based VTK Index)
        node_ids = sorted(self.nodes.keys())
        id_map = {old_id: i for i, old_id in enumerate(node_ids)}

        # 2. Combine all elements for the mesh definition
        # We merge BEM and MICS into one list for the 'Cells' section
        all_elements = {**self.elements, **self.mics_elements}
        sorted_el_ids = sorted(all_elements.keys())
        
        with open(filepath, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <UnstructuredGrid>\n')
            # NumberOfPoints and NumberOfCells MUST match the data below exactly
            f.write(f'    <Piece NumberOfPoints="{len(node_ids)}" NumberOfCells="{len(all_elements)}">\n')
            
            # --- A) POINTS ---
            f.write('      <Points>\n')
            f.write('        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii">\n')
            for nid in node_ids:
                f.write(f"{self.nodes[nid][0]} {self.nodes[nid][1]} {self.nodes[nid][2]} ")
            f.write('\n        </DataArray>\n')
            f.write('      </Points>\n')

            # --- B) CELLS ---
            f.write('      <Cells>\n')
            f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
            for eid in sorted_el_ids:
                conn = all_elements[eid]
                f.write(" ".join(str(id_map[nid]) for nid in conn) + " ")
            f.write('\n        </DataArray>\n')

            f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
            current_offset = 0
            for eid in sorted_el_ids:
                current_offset += len(all_elements[eid])
                f.write(f"{current_offset} ")
            f.write('\n        </DataArray>\n')

            f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
            for eid in sorted_el_ids:
                etype = 5 if len(all_elements[eid]) == 3 else 9
                f.write(f"{etype} ")
            f.write('\n        </DataArray>\n')
            f.write('      </Cells>\n')

            # --- C) POINT DATA ---
            f.write('      <PointData Vectors="Pressure" Scalars="SPL_dB">\n')

            # 1. Pressure Vector
            f.write('        <DataArray type="Float32" Name="Pressure" NumberOfComponents="3" format="ascii">\n')
            for nid in node_ids:
                val = nodal_pressures[nid]
                f.write(f"{val.real} {val.imag} 0.0 ")
            f.write('\n        </DataArray>\n')

            # 2. SPL (dB)
            f.write('        <DataArray type="Float32" Name="SPL_dB" format="ascii">\n')
            for nid in node_ids:
                p_mag = abs(nodal_pressures[nid])
                spl = 20 * np.log10(max(p_mag, 1e-15) / self.P_REF)
                f.write(f"{spl:.2f} ")
            f.write('\n        </DataArray>\n')

            # 3. Groups_ID
            f.write('        <DataArray type="Int32" Name="Groups_ID" format="ascii">\n')
            for nid in node_ids:
                gid = group_ids.get(nid, 1) if group_ids else 1
                f.write(f"{gid} ")
            f.write('\n        </DataArray>\n')

            f.write('      </PointData>\n')
            f.write('    </Piece>\n') # <--- WAS MISSING
            f.write('  </UnstructuredGrid>\n')
            f.write('</VTKFile>\n')

        self.results_list.append((freq, filename))

    def write_pvd(self):
        """Generates the master .pvd file for the whole simulation."""
        pvd_path = f"{self.project_name}_Results.pvd"
        
        root = ET.Element("VTKFile", type="Collection", version="0.1")
        collection = ET.SubElement(root, "Collection")
        
        for freq, filename in self.results_list:
            # Path must be relative to the .pvd file
            rel_path = os.path.join(self.pv_dir, filename)
            ET.SubElement(collection, "DataSet", timestep=str(freq), group="", part="0", file=rel_path)
            
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        with open(pvd_path, "w") as f:
            f.write(xml_str)
        # print(f"\n--- All DONE! ---")