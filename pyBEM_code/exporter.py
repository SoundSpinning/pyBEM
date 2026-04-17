import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil # Standard Python library for high-level file operations
from constants import P_REF

class PVExporter:
    def __init__(self, model_name, nodes, sorted_node_ids, nodal_id_map, all_elements, sorted_el_ids, group_ids):
        self.model_name = model_name
        self.nodes = nodes
        self.sorted_node_ids = sorted_node_ids
        self.nodal_id_map = nodal_id_map

        # Stores all elements (BEM & MICS)
        self.all_elements = all_elements
        self.sorted_el_ids = sorted_el_ids

        self.P_REF = P_REF   # see P_REF in constants.py
        self.pv_dir = f"PV_{model_name}"
        self.results_list = [] # Keeps track of (freq, filename) for the .pvd

        self._prepare_directory()
        # we write the mesh in PV format once
        self.fstr_mesh = self._get_PV_mesh()
        self.fstr_groups = self._get_PV_groups(group_ids)

    def _prepare_directory(self):
        """Wipes old results and creates a fresh PV directory."""
        if os.path.exists(self.pv_dir):
            print(f"\n--- Cleaning old results in '{self.pv_dir}' folder ---")
            shutil.rmtree(self.pv_dir) # Deletes the folder and everything in it
        os.makedirs(self.pv_dir)

    def _get_PV_mesh(self):
        fstr_mesh = ('<?xml version="1.0"?>\n')
        fstr_mesh += ('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        fstr_mesh += ('  <UnstructuredGrid>\n')
        # NumberOfPoints and NumberOfCells MUST match the data below exactly
        fstr_mesh += (f'    <Piece NumberOfPoints="{len(self.sorted_node_ids)}" NumberOfCells="{len(self.all_elements)}">\n')
        
        # --- A) POINTS ---
        fstr_mesh += ('      <Points>\n')
        fstr_mesh += ('        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii">\n')
        for nid in self.sorted_node_ids:
            # print(nid, self.nodes[nid])
            fstr_mesh += (f"{self.nodes[nid][0]} {self.nodes[nid][1]} {self.nodes[nid][2]} ")
            # print(f"{self.nodes[nid][0]} {self.nodes[nid][1]} {self.nodes[nid][2]} ")
        fstr_mesh += ('\n        </DataArray>\n')
        fstr_mesh += ('      </Points>\n')

        # --- B) CELLS ---
        fstr_mesh += ('      <Cells>\n')
        fstr_mesh += ('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for eid in self.sorted_el_ids:
            # print(eid, self.all_elements[eid])
            conn = self.all_elements[eid]
            fstr_mesh += (" ".join(str(self.nodal_id_map[nid]) for nid in conn) + " ")
            # print(" ".join(str(self.nodal_id_map[nid]) for nid in conn) + " ")
        fstr_mesh += ('\n        </DataArray>\n')

        fstr_mesh += ('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        current_offset = 0
        for eid in self.sorted_el_ids:
            current_offset += len(self.all_elements[eid])
            fstr_mesh += (f"{current_offset} ")
        fstr_mesh += ('\n        </DataArray>\n')

        fstr_mesh += ('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        for eid in self.sorted_el_ids:
            etype = 5 if len(self.all_elements[eid]) == 3 else 9
            fstr_mesh += (f"{etype} ")
        fstr_mesh += ('\n        </DataArray>\n')
        fstr_mesh += ('      </Cells>\n')
        return fstr_mesh
    
    def _get_PV_groups(self, group_ids):
        # 3. Groups_ID
        fstr_groups = ('        <DataArray type="Int32" Name="Groups_ID" format="ascii">\n')
        for nid in self.sorted_node_ids:
            gid = group_ids.get(nid, 1) if group_ids else 1
            fstr_groups += (f"{gid} ")
        fstr_groups += ('\n        </DataArray>\n')
        return fstr_groups

    def write_vtu(self, freq, nodal_pressures, group_ids=None):
        """
        Writes a single .vtu file per frequency.
        nodal_pressures: array of complex values
        group_ids: dict or array of integers (1 for BEM, 2 for Mics)
        """
        filename = f"Result_{freq:.1f}Hz.vtu"
        filepath = os.path.join(self.pv_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(self.fstr_mesh)

            # --- C) POINT DATA ---
            f.write('      <PointData Vectors="Pressure" Scalars="SPL_dB">\n')

            # 1. Pressure Vector
            f.write('        <DataArray type="Float32" Name="Pressure" NumberOfComponents="3" format="ascii">\n')
            for nid in self.sorted_node_ids:
                val = nodal_pressures[nid]
                f.write(f"{val.real} {val.imag} 0.0 ")
            f.write('\n        </DataArray>\n')

            # 2. SPL (dB)
            f.write('        <DataArray type="Float32" Name="SPL_dB" format="ascii">\n')
            for nid in self.sorted_node_ids:
                p_mag = max(abs(nodal_pressures[nid]), 2e-30)   # avoid log10(0)
                spl = 20 * np.log10(p_mag / self.P_REF)
                f.write(f"{spl:.2f} ")
            f.write('\n        </DataArray>\n')

            # 3. Groups_ID
            f.write(self.fstr_groups)

            f.write('      </PointData>\n')
            f.write('    </Piece>\n')
            f.write('  </UnstructuredGrid>\n')
            f.write('</VTKFile>\n')

        self.results_list.append((freq, filename))

    def write_pvd(self):
        """Generates the master .pvd file for the whole simulation."""
        pvd_path = f"{self.model_name}_Results.pvd"
        
        root = ET.Element("VTKFile", type="Collection", version="0.1")
        collection = ET.SubElement(root, "Collection")
        
        for freq, filename in self.results_list:
            # Path must be relative to the .pvd file
            rel_path = os.path.join(self.pv_dir, filename)
            ET.SubElement(collection, "DataSet", timestep=f'{freq:.2f}', group="", part="0", file=rel_path)
            
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        with open(pvd_path, "w") as f:
            f.write(xml_str)
        # print(f"\n--- All DONE! ---")