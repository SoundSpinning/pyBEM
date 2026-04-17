import os
import numpy as np
from constants import SUPPORTED_KEYWORDS, SUB_KEYWORDS, TOP_LOG_LINES
from version import __solver__

class PMXParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # --- Model Data ---
        self.header_comments = ''
        self.nodes = {}
        self.n_mics_nodes = 0
        self.elements = {}        
        self.mics_elements = {}        
        self.nsets = {}          
        self.elsets = {}          
        self.surfaces = {}          
        self.model_str = ''
        
        # --- Materials & Physics ---
        self.materials = {}       
        self.sections = []        
        self.amplitudes = {} # { 'Name': np.array([[freq, val], ...]) }
        self.speed_of_sound = None 
        self.density = None        
        
        # --- Step & BCs ---
        self.frequencies = []
        self.bc_data = []    # List of dicts for PRES, VELO, IMPE

        # --- top LOG lines ---
        self.top_log = None

    def load_model(self):
        """Main entry point: Maps the file and routes blocks to specific logic."""
        if not os.path.exists(self.file_path):
            return False

        with open(self.file_path, 'r') as f:
            # Keep raw lines but strip whitespace
            all_lines = [line.strip() for line in f]

        # STEP 1: Identify where every supported keyword starts and ends
        key_indices = []
        for idx, line in enumerate(all_lines):
            if line.startswith('*') and not line.startswith('**'):
                # Check if the command (first word) is in our supported list
                cmd = line.split(',')[0].upper()
                if cmd in SUPPORTED_KEYWORDS:
                    if cmd in SUB_KEYWORDS:
                        continue
                    else:
                        key_indices.append(idx)

        # Add the end of the file as a "final index"
        key_indices.append(len(all_lines))
        # print(key_indices)

        # STEP 2: Loop through the blocks and route them
        for j in range(len(key_indices) - 1):
            start = key_indices[j]
            end = key_indices[j+1]
            header = all_lines[start]
            data_lines = all_lines[start+1 : end] # All lines until the next * command

            self._route_block(header, data_lines)

        # STEP 3: Final Physics validation before handing to main.py
        self._validate_physics()
        return True

    def _route_block(self, header, data):
        """The Router: Dispatches blocks to specialised parsing logic."""
        h_up = header.upper()
        # setup a string for the parsed lines in bem_model_name.inp

        # --- 0) Heading ---
        if h_up.startswith('*HEADING'):
            head_str = f'***\n{header}\n'
            for line in data:
                if line.startswith('**') or not line: 
                    continue # SKIP COMMENTS
                head_str += line+'\n'
            head_str += '***'
            self.header_comments = head_str
        
        # --- 1) Nodes ---
        elif h_up.startswith('*NODE'):
            self.model_str += f'***\n{header}\n'
            for line in data:
                if line.startswith('**') or not line: 
                    continue # SKIP COMMENTS
                self.model_str += line+'\n'
                p = self._split(line)
                if len(p) >= 4:
                    self.nodes[int(p[0])] = [float(p[1]), float(p[2]), float(p[3])]

        # --- 2) Elements ---
        elif h_up.startswith('*ELEMENT'):
            self.model_str += f'***\n{header}\n'
            # Extract type (e.g., *ELEMENT, TYPE=S3, ELSET=Pipe)
            el_type = self._get_param(header, 'TYPE').upper()
            is_mic = el_type.startswith('MICS_')
            
            for line in data:
                if line.startswith('**') or not line: 
                    continue # SKIP COMMENTS
                self.model_str += line+'\n'
                p = self._split(line)
                eid = int(p[0])
                conn = [int(x) for x in p[1:]]
                if is_mic:
                    self.mics_elements[eid] = conn
                else:
                    self.elements[eid] = conn

        # --- 3) Sets (Elset & Nset) ---
        elif h_up.startswith('*ELSET') or h_up.startswith('*NSET'):
            self.model_str += f'***\n{header}\n'
            is_elset = h_up.startswith('*ELSET')
            name = self._get_param(header, 'ELSET' if is_elset else 'NSET')
            name = name.split('Internal-1_')[-1]
            target = self.elsets if is_elset else self.nsets
            
            if name not in target: target[name] = []
            for line in data:
                if line.startswith('**') or not line: 
                    continue # SKIP COMMENTS
                self.model_str += line+'\n'
                target[name].extend([int(x) for x in self._split(line) if x])
        
        # --- 3B) Surfaces ---
        elif h_up.startswith('*SURFACE'):
            self.model_str += f'***\n{header}\n'
            name = self._get_param(header, 'NAME')
            name = name.split('Internal-1_')[-1]
            
            for line in data:
                if line.startswith('**') or not line: 
                    continue # SKIP COMMENTS
                self.model_str += line+'\n'
                p = self._split(line)
                if len(p) >= 2:
                    self.surfaces[name] = {'elset': p[0].split('Internal-1_')[-1]}

        # --- 4) Materials ---
        elif h_up.startswith('*MATERIAL'):
            self.model_str += f'***\n{header}\n'
            mat_name = self._get_param(header, 'NAME')
            self.materials[mat_name] = {'density': None, 'bulk': None, 'c': None}
            # Note: Density and Acoustic Medium are sub-blocks. 
            # In a block-map, we find them inside the data of the Material block.
            curr_sub = None
            for line in data:
                if line.startswith('**') or not line: 
                    continue # SKIP COMMENTS
                self.model_str += line+'\n'
                if line.startswith('*'):
                    curr_sub = line.upper()
                    continue
                if '*DENSITY' in (curr_sub or ''):
                    self.materials[mat_name]['density'] = float(self._split(line)[0])
                if '*ACOUSTIC MEDIUM' in (curr_sub or ''):
                    self.materials[mat_name]['bulk'] = float(self._split(line)[0])

        # --- 5) Shell Sections (Mapping Elset -> Material) ---
        elif h_up.startswith('*SHELL SECTION'):
            self.model_str += f'***\n{header}\n'
            for line in data:
                self.model_str += line+'\n'
            self.sections.append({
                'elset': self._get_param(header, 'ELSET'),
                'material': self._get_param(header, 'MATERIAL')
            })

        # --- 6) Amplitudes ---
        elif h_up.startswith('*AMPLITUDE'):
            self.model_str += f'***\n{header}\n'
            name = self._get_param(header, 'NAME')
            pts = []
            for line in data:
                if line.startswith('**') or not line: 
                    continue # SKIP COMMENTS
                self.model_str += line+'\n'
                vals = [float(x) for x in self._split(line)]
                # Pairs of freq, value
                for i in range(0, len(vals), 2):
                    if i+1 < len(vals): pts.append([vals[i], vals[i+1]])
            self.amplitudes[name] = np.array(pts)

        # ---  ) Step ---
        elif h_up.startswith('*STEP'):
            self.model_str += f'***\n{header}\n'
        
        # --- 7) Step Type & Frequencies ---
        elif h_up.startswith('*STEADY STATE DYNAMICS'):
            self.model_str += f'***\n{header}\n'
            if data:
                for line in data:
                    self.model_str += line+'\n'
                p = self._split(data[0])
                f_min, f_max, n = float(p[0]), float(p[1]), int(p[2])
                bias = float(p[3]) if len(p) > 3 else 1.0
                if bias == 2.0: self.frequencies = np.geomspace(f_min, f_max, n).tolist()
                else: self.frequencies = np.linspace(f_min, f_max, n).tolist()

        # --- 8) BCs (Boundary, Cload, Impedance) ---
        # Acoustic Pressure, DoF = 8
        elif h_up.startswith('*BOUNDARY'):
            self.model_str += f'***\n{header}\n'
            if 'OP=NEW' in h_up: return
            r_i = self._get_param(header, 'LOAD CASE')
            amp = self._get_param(header, 'AMPLITUDE')
            for line in data:
                if line.startswith('**') or not line: 
                    continue # SKIP COMMENTS
                self.model_str += line+'\n'
                p = self._split(line)
                # PRES is index 3 in *Boundary entries
                if r_i == '1':
                    if amp != None:
                        self.bc_data.append({'type': 'PRES', 'set': p[0], 'val': complex(float(p[3])), 'AMP_real': amp})
                    else:
                        self.bc_data.append({'type': 'PRES', 'set': p[0], 'val': complex(float(p[3]))})
                elif r_i == '2':
                    if amp != None:
                        self.bc_data.append({'type': 'PRES', 'set': p[0], 'val': complex(float(p[3])*1j), 'AMP_imag': amp})
                    else:
                        self.bc_data.append({'type': 'PRES', 'set': p[0], 'val': complex(float(p[3])*1j)})

        # Acoustic Velocity, DoF = 8
        elif h_up.startswith('*CLOAD'):
            self.model_str += f'***\n{header}\n'
            if 'OP=NEW' in h_up: return
            r_i = self._get_param(header, 'LOAD CASE')
            amp = self._get_param(header, 'AMPLITUDE')
            for line in data:
                if line.startswith('**') or not line: 
                    continue # SKIP COMMENTS
                self.model_str += line+'\n'
                p = self._split(line)
                # VELO is index 2 in *Cload
                if r_i == '1':
                    if amp != None:
                        self.bc_data.append({'type': 'VELO', 'set': p[0], 'val': complex(float(p[2])), 'AMP_real': amp})
                    else:
                        self.bc_data.append({'type': 'VELO', 'set': p[0], 'val': complex(float(p[2]))})
                elif r_i == '2':
                    if amp != None:
                        self.bc_data.append({'type': 'VELO', 'set': p[0], 'val': complex(float(p[2])*1j), 'AMP_imag': amp})
                    else:
                        self.bc_data.append({'type': 'VELO', 'set': p[0], 'val': complex(float(p[2])*1j)})

        # Acoustic Impedance
        elif h_up.startswith('*IMPEDANCE'):
            self.model_str += f'***\n{header}\n'
            if 'OP=NEW' in h_up: return
            r_i = self._get_param(header, 'LOAD CASE')
            amp = self._get_param(header, 'AMPLITUDE')
            for line in data:
                if line.startswith('**') or not line: 
                    continue # SKIP COMMENTS
                self.model_str += line+'\n'
                p = self._split(line)
                # IMPE is index 1 in *Impedance
                if r_i == '1':
                    if amp != None:
                        self.bc_data.append({'type': 'IMPE', 'set': p[0], 'val': complex(float(p[1])), 'AMP_real': amp})
                    else:
                        self.bc_data.append({'type': 'IMPE', 'set': p[0], 'val': complex(float(p[1]))})
                elif r_i == '2':
                    if amp != None:
                        self.bc_data.append({'type': 'IMPE', 'set': p[0], 'val': complex(float(p[1])*1j), 'AMP_imag': amp})
                    else:
                        self.bc_data.append({'type': 'IMPE', 'set': p[0], 'val': complex(float(p[1])*1j)})

        # ---  ) End Step ---
        elif h_up.startswith('*END STEP'):
            self.model_str += f'***\n{header}\n'

    def _validate_physics(self):
        """Ensures the assigned material has density and bulk modulus."""
        if not self.sections:
            raise ValueError("No *SHELL SECTION found. Check your .inp file.")
        
        # For now, we use the first valid acoustic material found in sections
        for sec in self.sections:
            m = self.materials.get(sec['material'])
            if m and m['density'] and m['bulk']:
                self.density = m['density']
                self.speed_of_sound = np.sqrt(m['bulk'] / m['density'])
                m['c'] = float(f"{self.speed_of_sound:.2f}")
                return
        raise ValueError("No valid Acoustic Material properties found for assigned elements.")

    def get_bcs(self):
        """
        Converts Surface or Elset-based BCs into per-element values for the solver.
        This handles the PrePoMax / Abaqus hierarchy: BC -> Surface -> ELSET -> Elements.
        """
        bc_dict = {}; log_bc_info = ''
        
        # 1. Pre-resolve Surfaces to Element IDs
        # Mapping: { 'Inlet_Surface': [101, 102, 103...] }
        surface_to_elements = {}
        for surf_name, surf_data in self.surfaces.items():
            elset_name = surf_data.get('elset')
            if elset_name in self.elsets:
                surface_to_elements[surf_name] = self.elsets[elset_name]

        # 2. Process each BC definition
        for bc in self.bc_data:
            target_name = bc['set']
            target_ids = []

            # Step 1: Check if the target is a Surface
            if target_name in surface_to_elements:
                target_ids = surface_to_elements[target_name]
            
            # Step 2: Check if target is an ELSET
            elif target_name in self.elsets:
                target_ids = self.elsets[target_name]
            
            # Removed below as N/A for Prepomax / Abaqus syntax for acoustics
            # # Step 3: Check if target is an NSET (find elements sharing these nodes)
            # elif target_name in self.nsets:
            #     target_nodes = set(self.nsets[target_name])
            #     # Find any element where ALL nodes are in this NSET
            #     for eid, conn in self.elements.items():
            #         if all(nid in target_nodes for nid in conn):
            #             target_ids.append(eid)

            # 3. Assign bc_dict values to the map
            if not target_ids:
                # Log an error if a BC is defined but hits no elements
                log_bc_info += f" ERROR: BC target '{target_name}' could not be resolved to any elements.\n"
                # return log_bc_info
                raise ValueError(f"\n ( !e! ) BC target '{target_name}' could not be resolved to any elements.\n Please check your BCs were applied to the right surface or elset name.\n")

            for eid in target_ids:
                if eid not in bc_dict: 
                    bc_dict[eid] = {}
                    # Assign type (PRES, VELO, IMPE) and value
                    bc_dict[eid][bc['type']] = bc['val']
                    # Keep amplitude curves if they exist
                    for amp in ['AMP_real', 'AMP_imag']:
                        if amp in bc:
                            bc_dict[eid][f'{bc['type']}_{amp}'] = bc[amp]

                else: # if el_id already used then add to existing to get: (real, imag)
                    if bc['type'] in bc_dict[eid]:
                        bc_dict[eid][bc['type']] += bc['val']
                        # Keep amplitude curves if they exist
                        for amp in ['AMP_real', 'AMP_imag']:
                            if amp in bc:
                                bc_dict[eid][f'{bc['type']}_{amp}'] = bc[amp]
                    # Can't mix PRES with other BCs
                    elif bc['type'] != 'PRES' and 'PRES' in bc_dict[eid]:
                        raise ValueError(f"\n ( !e! ) Cannot mix PRES and other BCs type on the same element.\n Please check your '{self.model_name}.inp' file.\n")
                    # Allow for VELO + IMPE mix at same BEM element
                    elif bc['type'] != 'PRES' and bc['type'] not in bc_dict[eid]:
                        # Assign type (VELO, IMPE) and value
                        bc_dict[eid][bc['type']] = bc['val']
                        # Keep amplitude curves if they exist
                        for amp in ['AMP_real', 'AMP_imag']:
                            if amp in bc:
                                bc_dict[eid][f'{bc['type']}_{amp}'] = bc[amp]
                    elif bc['type'] != 'PRES' and bc['type'] in bc_dict[eid]:
                        bc_dict[eid][bc['type']] += bc['val']
                        # Keep amplitude curves if they exist
                        for amp in ['AMP_real', 'AMP_imag']:
                            if amp in bc:
                                bc_dict[eid][f'{bc['type']}_{amp}'] = bc[amp]

        log_bc_info += f" BC-PROCESSING: BC resolution complete ==> ( {len(bc_dict)} ) elements have active BCs."
        return bc_dict, log_bc_info

    def _get_param(self, line, key):
        """Extracts command parameters' values like 'NAME=Pipe'."""
        for p in line.split(','):
            if key.upper() in p.upper() and '=' in p:
                return p.split('=')[1].strip()
        return None

    def _split(self, line):
        return [x.strip() for x in line.split(',') if x.strip()]

    def write_debug_inp(self):
        """The 'Receipt' file for debugging."""
        with open(f"bem_{self.model_name}.inp", 'w') as f:
            f.write(__solver__)
            f.write(self.top_log)
            f.write(self.header_comments)
            f.write(self.model_str)

    def print_model_summary(self):
        self.top_log = TOP_LOG_LINES(self)
        self.write_debug_inp()
        print(self.header_comments)
        print(self.top_log)
