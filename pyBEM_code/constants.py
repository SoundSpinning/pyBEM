P_REF = 2e-11   # dB reference for MPa, mm models for air @20C

# List of all supported Abaqus/PrePoMax keywords for pyBEM
SUPPORTED_KEYWORDS = [
    '*HEADING', 
    '*NODE', 
    '*ELEMENT', 
    '*NSET', 
    '*ELSET', 
    '*SURFACE', 
    '*MATERIAL', 
    '*DENSITY', 
    '*ACOUSTIC MEDIUM', 
    '*SHELL SECTION', 
    '*TIE', 
    '*AMPLITUDE', 
    '*STEP', 
    '*STEADY STATE DYNAMICS', 
    '*MODAL DAMPING',
    '*BOUNDARY', 
    '*CLOAD', 
    '*IMPEDANCE', 
    '*END STEP'
]

SUB_KEYWORDS = [
    '*DENSITY', 
    '*ACOUSTIC MEDIUM'
]

def TOP_LOG_LINES(self):
    # --- Helper String Formatters ---
    def fmt_list(lst, max_line=75, indent="        "):
        """Formats a list of strings cleanly across multiple lines."""
        if not lst:
            return "None"
        items = [f"'{x}'" if isinstance(x, str) else str(x) for x in lst]
        line, lines = "", []
        for item in items:
            if line and len(line) + len(item) + 2 > max_line:
                lines.append(line)
                line = item
            else:
                line = f"{line}, {item}" if line else item
        if line:
            lines.append(line)
        return f"\n{indent}".join(lines)

    def fmt_dict_keys(d):
        return fmt_list(list(d.keys())) if d else "None"

    # 1. Surfaces
    surf_lines = []
    for s_name, s_info in self.surfaces.items():
        elset = s_info.get('elset', 'N/A')
        surf_lines.append(f"        * '{s_name}': elset = '{elset}'")
    surf_str = "\n".join(surf_lines) if surf_lines else "        None"

    # 2. Tied Pairs
    tie_lines = []
    for t in self.ties:
        tol = t.get('tolerance', 'Auto')
        tie_lines.append(f"        * [{t['name']}] Master: '{t['master']}' <---> Slave: '{t['slave']}' (Tol: {tol})")
    tie_str = "\n".join(tie_lines) if tie_lines else "        None"

    # 3. Materials
    mat_lines = []
    for m_name, m_props in self.materials.items():
        rho = m_props.get('density', 0.0)
        c = m_props.get('c', 0.0)
        mat_lines.append(f"        * '{m_name}': rho = {rho:.3e}, c = {c:.2f}")
    mat_str = "\n".join(mat_lines) if mat_lines else "        None"

    # 4. BEM Zones
    zone_lines = []
    for z_name, elsets in self.zone_to_elsets.items():
        zone_lines.append(f"        * Zone '{z_name}': elsets = {elsets}")
    zone_str = "\n".join(zone_lines) if zone_lines else "        None"

    # 5. Boundary Conditions
    bc_lines = []
    for bc in self.bc_data:
        b_type = bc.get('type', 'UNKNOWN')
        b_set = bc.get('set', 'N/A')
        b_val = bc.get('val', 0)
        bc_lines.append(f"        * Type: {b_type:<5} | Set: '{b_set}' | Value: {b_val}")
    bc_str = "\n".join(bc_lines) if bc_lines else "        None"

    # 6. Damping
    if isinstance(self.damping, dict):
        lf_val = self.damping.get('value', 0.0)
        damp_str = f"Loss Factor (LF) = {lf_val}"
    elif self.damping:
        damp_str = f"Loss Factor (LF) = {self.damping}"
    else:
        damp_str = "None (0.0)"

    # 7. Frequency Range
    freq_str = f"{len(self.frequencies)} steps [{min(self.frequencies):.1f} Hz --> {max(self.frequencies):.1f} Hz]" if self.frequencies else "None"

    n_bem_nodes = len(self.nodes) - self.n_mics_nodes
    n_tot_nodes = len(self.nodes)
    n_bem_els = len(self.elements)
    n_mics_els = len(self.mics_elements)
    n_tot_els = n_bem_els + n_mics_els

    return f"""
    =====================
    *** MODEL SUMMARY ***
    =====================
    INPUT FILE:   "{self.file_path}"
    MODEL NAME:   "{self.model_name}"

    [ MESH | SETS | SURFACES ]
      NODES:       {n_tot_nodes:<6} = BEM: {n_bem_nodes:<6} + MICS: {self.n_mics_nodes:<6} 
      ELEMENTS:    {n_tot_els:<6} = BEM: {n_bem_els:<6} + MICS: {n_mics_els:<6} 
      NSETS  ( {len(self.nsets)} ):   {fmt_dict_keys(self.nsets)}
      ELSETS ( {len(self.elsets)} ):   {fmt_dict_keys(self.elsets)}
      SURFACES ( {len(self.surfaces)} ):
{surf_str}

    [ BCs | TIED PAIRS | AMPLITUDES ]
      BCs ( {len(self.bc_data)} ):
{bc_str}

      TIED PAIRS ( {len(self.ties)} ):
{tie_str}

      AMPLITUDES ( {len(self.amplitudes)} ):  {fmt_dict_keys(self.amplitudes)}

    [ MATERIALS | ZONES ]
      MATERIALS ( {len(self.materials)} ):
{mat_str}

      BEM ZONES ( {len(self.zone_to_elsets)} ):
{zone_str}

    [ ANALYSIS ]
      FREQUENCIES: {freq_str}
      DAMPING:     {damp_str}
                   (Solver uses Damping Ratio: DR = LF * 0.5)
"""
