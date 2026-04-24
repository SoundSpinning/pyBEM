# True == writes debug info to model_name.log file
DEBUG = True

# Flag that should turn to "INTERIOR" or "EXTERIOR",
# after pre-processing and checking on the BEM model volume sign.
# BEM_TYPE = None

P_REF = 2e-11   # MPa, mm models for air @20C

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

def TOP_LOG_LINES(self): return f"""
{'-' * 80}
*** MODEL SUMMARY ***
    =============
    INPUT file: "{self.file_path}"
    MODEL name: "{self.model_name}"
    NODES:    {len(self.nodes):>6} = BEM: {len(self.nodes)-self.n_mics_nodes:<6} + MICS: {self.n_mics_nodes:<6}
    ELEMENTS: {len(self.elements)+len(self.mics_elements):>6} = BEM: {len(self.elements):<6} + MICS: {len(self.mics_elements):<6}
    NSETS ( {len(self.nsets)} ): {list(self.nsets.keys())}
    ELSETS  ( {len(self.elsets)} ): {list(self.elsets.keys())}
    SURFACES ( {len(self.surfaces)} ): {self.surfaces}
    AMPLITUDES ( {len(self.amplitudes)} ): {list(self.amplitudes.keys())}
    MATERIALS ( {len(self.materials)} ): {self.materials}
    FREQUENCIES: {len(self.frequencies)}  [{min(self.frequencies):.1f}Hz --> {max(self.frequencies):.1f}Hz]
    DAMPING: Input Loss Factor (LF) = {self.damping} | Solver uses Damping Ratio (DR), LF=2DR
    BCs ( {len(self.bc_data)} ): {self.bc_data}
{'-' * 80}
"""