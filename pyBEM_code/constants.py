# True == writes debug info to model_name.log file
DEBUG = True

# Flag that should turn to "INTERIOR" or "EXTERIOR",
# after pre-processing and checking on the BEM model volume sign.
BEM_TYPE = None

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
    MODEL name: "{self.project_name}"
    NODES: {len(self.nodes)}
    ELEMENTS: {len(self.elements)}
    NSETS ( {len(self.nsets)} ): {list(self.nsets.keys())}
    ELSETS  ( {len(self.elsets)} ): {list(self.elsets.keys())}
    SURFACES ( {len(self.surfaces)} ): {self.surfaces}
    AMPLITUDES ( {len(self.amplitudes)} ): {list(self.amplitudes.keys())}
    MATERIALS ( {len(self.materials)} ): {self.materials}
    FREQUENCIES: {len(self.frequencies)}
    BCs ( {len(self.bc_data)} ): {self.bc_data}
{'-' * 80}
"""