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
    MODEL name: {self.project_name}
    NODES: {len(self.nodes)}
    ELEMENTS: {len(self.elements)}
    NSETS: {list(self.nsets.keys())}
    ELSETS: {list(self.elsets.keys())}
    SURFACES: {self.surfaces}
    AMPLITUDES: {list(self.amplitudes.keys())}
    MATERIALS: {self.materials}
    FREQUENCIES: {len(self.frequencies)}
    BCs: {self.bc_data}
{'-' * 80}
"""
# this is flag that need to turn to INTERIOR or EXTERIOR,
# after pre-processing and checking on the BEM model volume sign.
BEM_TYPE = None