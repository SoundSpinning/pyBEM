# pyBEM: Multi-Zone Acoustics Solver

![pyBEM Version](https://img.shields.io/badge/pyBEM-v0.5.2--alpha-blue)

`pyBEM` is a Python-based Boundary Element Method (BEM) solver designed for direct collocation acoustic analysis in the frequency domain. Powered by **NumPy** and **Numba**, `pyBEM` supports complex multi-zone acoustics, frequency-dependent boundary conditions and damping, field microphone evaluations, and surface coupling across zones via TIED pair constraints.

---

> [Click to read more: 'How-To & Examples'](how-to_examples/README.md)

---

## Key Features

* **Good UX & Logging:** Comprehensive `.log` (& `_debug.log`, `--debug` flag) output files keep the user informed on model topology at PRE-processing, TIED pairs detailed information, system matrix dimensions per zone. Also for each frequency `pyBEM` writes: solve times (assy & solve), RAM (estimates) and (debug) matrix diagnostics. Furthermore, the power calculator also writes a summary at the bottom of the `.log` file for quick checks when testing model variants.
* **Direct Collocation Engine:**
  * PRE-computes distance maps, coordinate shifts, and static $G$ and $H$ matrices before frequency loops.
  * Solves each acoustic zone at element centroids via an adaptive Gauss-point integration (TRIA 3/7 GP, QUAD 4/9/14 GP) scheme based on source-to-receiver element distances. Applies BCs & damping (freq dependant curves supported) and enforces TIED pair constraints. Assembles and solves global complex system $A_{\text{global}} x = B_{\text{global}}$ per frequency.
* **Multi-Zone Analysis:** Multi-domain (zones) acoustics enables interior, exterior & simultaneous interior / exterior acoustics analysis; e.g. intake / exhaust systems.
* **TIED Pair Constraints:**  Zones are connected via a TIED pair constraint, which can handle *non-conforming meshes* across zone interfaces via Lagrange collocation methods. It also features a *slivers detection* method, which accounts for slight relative nodal placement on similar surface topologies due to mesher's tolerances. This reduces 'numerical noise' to maintain energy transmission.
* **Boundary Conditions (BCs):** Native support for Dirichlet (`PRES`), Neumann (`VELO`), Impedance (`IMPE`), and combined (`VELO + IMPE`) Robin conditions, fully compatible with frequency-dependent `*AMPLITUDE` curves and damping factor.
* **Acoustic Power Calculations & Plots:** Calculates Apparent (Mag), Active (Real) and Reactive (Imaginary) Sound Power Levels (SWL-dB) alongside Total Sound Power ($\text{TSW}$: total sum from all frequencies), plus RMS SPL(dB) and mean Velocity across all model surface (`*SURFACE`) user entries.
  * **Automated:** The calculator fires up by default on each job, after all frequencies have been solved, looping through all surfaces defined by the user in the input (*.inp) file.
  * **TIED Interface Validation:** Generates outputs to verify flux, energy conservation and phase across non-conformal master/slave interfaces. This is triggered in debug mode (`--debug` flag).
  * **SWL(dB) Plotting:** Generates frequency plots (output to a single `*_power.png`) for Apparent, Active, and Reactive sound power levels on logarithmic axes. Fully reflective (zero velocity) surfaces are flagged as `(Rigid)` with zero power in figure's legends.
  * **Text Results:** Outputs a single `*_power.csv` file comprising all calculations per surface for further plotting and/or comparisons across design variants.
  * **Why?:** Surface based power sums are mainly useful in two ways:
    * 1.- Sanity check a setup: input & output surface powers should make some sense with respect to BCs applied. Furthermore, power is a key metric to check on the validity of a multi-zone model on TIED pairs; i.e. conservation of energy.
    * 2.- Once a sanity checked setup is in place, SWL(dB) plot comparisons across design variants, make for fast & informed decision making on a preliminary design direction, even before looking at 3D contours.
* **ParaView Post-Processing:** Translates BEM (normal) elemental results to nodal averaged ones for post-processing both `BEM` and `MICS` (microphones) shell elements in ParaView (`*.pvd` file).
* **Batch Runner:** Fire up script `run_suite.py` to solve all input (`*.inp`) files found in a given folder. E.g. `python run_suite.py .\ --results=None` uses current directory, and will turn off all paraview results writing, while still getting all power outputs (`.png` & `.csv` files) per job.

---

## Open Source Workflow

```text
PrePoMax (*.inp) ──> pyBEM ──> ParaView / SWL (power) plots
```

---

## Input Deck & Mesh Rules

`pyBEM` parses model definitions directly from **PrePoMax `*.inp**` input decks:

* **Mesh Requirements:** Domain meshes must be watertight with element normals consistently pointing away from the acoustic domain.
* **ID Handling:** Full support for non-sequential node and element IDs with ID sequence jumps.
* **Zone Definition:** Multi-zone domains are declared natively in PrePoMax by assigning matching material names to BEM and microphone elements per zone.
* **Optional Microphones:** Field-point microphone meshes (`MICS`) can be optionally defined per zone for off-mesh visualization.

---

## Execution & CLI Arguments

Execute `pyBEM` directly from the terminal by passing a PrePoMax `*.inp` input file to the main solver script; e.g. (assuming *pyBEM = python /full-path-to/main.py $1*):

```bash
pyBEM model.inp --cpus=1 --Pref=2e-11 --Wref=1e-9 --debug
```

Type `pyBEM -h` for latest supported flags; e.g.:

```log
usage: main.py [-h] [--cpus CPUS] [--debug] [--Pref PREF] [--Wref WREF] [--results RESULTS]
               [filename]

pyBEM - V0.5.1-alpha | BEM Acoustics Solver

positional arguments:
  filename           PrePoMax *.inp input file

options:
  -h, --help         show this help message and exit
  --cpus CPUS        Number of CPUs to use for parallel Freqs solve. Defaults to 1 CPU, while
                     still multi-threading for matrix solve. pyBEM sets this automatically at the
                     start based on machine specs, in order to minimise race conditions.
  --debug            Enable debug logging to '*_debug.log' file
  --Pref PREF        dB PRESSURE reference (default: 2e-11MPa)
  --Wref WREF        dB POWER reference (default: 1e-09mW)
  --results RESULTS  Results output (default: paraview). Set to None to turn off. Power results
                     still get output.
```

---

## TIED Pair Interface Coupling

The TIED pair coupling allows dissimilar, non-matching surface meshes at zone boundaries to couple seamlessly, enabling complex assemblies and simultaneous interior / exterior acoustic analysis.

```text
+--------------------------------------+        +--------------------------------+
|               ZONE A                 |        |               ZONE B           |
|         (Master Elements)            |        |          (Slave Elements)      |
|  +------------+------------+------+  |        |  +--------+--------+--------+  |
|  |  Master 1  |  Master 2  | ...  |  |        |  | Slave1 | Slave2 | Slave3 |  |
+--+------------+------------+------+--+        +--+--------+--------+--------+--+
   ==================================              ============================
                            \                             /
                             \   Lagrange Collocation    /
                              \    Interface Coupling   /
                               +-----------------------+

```

### System Matrix Coupling Structure

```text
         [ Primary Unknowns: Element Pressures (p) ]   [ Trailing Columns: Slave Lambda ]
       +---------------------------------------------+----------------------------------+
Row  0 |                                             |                                  |
   .   |              BEM Domain Block               |     Tie Collocation Columns      |
 N_all |                                             |                                  |
       +---------------------------------------------+----------------------------------+
 N_all |                                             |                                  |
   .   |        Lagrange Pressure Continuity         |           Zero Block             |
 Total |         (p_slave - W * p_master = 0)        |                                  |
       +---------------------------------------------+----------------------------------+

```

* **Primal unknowns ($0 \dots N_{\text{all}}-1$):** Holds element centroid pressures ($p$).
* **Trailing unknowns ($N_{\text{all}} \dots N_{\text{all}}+N_{\text{slave}}-1$):** Trailing columns store slave interface velocities ($\lambda = v_{\text{slave}}$).
* **Constraint Rows:** Enforces interface pressure continuity across master-slave patches using weighted interpolation.

---

## Main Solver Structure

```text
pyBEM_code/
├── main.py            # Main entry point, CLI parsing, and solve orchestration
├── pmx_parser.py      # PrePoMax INP deck parser (nodes, elements, materials, sets, surfaces, ties, etc)
├── solver_core.py     # Multi-zone direct collocation system solver, TIED constraints, and more
├                      # Dense [G] & [H] matrix assembly, and Numpy/Numba-accelerated Gauss routines
├── utils.py           # Pre / Post-processing, OS file-lock management and power plotting routines
├── exporter_2.py      # Results writer in ParaView format
└── constants.py       # Physical constants, default reference values and precision thresholds
└── run_suite.py       # Script to batch run all '*.inp' files in a given folder
```

---

> [Click to read more: 'How-To & Examples'](how-to_examples/README.md)

---

![pyBEM examples, 1m pipe variants](how-to_examples/assets/pyBEM_1m_pipes_poster.png)
