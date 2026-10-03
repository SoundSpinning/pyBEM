# pyBEM: Multi-Zone Acoustic Solver

![pyBEM Version](https://img.shields.io/badge/pyBEM-v0.4.7--alpha-blue)

`pyBEM` is a Python-based Boundary Element Method (BEM) solver designed for direct collocation acoustic analysis in the frequency domain. Powered by **NumPy** and **Numba**, `pyBEM` supports complex multi-zone acoustics, frequency-dependent boundary conditions, field microphone evaluations, and surface coupling across zones via TIED pairs.

---

## Key Features

* **Direct Collocation Engine:** Solves independent acoustic zones at element centroids utilizing adaptive Gauss-point integration (TRIA 3/7 GP, QUAD 4/9/14 GP) based on source-to-receiver element distances.
* **TIED Pair Surface Coupling:** Enables simultaneous interior and exterior acoustic analysis by non-conforming mesh tying across zone interfaces via Lagrange collocation constraint logic.
* **Boundary Conditions (BCs):** Native support for Dirichlet (`PRES`), Neumann (`VELO`), Impedance (`IMPE`), and combined Robin (`VELO + IMPE`) conditions, fully compatible with frequency-dependent `*AMPLITUDE` curves and material damping.
* **Acoustic Power & Energy Conservation Diagnostics:** Calculates Apparent ($S$), Active ($P$), and Reactive ($Q$) Sound Power Levels (SWL in dB) alongside Total Sound Power ($\text{TSW}$) across all model `*SURFACE` sets.
* **TIED Interface Validation:** Verifies flux and energy conservation across non-conformal master/slave interfaces down to four significant figures.
* **Automated 3-Panel Plotting:** Generates headless frequency-sweep plots (`*_power.png`) for Apparent, Active, and Reactive power with logarithmic axes. Inactive surfaces ($<\varepsilon$) are dynamically flagged as `(Rigid)`.
* **OS File-Lock Resiliency:** Employs non-destructive binary file access checking (`get_writable_filepath`) to dynamically detect locked files (e.g., `.csv` open in Excel or `.png` open in an image viewer) and route outputs to safe fallback paths (`_new.csv`, `_new.png`) without interrupting solver execution.
* **ParaView Post-Processing:** Translates elemental solution vectors to nodal averages (`averaged_at_nodes()`) for display on both BEM and microphone (`MICS`) shell elements in ParaView.
* **UX & Logging:** Comprehensive `.log` & `_debug.log` output files keep the user informed on model topology at PRE-processing, system matrix dimensions, and solve times, RAM (estimates) and matrix diagnostics.

---

## Open Source Workflow & Solver Architecture

```text
PrePoMax (*.inp) ──> pyBEM ──> ParaView / SWL (power) plots
```

| Component | Function / Module | Purpose |
| --- | --- | --- |
| **Pre-Processing** | `pre_assembly()`, `pre_mics()` | Pre-computes distance maps, coordinate shifts, and static $G$ and $H$ matrices before frequency loops. |
| **Solve Worker** | `frequency_worker()` | Assembles global complex system $A_{\text{global}} x = B_{\text{global}}$, applies BC scaling curves, enforces TIED pair constraints, and solves system per frequency. |
| **Post-Processing** | `averaged_at_nodes()` | Maps elemental surface pressures and microphone outputs to nodal results for visualization in ParaView. |

---

## Input Deck & Mesh Rules

`pyBEM` parses model definitions directly from **PrePoMax `*.inp**` input decks:

* **Mesh Requirements:** Domain meshes must be watertight with element normals consistently pointing away from the acoustic domain.
* **ID Handling:** Full support for non-sequential node and element IDs with ID sequence jumps.
* **Zone Definition:** Multi-zone domains are declared natively in PrePoMax by assigning matching material names to BEM and microphone elements per zone.
* **Optional Microphones:** Field-point microphone meshes (`MICS`) can be optionally defined per zone for off-mesh visualization.

---

## Execution & CLI Arguments

Execute `pyBEM` directly from the terminal by passing a PrePoMax `*.inp` input file to the main solver script:

```bash
python main.py model.inp --cpus=1 --Pref=2e-11 --Wref=1e-9 --debug=False
```

| Argument | Default | Description |
| --- | --- | --- |
| `None` | *Required* | PrePoMax input deck: `*.inp` model file. |
| `--cpus` | `1` | Number of CPUs to use for parallel Freqs solve.  Defaults to 1 CPU, while still multi-threading for matrix solve. pyBEM sets this automatically at the start based on machine specs, in order to minimise race conditions. |
| `--Pref` | `2e-11` (MPa) | Reference acoustic pressure for Sound Pressure Level calculations ($\text{SPL} = 20 \log_{10}(\vert{}p\vert{} / P_{\text{ref}})$). |
| `--Wref` | `1e-9` (mW) | Reference acoustic power for Sound Power Level calculations ($\text{SWL} = 10 \log_{10}(W / W_{\text{ref}})$). |
| `--debug` | `False` | If `True`: Enables extended logging, including solve and TIED pair interface matrix info. |

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

## Output Deliverables

Upon completion of an analysis, `pyBEM` generates the following structured outputs:

* **ParaView Visualization Files (`*.vtk`):** Nodal-averaged spatial surface pressure ($p$), particle velocity ($v$), and acoustic intensity fields for both BEM zone boundaries and off-mesh `MICS` field point planes.
* **Sound Power Data File (`*_power.csv`):** Tabulated tabular frequency data detailing surface areas ($A$), sound power levels ($\text{SWL}$), and total power components ($\text{TSW}_{\text{real}}$, $\text{TSW}_{\text{imag}}$, $\text{TSW}_{\text{mag}}$) per frequency step and surface set.
* **3-Panel Power Spectrum Plot (`*_power.png`):** High-resolution comparative plots for Apparent Power (SWL Mag), Active Power (SWL Real), and Reactive Power (SWL Imag).
* **Execution & Debug Logs (`*.log`, `*_debug.log`):** Detailed execution logging containing model statistics, mesh topology validation, zone boundary setups, matrix dimensions, and frequency solve timing.

---

## Repository Structure

```text
pyBEM/
├── main.py            # Main entry point, CLI parsing, and solve orchestration
├── utils.py           # Post-processing, OS file-lock management, and SWL plotting routines
├── assembly.py        # Dense G & H matrix assembly and Numba-accelerated Gauss routines
├── solver.py          # Multi-zone direct collocation system solver and TIED constraints
├── parser.py          # PrePoMax INP deck parser (nodes, elements, materials, sets, ties)
└── constants.py       # Physical constants, default reference values, and precision thresholds

```
