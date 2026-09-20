# pyBEM: Multi-Zone Acoustic Solver

`pyBEM` is a Python-based Boundary Element Method (BEM) solver designed for direct collocation acoustic analysis in the frequency domain. Powered by **NumPy** and **Numba**, `pyBEM` supports complex multi-zone acoustics, frequency-dependent boundary conditions, field microphone evaluations, and surface coupling across zones via TIED pairs.

---

## Input Deck & Mesh Rules

`pyBEM` parses model definitions directly from **PrePoMax `*.inp`** input decks:

* **Mesh Requirements:** Domain meshes must be watertight with element normal consistently pointing away from the acoustic domain.
* **ID Handling:** Full support for non-sequential node and element IDs with ID sequence jumps.
* **Zone Definition:** Multi-zone domains are declared natively in PrePoMax by assigning matching material names to BEM and microphone elements per zone.
* **Optional Microphones:** Field-point microphone meshes (`MICS`) can be optionally defined per zone for off-mesh visualization.

---

## Execution & Usage

`pyBEM` is executed directly from the terminal by passing the PrePoMax `*.inp` input file to the main solver script:

```bash
python main.py model.inp [optional: --cpus --debug --Pref --Wref]
```

---

## Key Features

* **Direct Collocation Engine:** Solves independent acoustic zones at element centroids utilizing adaptive Gauss-point integration (TRIA 3/7 GP, QUAD 4/9/14 GP) based on source-to-receiver element distances.
* **TIED Pair Surface Coupling:** Enables simultaneous interior and exterior acoustic analysis by non-conforming mesh tying across zone interfaces via Lagrange collocation constraint logic.
* **Boundary Conditions (BCs):** Native support for Dirichlet (`PRES`), Neumann (`VELO`), Impedance (`IMPE`), and combined Robin (`VELO + IMPE`) conditions, fully compatible with frequency-dependent *AMPLITUDE curves and material damping.
* **Acoustic Power Calculator:** Calculates Sound Power Level (SWL - dB) and Total Sound Power (TSW) sums on all model *SURFACEs. It also provides interface power balance checks across TIED pairs (debug mode).
  * SWL & TSW power plots allow for fast comparisons across different design variants.
  * It is a very useful sanity check at all zone interfaces (TIED pairs), and for interior vs exterior power checks; i.e. conservation of energy checks.
* **ParaView Post-Processing:** Translates elemental solution vectors to nodal averages (`averaged_at_nodes()`) for display on both BEM and microphone (`MICS`) shell elements in ParaView.
* **UX:** Significant effort has been injected on the solver side UX via comprehensive `.log` & `_debug.log` files, in order to inform the user as much as possible.

---

## Open Source Workflow & Solver Architecture

```text
PrePoMax (*.inp) ──> pyBEM ──> ParaView / SWL (power) plots
```

| Component | Function / Module | Purpose |
| --- | --- | --- |
| **Pre-Processing** | `pre_assembly()`, `pre_mics()` | Pre-computes heavy distance maps, coordinate shifts, and static $G$ and $H$ matrices before frequency loops. |
| **Solve Worker** | `frequency_worker()` | Assembles global complex system $A_{\text{global}} x = B_{\text{global}}$, applies BC scaling curves, enforces TIED pair constraints, and solves system per frequency. |
| **Post-Processing** | `averaged_at_nodes()` | Maps elemental surface pressures and microphone outputs to nodal results for visualization in Paraview. |

---

## TIED Pair Interface

The TIED pair coupling allows dissimilar, non-matching surface meshes at zone boundaries to couple seamlessly. This allows for complex assemblies and simultaneous interior / exterior acoustic analysis.

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
