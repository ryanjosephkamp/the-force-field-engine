# The Force Field Engine — Calculating U(r)

> **Week 16, Project 1** · Biophysics Portfolio · CS Research Self-Study

A from-scratch molecular mechanics force field engine that computes the total potential energy $U(\mathbf{r}) = U_{\text{bonds}} + U_{\text{angles}} + U_{\text{dihedrals}} + U_{\text{VdW}} + U_{\text{elec}}$ for polypeptide systems using AMBER ff99 parameters. Features per-bond stress visualization, an interactive "Break a Bond" experiment, parameter scanning, and coordinate perturbation analysis.

---

## Overview

| Feature | Description |
|---------|-------------|
| **Harmonic Bonds** | $U = \frac{1}{2}k(r - r_0)^2$ with AMBER ff99 force constants |
| **Harmonic Angles** | $U = \frac{1}{2}k(\theta - \theta_0)^2$ for tetrahedral geometry |
| **Periodic Dihedrals** | $U = \frac{V_n}{2}[1 + \cos(n\phi - \gamma)]$ torsion barriers |
| **Lennard-Jones 12-6** | Van der Waals with Lorentz-Berthelot combining rules |
| **Coulomb Electrostatics** | Partial charges with 332.0636 kcal·Å/(mol·e²) constant |
| **Stress Visualizer** | Per-bond stress coloring to identify frustrated regions |
| **Break a Bond** | Interactive bond-stretching energy experiment |
| **Parameter Scanning** | Sweep force constants and dielectric to explore sensitivity |

## Preset Molecules

| Molecule | Atoms | Description |
|----------|-------|-------------|
| Alanine dipeptide | 22 | Standard Ace-Ala-Nme benchmark |
| Glycine tripeptide | 33 | Extended backbone system |
| Alpha helix (5-res) | 50 | Poly-alanine helix (NeRF coordinates) |
| Beta hairpin | 16 | Backbone-only antiparallel strand |
| Salt bridge | 14 | Lys-Asp electrostatic pair |
| Disulfide bond | 12 | Cys-S-S-Cys covalent cross-link |

## Key Results

- **Non-bonded dominance:** VdW + electrostatic interactions dominate in systems > 20 atoms
- **Harmonic accuracy:** Bond/angle terms reproduce near-equilibrium behavior within chemical accuracy
- **Stress mapping:** Terminal bonds show higher stress than interior backbone bonds; per-residue frustration diagnostics identify stressed bonds, max deviation, and residue-level statistics
- **Break a Bond:** Stretching by 2.0 Å costs hundreds of kcal/mol vs. $k_BT \approx 0.6$ kcal/mol
- **Perturbation sensitivity:** 0.1 Å perturbation induces 10–50 kcal/mol energy change
- **Test coverage:** 109 tests across 23 classes — all passing

---

## Project Structure

```
week_16_project_1/
├── README.md                          # This file
├── app.py                             # Streamlit dashboard (8 pages, ~940 lines)
├── main.py                            # CLI entry point (5 modes)
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Build artifacts exclusion
├── week_16_project_1_outline.md       # Project specification
├── src/
│   ├── __init__.py                    # Package re-exports
│   ├── forcefield_engine.py           # Core engine (~1716 lines)
│   ├── analysis.py                    # Analysis pipelines (~640 lines)
│   └── visualization.py              # Plotly + Matplotlib renderers (~850 lines)
├── tests/
│   └── test_force_field_engine.py     # 109 tests, 23 classes
├── docs/
│   ├── scientific_report.md           # Full scientific report
│   ├── w16p1_force_field_engine_ieee.tex  # IEEE conference paper
│   └── w16p1_force_field_engine_ieee.pdf  # Compiled paper
└── figures/                           # Generated output (gitignored)
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd week_16_project_1
pip install -r requirements.txt
```

### 2. Run the CLI

```bash
# Default: energy analysis of alanine dipeptide
python main.py

# Stress analysis
python main.py --stress

# Break a bond experiment
python main.py --stretch --bond 0 --displacement 2.0

# Compare all preset molecules
python main.py --compare

# Parameter scan
python main.py --scan --param bond_k

# Save figures
python main.py --stress --save
```

### 3. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

---

## Theory — Molecular Mechanics in Brief

The total potential energy function:

$$U(\mathbf{r}) = \sum_{\text{bonds}} \frac{1}{2}k_b(r - r_0)^2 + \sum_{\text{angles}} \frac{1}{2}k_\theta(\theta - \theta_0)^2 + \sum_{\text{dihedrals}} \frac{V_n}{2}[1 + \cos(n\phi - \gamma)] + \sum_{i<j} 4\varepsilon_{ij}\left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^6\right] + \sum_{i<j} \frac{332.0636 \, q_i q_j}{\varepsilon_r \, r_{ij}}$$

Parameters from the AMBER ff99 force field (Cornell et al., 1995) ensure energies in kcal/mol, distances in Å, and angles in radians.

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--molecule` | `alanine_dipeptide` | Preset molecule to analyze |
| `--analyze` | (default mode) | Full energy decomposition |
| `--stress` | — | Per-bond stress analysis |
| `--stretch` | — | Break a Bond experiment |
| `--compare` | — | Compare all preset molecules |
| `--scan` | — | Parameter sensitivity scan |
| `--bond` | `0` | Bond index for stretch mode |
| `--displacement` | `2.0` | Max stretch displacement (Å) |
| `--param` | `bond_k` | Parameter to scan |
| `--save` | `False` | Save figures to `figures/` |
| `--verbose` | `False` | Print additional detail |

---

## Streamlit Dashboard Pages

1. **🏠 Home** — Project overview, molecule selector, energy summary
2. **⚡ Energy Decomposition** — Bar chart + pie chart of all five energy terms
3. **� The Stress Visualizer** — Per-bond stress heatmap, 3D structure, and per-residue frustration table
4. **💥 Break a Bond** — Interactive bond-stretching with real-time energy curve
5. **🎛️ Parameter Tuning** — Sweep force constants, dielectric, and cutoff
6. **🎲 Perturbation Lab** — Gaussian coordinate noise vs. energy stability
7. **📊 Molecule Comparison** — Side-by-side energy profiles across all presets
8. **📚 Theory & Mathematics** — 12 expandable derivation sections plus a 📖 References dropdown

---

## Testing

**109 tests** across **23 test classes** covering:

- Atom, BondParam, AngleParam, DihedralParam dataclass construction
- ForceField topology building and exclusion sets
- Geometry calculations (distance, angle, dihedral)
- Individual energy calculators (bond, angle, dihedral, non-bonded)
- Full energy computation and decomposition
- Stress analysis and frustrated region detection
- Bond stretching experiment
- Coordinate perturbation and parameter scanning
- All six preset molecule builders
- Analysis pipeline functions
- All PlotlyRenderer static methods (11 tests)
- All MatplotlibRenderer static methods (9 tests)
- CLI argument parsing
- End-to-end integration tests
- Edge cases (empty systems, single atoms, very close atoms)

---

## Dependencies

- **numpy** >= 1.24 — Array operations and linear algebra
- **scipy** >= 1.10 — Scientific computing utilities
- **matplotlib** >= 3.7 — Publication-quality figures
- **plotly** >= 5.14 — Interactive visualizations
- **streamlit** >= 1.28 — Web dashboard framework
- **pandas** >= 2.0 — Data manipulation
- **pytest** >= 7.3 — Testing framework

---

## Author

**Ryan Kamp**
Department of Computer Science, University of Cincinnati
kamprj@mail.uc.edu | [GitHub](https://github.com/ryanjosephkamp)
