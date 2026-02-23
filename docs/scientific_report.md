# The Force Field Engine — Scientific Report

## Constructing Molecular Mechanics Force Fields: A Modular Implementation of Harmonic and Non-Bonded Potentials

**Author:** Ryan Kamp
**Affiliation:** Department of Computer Science, University of Cincinnati
**Contact:** kamprj@mail.uc.edu
**GitHub:** [ryanjosephkamp](https://github.com/ryanjosephkamp)
**Date:** February 23, 2026
**Course:** CS Research Self-Study — Biophysics Portfolio (Week 16, Project 1)

---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Theoretical Background](#theoretical-background)
4. [Methods](#methods)
5. [Results](#results)
6. [Discussion](#discussion)
7. [Conclusion](#conclusion)
8. [References](#references)

---

## Abstract

Molecular mechanics force fields underpin virtually all atomistic simulations of biological macromolecules, translating the quantum-mechanical reality of chemical bonding into a computationally tractable classical potential energy function $U(\mathbf{r})$. This project presents a from-scratch Python implementation of the five canonical force field energy terms — harmonic bond stretching, harmonic angle bending, periodic dihedral torsions, Lennard-Jones 12-6 van der Waals interactions, and Coulomb electrostatics — parameterized with AMBER ff99 values. The engine evaluates six preset polypeptide systems (alanine dipeptide through a 5-residue alpha helix), computes per-bond stress to identify frustrated regions, and provides an interactive "Break a Bond" experiment that visualizes the steep energy cost of covalent bond rupture. All energies are reported in kcal/mol with physically meaningful parameters drawn from the Cornell et al. (1995) AMBER force field. The implementation validates against expected energy decompositions and demonstrates the dominance of non-bonded interactions in larger systems.

---

## Introduction

### The Problem

Predicting the three-dimensional structure and stability of proteins requires evaluating how "favorable" a given atomic configuration is. In molecular mechanics, this favorability is quantified by the total potential energy $U(\mathbf{r})$, a function of all $3N$ atomic coordinates. A lower energy indicates a more stable configuration. Computing $U(\mathbf{r})$ efficiently and accurately is the foundational step in energy minimization, molecular dynamics, and Monte Carlo sampling.

### Why Molecular Mechanics?

Quantum mechanics provides exact energies in principle but scales as $\mathcal{O}(N^3)$ to $\mathcal{O}(N^7)$ with system size, making it impractical for proteins containing thousands of atoms. Molecular mechanics replaces the electronic Schrödinger equation with a classical energy function composed of simple analytical terms — springs for bonds, cosines for torsions, power laws for van der Waals interactions. This reduces the cost to $\mathcal{O}(N^2)$ (or $\mathcal{O}(N \log N)$ with cutoffs) while retaining chemical accuracy sufficient for structural biology.

### Project Goals

1. Implement the five standard force field energy terms with AMBER ff99 parameters.
2. Build a modular engine that separates topology (bonds, angles, dihedrals) from energy evaluation.
3. Compute per-bond stress to visualize frustrated regions in protein structures.
4. Provide a "Break a Bond" experiment showing the harmonic energy cost of bond stretching.
5. Enable parameter scanning and coordinate perturbation to explore the sensitivity of $U(\mathbf{r})$.

---

## Theoretical Background

### 1. The Total Potential Energy Function

The molecular mechanics potential energy is decomposed into bonded and non-bonded contributions:

$$U(\mathbf{r}) = U_{\text{bonds}} + U_{\text{angles}} + U_{\text{dihedrals}} + U_{\text{VdW}} + U_{\text{elec}}$$

Each term captures a different physical interaction and is parameterized independently.

### 2. Harmonic Bond Stretching

Covalent bonds are modeled as harmonic oscillators. The energy for bond $b$ between atoms $i$ and $j$ is:

$$U_{\text{bond}} = \frac{1}{2} k_b (r_{ij} - r_0)^2$$

where $k_b$ is the force constant (kcal/(mol·Å²)), $r_{ij}$ is the interatomic distance, and $r_0$ is the equilibrium bond length. Typical values from AMBER ff99: $k_{\text{C-C}} = 340$ kcal/(mol·Å²), $r_0 = 1.526$ Å.

### 3. Harmonic Angle Bending

The angle formed by three consecutively bonded atoms $i$-$j$-$k$ resists deformation from its equilibrium value:

$$U_{\text{angle}} = \frac{1}{2} k_\theta (\theta_{ijk} - \theta_0)^2$$

Tetrahedral sp³ carbon centers have $\theta_0 \approx 109.5°$ with $k_\theta \approx 63$ kcal/(mol·rad²).

### 4. Periodic Dihedral Torsions

Rotation about the central bond of four consecutive atoms is governed by a periodic potential:

$$U_{\text{dihedral}} = \frac{V_n}{2} [1 + \cos(n\phi - \gamma)]$$

where $V_n$ is the barrier height, $n$ is the periodicity (typically 3 for sp³ carbons), $\phi$ is the torsion angle, and $\gamma$ is the phase offset.

### 5. Lennard-Jones 12-6 Potential

Non-bonded van der Waals interactions are modeled with the Lennard-Jones potential:

$$U_{\text{LJ}} = 4\varepsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]$$

The $r^{-12}$ term captures Pauli repulsion at short range, while the $r^{-6}$ term represents London dispersion attraction. Cross-type parameters use the Lorentz-Berthelot combining rules:

$$\sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}, \quad \varepsilon_{ij} = \sqrt{\varepsilon_i \varepsilon_j}$$

### 6. Coulomb Electrostatics

Partial atomic charges interact via Coulomb's law:

$$U_{\text{elec}} = \frac{332.0636 \, q_i q_j}{\varepsilon_r \, r_{ij}}$$

The constant 332.0636 kcal·Å/(mol·e²) converts to kcal/mol when charges are in elementary charge units and distances in angstroms. The relative dielectric $\varepsilon_r$ models solvent screening (1.0 for vacuum, ~80 for water).

### 7. Exclusion Scheme

Atoms separated by one bond (1-2 pairs) or two bonds (1-3 pairs) are excluded from non-bonded calculations. These short-range interactions are already accounted for by the bond and angle terms. The exclusion set is constructed automatically from the bond and angle topology.

### 8. Bond Stress and Frustrated Regions

The "stress" of a bond is defined as the normalized deviation from equilibrium:

$$\text{stress}_b = \frac{U_b}{\max_b U_b}$$

Bonds with stress near 1.0 are maximally strained — these correspond to frustrated regions that may indicate catalytic sites, binding pockets, or conformational strain.

---

## Methods

### Software Architecture

| Module | Responsibility | Lines |
|--------|---------------|-------|
| `forcefield_engine.py` | Atom/bond data structures, energy calculators, preset builders | ~1716 |
| `analysis.py` | Typed result containers, pipeline functions, CLI summaries | ~640 |
| `visualization.py` | PlotlyRenderer (interactive) + MatplotlibRenderer (publication) | ~850 |
| `app.py` | 8-page Streamlit dashboard | ~940 |
| `main.py` | CLI with 5 mutually exclusive modes | ~230 |

### ForceField Data Model

The `ForceField` class maintains:
- A list of `Atom` objects (index, name, element, type, coordinates, charge, mass)
- Lists of `BondParam`, `AngleParam`, and `DihedralParam` objects defining the topology
- An exclusion set (1-2 and 1-3 pairs) built automatically from bonds and angles
- System parameters: cutoff distance, dielectric constant

### Energy Evaluation Protocol

1. **Bond energy:** Iterate over all `BondParam` entries; compute $r_{ij}$ via Euclidean distance; apply harmonic formula.
2. **Angle energy:** For each `AngleParam`, compute the angle $\theta_{ijk}$ via the dot product of bond vectors; apply harmonic formula.
3. **Dihedral energy:** For each `DihedralParam`, compute $\phi$ via the cross-product method; apply periodic cosine formula.
4. **Non-bonded energy:** Double loop over all atom pairs not in the exclusion set; apply distance cutoff; compute LJ + Coulomb.
5. **Total:** Sum all five contributions.

### Preset Molecule Systems

| Molecule | Atoms | Bonds | Angles | Dihedrals | Description |
|----------|-------|-------|--------|-----------|-------------|
| Alanine dipeptide | 22 | 21 | 20 | 8 | Standard benchmark |
| Glycine tripeptide | 33 | 32 | 31 | 12 | Extended backbone |
| Alpha helix (5-res) | 50 | 49 | 67 | 4 | Poly-alanine helix (NeRF) |
| Beta hairpin | 16 | 15 | 12 | 4 | Backbone-only strand |
| Salt bridge | 14 | 13 | 12 | 4 | Lys-Asp charge pair |
| Disulfide bond | 12 | 11 | 10 | 4 | Cys-S-S-Cys cross-link |

### Stress Visualization Protocol

For each bond, the individual harmonic energy is computed. Stresses are normalized to [0, 1] by dividing by the maximum bond energy. Bonds are color-coded on a blue (relaxed) to red (strained) scale.

---

## Results

### 1. Energy Decomposition — Alanine Dipeptide

The alanine dipeptide (Ace-Ala-Nme, 22 atoms) serves as the primary benchmark. The total energy decomposes as:

| Term | Energy (kcal/mol) | Fraction |
|------|-------------------|----------|
| Bonds | ~5–15 | Minor |
| Angles | ~10–30 | Moderate |
| Dihedrals | ~3–8 | Minor |
| Van der Waals | Variable | Dominant at short range |
| Electrostatics | Variable | Dominant overall |
| **Total** | System-dependent | — |

The non-bonded terms (VdW + electrostatics) consistently dominate the total energy, as expected for molecular mechanics force fields.

### 2. Stress Analysis

Bond stress analysis reveals that terminal bonds (N-terminus, C-terminus cappings) tend to be more stressed than backbone peptide bonds, reflecting the idealized geometry of the preset builders. Interior backbone bonds near equilibrium show near-zero stress. The dashboard's Stress Visualizer page provides a per-residue frustration table that reports — for each frustrated residue — the residue name, atom count, number of bonds exceeding the system-wide mean stress, maximum and mean bond stress values, and the largest bond displacement $|r - r_0|$ in angstroms. This multi-column diagnostic enables rapid identification of the most structurally strained regions.

### 3. Break a Bond Experiment

Stretching bond 0 (N-CA) of the alanine dipeptide from equilibrium to +2.0 Å displacement produces a parabolic energy curve characteristic of the harmonic approximation. The energy increases as $\sim k(\Delta r)^2$, reaching several hundred kcal/mol — far exceeding the thermal energy $k_BT \approx 0.6$ kcal/mol at 300 K.

### 4. Parameter Sensitivity

Scanning the bond force constant $k_b$ from 50 to 600 kcal/(mol·Å²) shows a linear relationship with total bond energy contribution. The total energy is most sensitive to the bond force constant when bonds are displaced from equilibrium.

### 5. Perturbation Robustness

Gaussian coordinate perturbation with scale factors 0.01–0.5 Å demonstrates that:
- At 0.01 Å: Energy changes < 1 kcal/mol (thermal noise regime)
- At 0.1 Å: Energy changes ~10–50 kcal/mol (significant strain)
- At 0.5 Å: Energy changes ~100+ kcal/mol (non-physical regime)

### 6. Cross-Molecule Comparison

Larger systems exhibit proportionally larger non-bonded contributions due to the $\mathcal{O}(N^2)$ scaling of pairwise interactions. The alpha helix (50 atoms) shows pronounced electrostatic stabilization from hydrogen-bond-like arrangements, while the salt bridge system is dominated by the strong Coulomb attraction between the charged Lys and Asp side chains.

---

## Discussion

### Strengths of the Implementation

The modular architecture cleanly separates topology construction, energy evaluation, analysis, and visualization. AMBER ff99 parameters ensure physically meaningful energies in kcal/mol. The exclusion scheme correctly prevents double-counting of bonded interactions in the non-bonded sum.

### The Harmonic Approximation

The harmonic bond model $U = \frac{1}{2}k(r - r_0)^2$ is accurate near equilibrium but diverges from reality at large displacements. Real bonds exhibit a Morse-like dissociation curve with a finite dissociation energy. The "Break a Bond" experiment highlights this limitation — the harmonic energy grows without bound, while a real bond would plateau at the dissociation energy (~80–100 kcal/mol for C-C bonds).

### Non-Bonded Dominance

In systems larger than ~20 atoms, non-bonded interactions consistently dominate the energy landscape. This reflects the fundamental physics: bonded terms act only on directly connected atom pairs, while non-bonded terms involve all $N(N-1)/2$ pairs (minus exclusions). Electrostatics are particularly important in polypeptides due to the large partial charges on backbone N, C, and O atoms.

### Limitations

1. **Harmonic approximation:** No bond dissociation, anharmonic effects, or polarization.
2. **Fixed charges:** Atomic charges do not respond to environment (no polarizable force field).
3. **No explicit solvent:** The dielectric constant approximates bulk solvent effects but misses specific hydrogen bonds with water.
4. **Idealized geometries:** Preset molecules use idealized coordinates that may not represent crystal structures.
5. **No periodic boundary conditions:** The system is treated in vacuum without periodic images.

### Connections to Computer Science

Force field evaluation is a canonical example of computational complexity tradeoffs. The naive $O(N^2)$ non-bonded calculation can be accelerated to $O(N \log N)$ via particle-mesh Ewald summation or to $O(N)$ via fast multipole methods. Neighbor lists, cell lists, and Verlet lists are classical spatial data structures used in high-performance MD codes.

### Biological Significance

Force fields enable:
- **Structure refinement:** Minimizing $U(\mathbf{r})$ after X-ray crystallography or cryo-EM
- **Drug design:** Scoring protein-ligand binding poses
- **Protein folding:** Molecular dynamics simulations of folding pathways
- **Mutation analysis:** Predicting the energetic impact of amino acid substitutions

---

## Conclusion

This project demonstrates a complete, from-scratch implementation of the molecular mechanics potential energy function $U(\mathbf{r})$ with five standard energy terms and AMBER ff99 parameters. The engine evaluates six preset polypeptide systems, computes per-bond stress to identify frustrated regions, and provides interactive experiments for bond stretching, parameter scanning, and coordinate perturbation. The implementation validates the expected dominance of non-bonded interactions in polypeptide systems and highlights the limitations of the harmonic approximation through the "Break a Bond" experiment. The modular Python architecture — engine, analysis, visualization — provides a pedagogical platform for exploring the physics underlying all modern biomolecular simulation.

---

## References

1. Cornell, W. D. et al. (1995). "A Second Generation Force Field for the Simulation of Proteins, Nucleic Acids, and Organic Molecules." *J. Am. Chem. Soc.* 117, 5179-5197.
2. Weiner, S. J. et al. (1984). "A New Force Field for Molecular Mechanical Simulation of Nucleic Acids and Proteins." *J. Am. Chem. Soc.* 106, 765-784.
3. Ponder, J. W. & Case, D. A. (2003). "Force Fields for Protein Simulations." *Adv. Protein Chem.* 66, 27-85.
4. Lennard-Jones, J. E. (1924). "On the Determination of Molecular Fields." *Proc. R. Soc. Lond. A* 106, 463-477.
5. Jorgensen, W. L. et al. (1996). "Development and Testing of the OPLS All-Atom Force Field." *J. Am. Chem. Soc.* 118, 11225-11236.
6. MacKerell, A. D. et al. (1998). "All-Atom Empirical Potential for Molecular Modeling and Dynamics Studies of Proteins." *J. Phys. Chem. B* 102, 3586-3616.
7. Wang, J. et al. (2004). "Development and Testing of a General AMBER Force Field." *J. Comput. Chem.* 25, 1157-1174.
8. Duan, Y. et al. (2003). "A Point-Charge Force Field for Molecular Mechanics Simulations of Proteins." *J. Comput. Chem.* 24, 1999-2012.
9. Brooks, B. R. et al. (2009). "CHARMM: The Biomolecular Simulation Program." *J. Comput. Chem.* 30, 1545-1614.
10. Karplus, M. & McCammon, J. A. (2002). "Molecular Dynamics Simulations of Biomolecules." *Nat. Struct. Biol.* 9, 646-652.
11. Schlick, T. (2010). *Molecular Modeling and Simulation: An Interdisciplinary Guide.* 2nd ed. Springer.
12. Leach, A. R. (2001). *Molecular Modelling: Principles and Applications.* 2nd ed. Pearson.
13. Frenkel, D. & Smit, B. (2002). *Understanding Molecular Simulation.* 2nd ed. Academic Press.
14. Allen, M. P. & Tildesley, D. J. (2017). *Computer Simulation of Liquids.* 2nd ed. Oxford University Press.
15. Ferreiro, D. U. et al. (2007). "Localizing Frustration in Native Proteins and Protein Assemblies." *Proc. Natl. Acad. Sci. USA* 104, 19819-19824.
16. Jensen, F. (2007). *Introduction to Computational Chemistry.* 2nd ed. Wiley.
17. Morse, P. M. (1929). "Diatomic Molecules According to the Wave Mechanics. II. Vibrational Levels." *Physical Review* 34, 57-64.
