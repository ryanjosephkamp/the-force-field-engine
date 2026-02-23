# Week 16 - Project 1: "The Force Field Engine" – Calculating U(r)

## Overview

**Week:** 16 (May 5 – May 12)  
**Theme:** Potential Energy Surfaces, Optimization Algorithms, and Normal Modes  
**Goal:** Implement the mathematical function U(r) that defines the "cost" of a protein structure and minimize it to find the native state.

---

## Project Details

### The "Gap" It Fills
Mastery of **Physical Chemistry** and **Intermolecular Forces**.

Biochemists know that "steric clashes are bad" and "salt bridges are good." You will quantify this by implementing the classical potential energy equation:

U = U_bonds + U_angles + U_dihedrals + U_VdW + U_electrostatics

This proves you understand the harmonic approximation of bonds and the physics of non-bonded interactions.

### The Concept
- **Input:** A PDB file and a simplified parameter set (e.g., force constants k, equilibrium lengths r₀).
- **The Math:**
  - **Bonds:** Hooke's Law U = ½k(r - r₀)².
  - **Angles:** Harmonic U = ½k(θ - θ₀)².
  - **Non-bonded:** Lennard-Jones (12-6) + Coulomb's Law (computed via Neighbor List).
- **Output:** The total energy in kcal/mol.

### Novelty/Creative Angle
**"The Stress Visualizer":**
- Color-code the protein bond-by-bond based on energy stress.
- If a bond is stretched too far (high energy), color it **Red**. If it is relaxed, **Blue**.
- This highlights "Frustrated Regions" in the protein structure, which are often active sites or catalytic centers.

### Technical Implementation
- **Language:** Python (NumPy).
- **Data:** Parse standard topology files (like a simplified `.prmtop`).

### The "Paper" & Interactive Element
- *Interactive:* "Break a Bond." User drags an atom. The energy counter skyrockets as the bond stretches.
- *Paper Focus:* "Constructing Molecular Mechanics Force Fields: A Modular Implementation of Harmonic and Non-Bonded Potentials."

---

## Progress Tracking

- [ ] Initial research and planning
- [ ] Core implementation
- [ ] Testing and validation
- [ ] Documentation and paper draft
- [ ] Interactive demo creation
