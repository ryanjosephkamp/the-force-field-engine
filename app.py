"""
The Force Field Engine — Interactive Streamlit Dashboard.

Pages
-----
1. 🏠 Home — project overview and molecule selector
2. ⚡ Energy Decomposition — five-term U(r) breakdown
3. 🔴 The Stress Visualizer — bond stress heatmap and 3D view
4. 💥 Break a Bond — interactive bond-stretching experiment
5. 🔧 Parameter Tuning — force-field parameter sensitivity
6. 🌊 Perturbation Lab — coordinate perturbation robustness
7. 🦠 Molecule Comparison — side-by-side energy analysis
8. 📚 Theory & Mathematics — derivations and equations
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.forcefield_engine import (
    ForceField,
    EnergyResult,
    StressResult,
    BondStretchResult,
    Atom,
    BondParam,
    build_molecule,
    compute_energy,
    compute_stress,
    get_bond_labels,
    get_atom_labels,
    topology_summary,
    energy_decomposition_dict,
    PRESET_MOLECULES,
    MOLECULE_DESCRIPTIONS,
    ENERGY_TERM_COLORS,
)
from src.analysis import (
    FullEnergyAnalysis,
    StressAnalysis,
    BondStretchAnalysis,
    ParameterScanAnalysis,
    PerturbationAnalysis,
    analyze_energy,
    analyze_stress,
    analyze_bond_stretch,
    analyze_parameter_scan,
    analyze_perturbation,
    forcefield_summary,
)
from src.visualization import PlotlyRenderer, MatplotlibRenderer


# ═══════════════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="The Force Field Engine",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "🏠 Home",
    "⚡ Energy Decomposition",
    "🔴 The Stress Visualizer",
    "💥 Break a Bond",
    "🔧 Parameter Tuning",
    "🌊 Perturbation Lab",
    "🦠 Molecule Comparison",
    "📚 Theory & Mathematics",
]

FOOTER = """
<div style="text-align: center; padding: 2rem 0 1rem 0;
            color: #888; font-size: 0.85rem;">
  <strong>The Force Field Engine</strong> — Calculating U(r)<br>
  Biophysics Portfolio · Week 16 · Project 1<br>
  Ryan Kamp | University of Cincinnati<br>
  <a href="mailto:kamprj@mail.uc.edu">kamprj@mail.uc.edu</a> |
  <a href="https://github.com/ryanjosephkamp">GitHub</a><br>
  February 23, 2026
</div>
"""


# ═══════════════════════════════════════════════════════════════════════
# Cached Helpers
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _cached_molecule(name: str) -> ForceField:
    """Build and cache a preset molecule."""
    return build_molecule(name)


@st.cache_data(show_spinner=False)
def _cached_energy(name: str) -> FullEnergyAnalysis:
    """Cache energy analysis for a molecule."""
    ff = build_molecule(name)
    return analyze_energy(ff)


@st.cache_data(show_spinner=False)
def _cached_stress(name: str) -> StressAnalysis:
    """Cache stress analysis for a molecule."""
    ff = build_molecule(name)
    return analyze_stress(ff)


@st.cache_data(show_spinner=False)
def _cached_stretch(name: str, bond_index: int, max_disp: float, n_steps: int) -> BondStretchAnalysis:
    """Cache bond-stretch analysis."""
    ff = build_molecule(name)
    return analyze_bond_stretch(ff, bond_index, max_disp, n_steps)


@st.cache_data(show_spinner=False)
def _cached_param_scan(name: str, param: str, bond_idx: int) -> ParameterScanAnalysis:
    """Cache parameter scan analysis."""
    ff = build_molecule(name)
    return analyze_parameter_scan(ff, param, bond_index=bond_idx)


@st.cache_data(show_spinner=False)
def _cached_perturbation(name: str, n_samples: int) -> PerturbationAnalysis:
    """Cache perturbation analysis."""
    ff = build_molecule(name)
    return analyze_perturbation(ff, n_samples=n_samples)


# ═══════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    """Render sidebar navigation and molecule selector."""
    st.sidebar.title("⚛️ The Force Field Engine")
    st.sidebar.markdown("---")

    molecule = st.sidebar.selectbox(
        "Molecule",
        PRESET_MOLECULES,
        index=0,
        help="Select a preset molecular system to analyze.",
    )
    desc = MOLECULE_DESCRIPTIONS.get(molecule, "")
    if desc:
        st.sidebar.caption(desc)

    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "*Week 16, Project 1 — Biophysics Portfolio*\n\n"
        "*Ryan Kamp • University of Cincinnati*"
    )
    return page, molecule


def _energy_sidebar() -> Dict:
    """Sidebar controls for energy decomposition page."""
    return {}


def _stress_sidebar() -> Dict:
    """Sidebar controls for stress visualizer page."""
    show_3d = st.sidebar.checkbox(
        "Show 3D Structure",
        value=True,
        help="Display the 3D molecular structure colored by bond stress.",
    )
    return {"show_3d": show_3d}


def _stretch_sidebar(n_bonds: int) -> Dict:
    """Sidebar controls for bond-stretch page."""
    bond_idx = st.sidebar.slider(
        "Bond Index",
        min_value=0,
        max_value=max(0, n_bonds - 1),
        value=0,
        help="Select which bond to stretch. Index 0 is the first bond in the topology.",
    )
    max_disp = st.sidebar.slider(
        "Max Displacement (Å)",
        min_value=0.5,
        max_value=5.0,
        value=3.0,
        step=0.1,
        help="Maximum displacement beyond the equilibrium bond length r₀.",
    )
    n_steps = st.sidebar.slider(
        "Number of Steps",
        min_value=10,
        max_value=120,
        value=60,
        step=5,
        help="Number of displacement increments from 0 to max displacement.",
    )
    return {"bond_index": bond_idx, "max_displacement": max_disp, "n_steps": n_steps}


def _param_sidebar() -> Dict:
    """Sidebar controls for parameter tuning page."""
    param = st.sidebar.selectbox(
        "Parameter to Scan",
        ["bond_k", "bond_r0", "angle_k", "dielectric"],
        index=0,
        help="Choose a force-field parameter to vary and observe the energy response.",
    )
    bond_idx = st.sidebar.number_input(
        "Bond/Angle Index",
        min_value=0,
        max_value=100,
        value=0,
        help="Index of the bond or angle to modify for bond_k, bond_r0, or angle_k scans.",
    )
    return {"param_name": param, "bond_index": int(bond_idx)}


def _perturb_sidebar() -> Dict:
    """Sidebar controls for perturbation lab page."""
    n_samples = st.sidebar.slider(
        "Samples per Scale",
        min_value=3,
        max_value=20,
        value=5,
        help="Number of random perturbation samples at each noise level.",
    )
    return {"n_samples": n_samples}


# ═══════════════════════════════════════════════════════════════════════
# Pages
# ═══════════════════════════════════════════════════════════════════════

def page_home(molecule: str) -> None:
    """Home page with project overview."""
    st.title("🏠 The Force Field Engine")
    st.markdown(
        "Implementing the classical molecular mechanics potential energy "
        "function **U(r)** to evaluate and visualize protein structure energetics."
    )
    st.markdown("---")

    ff = build_molecule(molecule)
    topo = topology_summary(ff)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Atoms", ff.n_atoms)
    with col2:
        st.metric("Bonds", ff.n_bonds)
    with col3:
        st.metric("Angles", ff.n_angles)
    with col4:
        st.metric("Dihedrals", ff.n_dihedrals)

    with st.expander("ℹ️ About These Metrics"):
        st.markdown(
            f"**{ff.name}** topology overview. The force field evaluates energy "
            f"contributions from {ff.n_bonds} harmonic bonds, {ff.n_angles} "
            f"harmonic angle terms, {ff.n_dihedrals} periodic dihedral torsions, "
            f"plus all non-bonded Lennard-Jones and Coulomb pairs within "
            f"a {ff.cutoff:.1f} Å cutoff."
        )

    st.markdown("### Energy Terms")
    st.markdown("""
| Term | Functional Form | Physical Origin |
|------|----------------|-----------------|
| **Bonds** | $U = \\frac{1}{2}k(r - r_0)^2$ | Covalent bond stretching |
| **Angles** | $U = \\frac{1}{2}k(\\theta - \\theta_0)^2$ | Bond angle bending |
| **Dihedrals** | $U = \\frac{V_n}{2}[1 + \\cos(n\\phi - \\gamma)]$ | Torsional rotation |
| **Van der Waals** | $U = 4\\varepsilon[(\\sigma/r)^{12} - (\\sigma/r)^6]$ | Steric repulsion/dispersion |
| **Electrostatics** | $U = \\frac{332.0636 \\, q_i q_j}{\\varepsilon_r \\, r}$ | Charge–charge interactions |
""")

    with st.expander("ℹ️ About This Table"):
        st.markdown(
            "The five standard terms of a molecular mechanics force field. "
            "Bonded terms (bonds, angles, dihedrals) model covalent geometry, "
            "while non-bonded terms (VdW, electrostatics) capture through-space "
            "interactions. Parameters are from the AMBER ff99 force field "
            "(Cornell et al., 1995)."
        )

    st.markdown("### Available Molecules")
    mol_df = pd.DataFrame([
        {"Molecule": m, "Description": MOLECULE_DESCRIPTIONS.get(m, "")}
        for m in PRESET_MOLECULES
    ])
    st.dataframe(mol_df, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ About These Presets"):
        st.markdown(
            "Each preset builds a fully parameterized molecular system with "
            "AMBER-family bond lengths, force constants, partial charges, and "
            "Lennard-Jones parameters. The alanine dipeptide (22 atoms) is the "
            "gold-standard minimal model for backbone conformational analysis."
        )

    st.markdown(FOOTER, unsafe_allow_html=True)


def page_energy_decomposition(molecule: str) -> None:
    """Energy decomposition page."""
    st.title("⚡ Energy Decomposition")
    st.markdown("Full five-term potential energy breakdown for the selected molecule.")
    st.markdown("---")

    with st.spinner("Computing energy..."):
        analysis = _cached_energy(molecule)

    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Energy", f"{analysis.energy_result.total_energy:.4f} kcal/mol")
    with col2:
        st.metric("Bonded Total", f"{analysis.bonded_total:.4f} kcal/mol")
    with col3:
        st.metric("Non-bonded Total", f"{analysis.nonbonded_total:.4f} kcal/mol")

    with st.expander("ℹ️ About These Metrics"):
        st.markdown(
            f"The total potential energy of **{analysis.molecule_name}** is "
            f"**{analysis.energy_result.total_energy:.4f} kcal/mol**. "
            f"Bonded terms (bonds + angles + dihedrals) contribute "
            f"{analysis.bonded_total:.4f} kcal/mol, and non-bonded terms "
            f"(VdW + electrostatics) contribute {analysis.nonbonded_total:.4f} kcal/mol. "
            f"The dominant contribution is **{analysis.dominant_term}** "
            f"({analysis.dominant_value:.4f} kcal/mol)."
        )

    # Bar chart
    fig_bar = PlotlyRenderer.energy_decomposition(analysis)
    st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("ℹ️ About Energy Decomposition"):
        st.markdown(analysis.explanation)

    # Pie chart
    fig_pie = PlotlyRenderer.energy_pie(analysis)
    st.plotly_chart(fig_pie, use_container_width=True)

    with st.expander("ℹ️ About Composition Pie"):
        st.markdown(
            "The pie chart shows the **absolute** contribution of each energy term "
            "as a fraction of the total absolute energy. This reveals which physical "
            "interaction dominates the energy landscape of this molecule."
        )

    # Bonded vs Non-bonded
    fig_bn = PlotlyRenderer.bonded_vs_nonbonded(analysis)
    st.plotly_chart(fig_bn, use_container_width=True)

    with st.expander("ℹ️ About Bonded vs. Non-bonded"):
        st.markdown(
            f"Bonded terms represent covalent geometry strain (deviations from "
            f"equilibrium bond lengths and angles), while non-bonded terms capture "
            f"through-space interactions. The bonded fraction is "
            f"**{analysis.bonded_fraction:.1%}** of the total absolute energy."
        )

    # Per-bond energies
    fig_pb = PlotlyRenderer.per_bond_energy(analysis)
    st.plotly_chart(fig_pb, use_container_width=True)

    with st.expander("ℹ️ About Per-Bond Energies"):
        st.markdown(
            "Each bar represents one covalent bond's harmonic energy "
            "$U = \\frac{1}{2}k(r - r_0)^2$. High-energy bonds are stretched "
            "or compressed relative to their equilibrium length. This view "
            "pinpoints which bonds contribute most to the bonded energy."
        )

    # Data table
    decomp_df = pd.DataFrame([
        {"Term": k, "Energy (kcal/mol)": f"{v:.6f}"}
        for k, v in analysis.decomposition.items()
    ])
    st.dataframe(decomp_df, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ About This Table"):
        st.markdown("Numerical values of each energy term and the total.")

    st.markdown(FOOTER, unsafe_allow_html=True)


def page_stress_visualizer(molecule: str) -> None:
    """Stress visualizer page."""
    st.title("🔴 The Stress Visualizer")
    st.markdown(
        "Color-code bonds by energy stress: **red** = high stress (stretched), "
        "**blue** = relaxed (at equilibrium)."
    )
    st.markdown("---")

    params = _stress_sidebar()

    with st.spinner("Computing stress..."):
        stress_analysis = _cached_stress(molecule)

    sr = stress_analysis.stress_result

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Stress", f"{sr.mean_stress:.4f}")
    with col2:
        st.metric("Frustrated Residues", stress_analysis.n_frustrated)
    with col3:
        st.metric("Most Stressed Bond", stress_analysis.max_stress_label[:30] if stress_analysis.max_stress_label else "—")

    with st.expander("ℹ️ About These Metrics"):
        st.markdown(
            f"Stress is the normalized bond energy relative to the maximum "
            f"in the system. A stress of 1.0 means the bond has the highest "
            f"harmonic energy. **Frustrated residues** ({stress_analysis.n_frustrated}) "
            f"contain bonds with above-average stress — these often correspond to "
            f"active sites or catalytic centers in real proteins."
        )

    # Heatmap
    fig_hm = PlotlyRenderer.stress_heatmap(stress_analysis)
    st.plotly_chart(fig_hm, use_container_width=True)

    with st.expander("ℹ️ About Stress Heatmap"):
        st.markdown(stress_analysis.explanation)

    # 3D structure
    if params["show_3d"]:
        ff = build_molecule(molecule)
        fig_3d = PlotlyRenderer.molecular_structure_3d(ff, stress_analysis)
        st.plotly_chart(fig_3d, use_container_width=True)

        with st.expander("ℹ️ About 3D Structure"):
            st.markdown(
                "The 3D structure shows atoms as spheres (colored by element) "
                "and bonds as lines colored by stress (blue → red). Rotate and "
                "zoom to explore frustrated regions of the structure."
            )

    # Frustrated regions table
    if sr.frustrated_regions:
        st.markdown("### Frustrated Residues")
        ff_obj = build_molecule(molecule)
        # Build per-residue summary
        residue_info: Dict[int, Dict] = {}
        for rid in sr.frustrated_regions:
            residue_info[rid] = {
                "Residue ID": rid,
                "Residue Name": "",
                "Atoms": 0,
                "Stressed Bonds": 0,
                "Max Bond Stress": 0.0,
                "Mean Bond Stress": 0.0,
                "Max Deviation (Å)": 0.0,
            }
        # Count atoms per frustrated residue and get residue names
        for atom in ff_obj.atoms:
            if atom.residue_id in residue_info:
                residue_info[atom.residue_id]["Atoms"] += 1
                residue_info[atom.residue_id]["Residue Name"] = atom.residue
        # Compute per-residue stress statistics from bonds
        from collections import defaultdict
        _res_stresses: Dict[int, List[float]] = defaultdict(list)
        _res_devs: Dict[int, List[float]] = defaultdict(list)
        for idx, bond in enumerate(ff_obj.bonds):
            a_rid = ff_obj.atoms[bond.atom_i].residue_id
            b_rid = ff_obj.atoms[bond.atom_j].residue_id
            for rid in {a_rid, b_rid}:
                if rid in residue_info:
                    _res_stresses[rid].append(sr.bond_stresses[idx])
                    _res_devs[rid].append(sr.bond_deviations[idx])
        for rid, info in residue_info.items():
            slist = _res_stresses.get(rid, [0.0])
            dlist = _res_devs.get(rid, [0.0])
            above_mean = [s for s in slist if s > sr.mean_stress]
            info["Stressed Bonds"] = len(above_mean)
            info["Max Bond Stress"] = round(max(slist), 4)
            info["Mean Bond Stress"] = round(sum(slist) / len(slist), 4)
            info["Max Deviation (Å)"] = round(max(dlist), 4)

        frust_df = pd.DataFrame(list(residue_info.values()))
        st.dataframe(frust_df, use_container_width=True, hide_index=True)

        with st.expander("ℹ️ About Frustrated Residues"):
            st.markdown(
                "Residues with at least one bond above the mean stress level. "
                "**Stressed Bonds** counts how many bonds within (or touching) "
                "the residue exceed the system-wide mean stress. "
                "**Max Bond Stress** and **Mean Bond Stress** summarize the "
                "normalized stress of all bonds associated with the residue. "
                "**Max Deviation** shows the largest |r − r₀| displacement "
                "in angstroms. In real proteins, frustrated regions are "
                "associated with functional motions, ligand binding, and "
                "enzymatic activity (Ferreiro et al., 2007)."
            )

    st.markdown(FOOTER, unsafe_allow_html=True)


def page_break_a_bond(molecule: str) -> None:
    """Break a Bond interactive page."""
    st.title("💥 Break a Bond")
    st.markdown(
        "Stretch a single bond and watch the energy skyrocket — the harmonic "
        "approximation in action."
    )
    st.markdown("---")

    ff = build_molecule(molecule)
    params = _stretch_sidebar(ff.n_bonds)

    with st.spinner("Stretching bond..."):
        stretch_analysis = _cached_stretch(
            molecule, params["bond_index"],
            params["max_displacement"], params["n_steps"],
        )

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Bond", stretch_analysis.bond_label[:30])
    with col2:
        st.metric("Energy Increase", f"{stretch_analysis.energy_increase:.4f} kcal/mol")
    with col3:
        st.metric("Max Force", f"{stretch_analysis.max_force_approx:.2f} kcal/(mol·Å)")

    with st.expander("ℹ️ About These Metrics"):
        st.markdown(
            f"Stretching **{stretch_analysis.bond_label}** by "
            f"{params['max_displacement']:.1f} Å beyond equilibrium increases "
            f"the total energy by **{stretch_analysis.energy_increase:.4f} kcal/mol**. "
            f"The approximate maximum restoring force (−dU/dr) is "
            f"**{stretch_analysis.max_force_approx:.2f} kcal/(mol·Å)**, which equals "
            f"**{stretch_analysis.max_force_approx * 69.48:.0f} pN** — comparable to "
            f"forces measured in single-molecule pulling experiments."
        )

    # Stretch curve
    fig = PlotlyRenderer.bond_stretch_curve(stretch_analysis)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About This Curve"):
        st.markdown(stretch_analysis.explanation)

    # Data table
    sr = stretch_analysis.stretch_result
    if sr:
        # Show every 5th point
        step = max(1, len(sr.displacements) // 12)
        table_data = [
            {
                "Displacement (Å)": f"{sr.displacements[i]:.3f}",
                "Total Energy": f"{sr.energies[i]:.4f}",
                "Bond Energy": f"{sr.bond_energies_curve[i]:.4f}",
                "VdW Energy": f"{sr.vdw_energies_curve[i]:.4f}",
                "Elec Energy": f"{sr.elec_energies_curve[i]:.4f}",
            }
            for i in range(0, len(sr.displacements), step)
        ]
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        with st.expander("ℹ️ About This Table"):
            st.markdown(
                "Sampled data points from the stretch curve showing how each "
                "energy component changes as the bond is displaced."
            )

    st.markdown(FOOTER, unsafe_allow_html=True)


def page_parameter_tuning(molecule: str) -> None:
    """Parameter tuning page."""
    st.title("🔧 Parameter Tuning")
    st.markdown(
        "Explore how sensitive the total energy is to force-field parameters."
    )
    st.markdown("---")

    params = _param_sidebar()

    with st.spinner("Scanning parameter..."):
        scan_analysis = _cached_param_scan(
            molecule, params["param_name"], params["bond_index"],
        )

    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sensitivity (dU/dp)", f"{scan_analysis.sensitivity:.4f}")
    with col2:
        energy_range = max(scan_analysis.total_energies) - min(scan_analysis.total_energies)
        st.metric("Energy Range", f"{energy_range:.4f} kcal/mol")

    with st.expander("ℹ️ About These Metrics"):
        st.markdown(
            f"The linear sensitivity **dU/dp = {scan_analysis.sensitivity:.4f}** "
            f"measures how much the total energy changes per unit change in "
            f"`{params['param_name']}`. A high absolute sensitivity means the "
            f"energy landscape is strongly influenced by this parameter, "
            f"requiring careful calibration."
        )

    # Scan plot
    fig = PlotlyRenderer.parameter_scan(scan_analysis)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About This Plot"):
        st.markdown(scan_analysis.explanation)

    st.markdown(FOOTER, unsafe_allow_html=True)


def page_perturbation_lab(molecule: str) -> None:
    """Perturbation lab page."""
    st.title("🌊 Perturbation Lab")
    st.markdown(
        "Add Gaussian noise to atomic coordinates and measure energy robustness."
    )
    st.markdown("---")

    params = _perturb_sidebar()

    with st.spinner("Running perturbation analysis..."):
        pert_analysis = _cached_perturbation(molecule, params["n_samples"])

    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Baseline Energy", f"{pert_analysis.baseline_energy:.4f} kcal/mol")
    with col2:
        st.metric("Robustness Score", f"{pert_analysis.robustness_score:.4f}")

    with st.expander("ℹ️ About These Metrics"):
        st.markdown(
            f"The **baseline energy** ({pert_analysis.baseline_energy:.4f} kcal/mol) "
            f"is the energy of the unperturbed structure. The **robustness score** "
            f"({pert_analysis.robustness_score:.4f}) is the ratio of baseline to "
            f"mean energy at the largest perturbation — values near 1.0 indicate "
            f"a broad, stable energy basin."
        )

    # Perturbation plot
    fig = PlotlyRenderer.perturbation_robustness(pert_analysis)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About This Plot"):
        st.markdown(pert_analysis.explanation)

    # Data table
    pert_df = pd.DataFrame([
        {
            "σ (Å)": f"{sc:.3f}",
            "Mean Energy (kcal/mol)": f"{mn:.4f}",
            "Std Dev": f"{sd:.4f}",
        }
        for sc, mn, sd in zip(
            pert_analysis.scales, pert_analysis.mean_energies, pert_analysis.std_energies
        )
    ])
    st.dataframe(pert_df, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ About This Table"):
        st.markdown(
            "Mean and standard deviation of total energy across random "
            "perturbation samples. The baseline (σ = 0) shows zero variance."
        )

    st.markdown(FOOTER, unsafe_allow_html=True)


def page_molecule_comparison(molecule: str) -> None:
    """Molecule comparison page."""
    st.title("🦠 Molecule Comparison")
    st.markdown("Compare energy profiles across all preset molecules.")
    st.markdown("---")

    selected = st.multiselect(
        "Molecules to Compare",
        PRESET_MOLECULES,
        default=PRESET_MOLECULES[:3],
        help="Choose two or more molecules for side-by-side energy comparison.",
    )

    if len(selected) < 2:
        st.warning("Select at least two molecules to compare.")
        st.markdown(FOOTER, unsafe_allow_html=True)
        return

    with st.spinner("Computing energies..."):
        analyses = {}
        for name in selected:
            analyses[name] = _cached_energy(name)

    # Comparison chart
    fig = PlotlyRenderer.molecule_comparison(analyses)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About This Comparison"):
        st.markdown(
            "Grouped bars show each of the five energy components for every "
            "selected molecule. Larger molecules with many non-bonded pairs "
            "tend to have larger VdW and electrostatic contributions. The "
            "salt bridge system typically shows the strongest electrostatic "
            "interaction due to its charged side-chains."
        )

    # Summary table
    summary_data = []
    for name, a in analyses.items():
        summary_data.append({
            "Molecule": name,
            "Total (kcal/mol)": f"{a.energy_result.total_energy:.4f}",
            "Bonds": f"{a.energy_result.bond_energy:.4f}",
            "Angles": f"{a.energy_result.angle_energy:.4f}",
            "Dihedrals": f"{a.energy_result.dihedral_energy:.4f}",
            "VdW": f"{a.energy_result.vdw_energy:.4f}",
            "Electrostatics": f"{a.energy_result.electrostatic_energy:.4f}",
            "Dominant Term": a.dominant_term,
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    with st.expander("ℹ️ About This Table"):
        st.markdown(
            "Full numerical breakdown allowing direct comparison of energy "
            "magnitudes. The dominant term column shows which physical "
            "interaction contributes most to each molecule's energy."
        )

    st.markdown(FOOTER, unsafe_allow_html=True)


def page_theory() -> None:
    """Theory & Mathematics page."""
    st.title("📚 Theory & Mathematics")
    st.markdown("Mathematical foundations of the molecular mechanics force field.")
    st.markdown("---")

    with st.expander("1. The Molecular Mechanics Potential Energy Function"):
        st.markdown(
            "The total potential energy of a molecular system is a function of all "
            "atomic coordinates **r** = {r₁, r₂, …, rₙ}:"
        )
        st.latex(r"U(\mathbf{r}) = U_{\text{bonds}} + U_{\text{angles}} + U_{\text{dihedrals}} + U_{\text{VdW}} + U_{\text{elec}}")
        st.markdown(
            "This decomposition is the cornerstone of molecular mechanics (MM), "
            "enabling efficient evaluation of protein energetics without solving "
            "the Schrödinger equation."
        )

    with st.expander("2. Harmonic Bond Stretching"):
        st.markdown(
            "Each covalent bond is modeled as a harmonic spring (Hooke's law):"
        )
        st.latex(r"U_{\text{bond}} = \frac{1}{2} k_b (r - r_0)^2")
        st.markdown(
            "where $k_b$ is the force constant (kcal/(mol·Å²)), $r$ is the "
            "current bond length, and $r_0$ is the equilibrium length. Typical "
            "values: $k_b ≈ 300$–$500$ kcal/(mol·Å²) for C–C bonds, "
            "$r_0 ≈ 1.53$ Å."
        )

    with st.expander("3. Harmonic Angle Bending"):
        st.markdown("Bond angles are similarly treated with a harmonic potential:")
        st.latex(r"U_{\text{angle}} = \frac{1}{2} k_\theta (\theta - \theta_0)^2")
        st.markdown(
            "where θ is the angle in radians between three bonded atoms (i–j–k), "
            "and θ₀ is the equilibrium angle (e.g., 109.5° = 1.911 rad for "
            "tetrahedral sp³ carbon)."
        )

    with st.expander("4. Periodic Dihedral Torsions"):
        st.markdown(
            "The torsional energy around bond j–k in the chain i–j–k–l uses a "
            "cosine series:"
        )
        st.latex(r"U_{\text{dihedral}} = \frac{V_n}{2} [1 + \cos(n\phi - \gamma)]")
        st.markdown(
            "where $V_n$ is the barrier height, $n$ is the periodicity, $\\phi$ "
            "is the current dihedral angle, and $\\gamma$ is the phase offset. "
            "Multiple terms may be summed for the same dihedral to capture complex "
            "torsional profiles."
        )

    with st.expander("5. Lennard-Jones 12-6 Potential"):
        st.markdown("Non-bonded van der Waals interactions use the LJ potential:")
        st.latex(r"U_{\text{LJ}} = 4\varepsilon \left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right]")
        st.markdown(
            "The $r^{-12}$ term models Pauli repulsion (steric clashes) and the "
            "$r^{-6}$ term models London dispersion attraction. The combining "
            "rules (Lorentz–Berthelot) mix parameters for unlike atom types: "
            "$\\sigma_{ij} = (\\sigma_i + \\sigma_j)/2$, $\\varepsilon_{ij} = "
            "\\sqrt{\\varepsilon_i \\varepsilon_j}$."
        )

    with st.expander("6. Coulomb Electrostatics"):
        st.markdown(
            "Point-charge electrostatics between non-bonded atoms:"
        )
        st.latex(r"U_{\text{elec}} = \frac{332.0636 \, q_i \, q_j}{\varepsilon_r \, r_{ij}}")
        st.markdown(
            "where 332.0636 is the conversion factor to kcal/mol when charges "
            "are in elementary charge units and distances in angstroms. $\\varepsilon_r$ "
            "is the relative permittivity (1.0 for vacuum, ~80 for water)."
        )

    with st.expander("7. Exclusion Rules"):
        st.markdown(
            "Atoms separated by 1 or 2 bonds (1-2 and 1-3 pairs) are excluded "
            "from non-bonded calculations because their interactions are already "
            "captured by the bond and angle terms. Some force fields also scale "
            "1-4 interactions (atoms separated by 3 bonds) by a factor of 0.5 "
            "for electrostatics and 0.5 for VdW. Our implementation excludes "
            "1-2 and 1-3 pairs completely."
        )

    with st.expander("8. Lorentz–Berthelot Combining Rules"):
        st.markdown("For unlike atom pairs $i, j$ with individual LJ parameters:")
        st.latex(r"\sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}")
        st.latex(r"\varepsilon_{ij} = \sqrt{\varepsilon_i \cdot \varepsilon_j}")
        st.markdown(
            "These arithmetic/geometric mean rules are the standard in the AMBER "
            "and CHARMM force field families. They allow a compact parameter set "
            "that scales as $O(N)$ atom types rather than $O(N^2)$ pair types."
        )

    with st.expander("9. Force and Energy Minimization"):
        st.markdown(
            "The force on atom $i$ is the negative gradient of the potential:"
        )
        st.latex(r"\mathbf{F}_i = -\nabla_i U(\mathbf{r})")
        st.markdown(
            "Energy minimization algorithms (steepest descent, conjugate gradient, "
            "L-BFGS) drive the structure toward the nearest local minimum on "
            "the potential energy surface by iteratively following these forces."
        )

    with st.expander("10. The Stress Tensor and Frustrated Regions"):
        st.markdown(
            "Bond stress quantifies how far each bond deviates from "
            "equilibrium. We define normalized stress as:"
        )
        st.latex(r"s_i = \frac{U_{\text{bond},i}}{\max_j(U_{\text{bond},j})}")
        st.markdown(
            "Residues with $s_i$ above the system mean are flagged as "
            "**frustrated**. In protein biophysics, frustrated regions often "
            "correspond to catalytic residues or allosteric sites where "
            "conformational strain is functionally critical (Ferreiro et al., "
            "*PNAS*, 2007)."
        )

    with st.expander("11. Bond Dissociation and the Harmonic Limit"):
        st.markdown(
            "The harmonic potential grows without bound as $r \\to \\infty$, "
            "which is unphysical — real bonds dissociate at finite energy. "
            "The Morse potential accounts for this:"
        )
        st.latex(r"U_{\text{Morse}} = D_e [1 - e^{-\alpha(r - r_0)}]^2")
        st.markdown(
            "where $D_e$ is the dissociation energy and "
            "$\\alpha = \\sqrt{k / (2 D_e)}$. The harmonic approximation is "
            "the second-order Taylor expansion of the Morse potential around "
            "$r_0$ and is accurate for small displacements (< 0.3 Å)."
        )

    with st.expander("12. Units and Conversion Factors"):
        st.markdown("""
| Quantity | Units | Conversion |
|----------|-------|------------|
| Energy | kcal/mol | 1 kcal/mol = 4.184 kJ/mol |
| Distance | Å | 1 Å = 0.1 nm |
| Force | kcal/(mol·Å) | 1 kcal/(mol·Å) = 69.48 pN |
| Charge | e (elementary) | 1 e = 1.602 × 10⁻¹⁹ C |
| Angle | radians | 1 rad = 57.296° |
| Coulomb constant | 332.0636 | kcal·Å/(mol·e²) |
""")

    with st.expander("📖 References"):
        st.markdown("""
1. Cornell, W. D. et al. (1995). A second generation force field for the simulation of proteins, nucleic acids, and organic molecules. *Journal of the American Chemical Society*, 117(19), 5179–5197.
2. Leach, A. R. (2001). *Molecular Modelling: Principles and Applications* (2nd ed.). Prentice Hall.
3. Ponder, J. W. & Case, D. A. (2003). Force fields for protein simulations. *Advances in Protein Chemistry*, 66, 27–85.
4. Ferreiro, D. U., Hegler, J. A., Komives, E. A., & Wolynes, P. G. (2007). Localizing frustration in native proteins and protein assemblies. *Proceedings of the National Academy of Sciences*, 104(50), 19819–19824.
5. Jensen, F. (2007). *Introduction to Computational Chemistry* (2nd ed.). Wiley.
6. Schlick, T. (2010). *Molecular Modeling and Simulation: An Interdisciplinary Guide* (2nd ed.). Springer.
7. Jones, J. E. (1924). On the determination of molecular fields. *Proceedings of the Royal Society of London A*, 106(738), 463–477.
8. Lennard-Jones, J. E. (1931). Cohesion. *Proceedings of the Physical Society*, 43(5), 461–482.
9. Wang, J., Wolf, R. M., Caldwell, J. W., Kollman, P. A., & Case, D. A. (2004). Development and testing of a general amber force field. *Journal of Computational Chemistry*, 25(9), 1157–1174.
10. Morse, P. M. (1929). Diatomic molecules according to the wave mechanics. II. Vibrational levels. *Physical Review*, 34(1), 57–64.
11. Jorgensen, W. L., Maxwell, D. S., & Tirado-Rives, J. (1996). Development and testing of the OPLS all-atom force field on conformational energetics and properties of organic liquids. *Journal of the American Chemical Society*, 118(45), 11225–11236.
12. Frenkel, D. & Smit, B. (2002). *Understanding Molecular Simulation: From Algorithms to Applications* (2nd ed.). Academic Press.
""")

    st.markdown(FOOTER, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    """Application entry point with page dispatch."""
    page, molecule = render_sidebar()

    if page == PAGES[0]:
        page_home(molecule)
    elif page == PAGES[1]:
        page_energy_decomposition(molecule)
    elif page == PAGES[2]:
        page_stress_visualizer(molecule)
    elif page == PAGES[3]:
        page_break_a_bond(molecule)
    elif page == PAGES[4]:
        page_parameter_tuning(molecule)
    elif page == PAGES[5]:
        page_perturbation_lab(molecule)
    elif page == PAGES[6]:
        page_molecule_comparison(molecule)
    elif page == PAGES[7]:
        page_theory()
    else:
        page_home(molecule)


if __name__ == "__main__":
    main()
