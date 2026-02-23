"""
The Force Field Engine — Energy Analysis Pipelines.

Provides typed result containers and reusable pipeline functions that
combine engine computations into high-level analyses: full energy
decomposition, stress mapping, bond-stretch experiments, parameter
sensitivity scans, and perturbation robustness tests.

Pipelines
---------
analyze_energy
    Full energy decomposition with per-term breakdown.
analyze_stress
    Bond-by-bond stress mapping for frustrated-region detection.
analyze_bond_stretch
    Interactive bond-stretching energy response curve.
analyze_parameter_scan
    Sensitivity of total energy to force-field parameters.
analyze_perturbation
    Coordinate perturbation robustness across noise levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.forcefield_engine import (
    ForceField,
    EnergyResult,
    StressResult,
    BondStretchResult,
    compute_energy,
    compute_stress,
    stretch_bond,
    perturb_coordinates,
    scan_parameter,
    energy_decomposition_dict,
    get_bond_labels,
    topology_summary,
    ENERGY_TERM_COLORS,
)


# ═══════════════════════════════════════════════════════════════════════
# Analysis Result Data Classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FullEnergyAnalysis:
    """Complete energy decomposition with statistics.

    Attributes
    ----------
    molecule_name : str
        Human-readable name of the molecular system.
    topology : str
        Topology summary string.
    energy_result : EnergyResult
        Raw energy evaluation.
    decomposition : Dict[str, float]
        Energy term names → values (kcal/mol).
    dominant_term : str
        Name of the largest-magnitude energy term.
    dominant_value : float
        Value of the dominant term (kcal/mol).
    bonded_total : float
        Sum of bonds + angles + dihedrals (kcal/mol).
    nonbonded_total : float
        Sum of VdW + electrostatics (kcal/mol).
    bonded_fraction : float
        Fraction of total from bonded terms (0–1).
    explanation : str
        Markdown explanation for Streamlit expanders.
    """

    molecule_name: str = ""
    topology: str = ""
    energy_result: Optional[EnergyResult] = None
    decomposition: Dict[str, float] = field(default_factory=dict)
    dominant_term: str = ""
    dominant_value: float = 0.0
    bonded_total: float = 0.0
    nonbonded_total: float = 0.0
    bonded_fraction: float = 0.0
    explanation: str = ""


@dataclass
class StressAnalysis:
    """Stress visualization analysis for frustrated regions.

    Attributes
    ----------
    molecule_name : str
        Name of the molecular system.
    stress_result : StressResult
        Raw per-bond stress data.
    bond_labels : List[str]
        Human-readable labels for each bond.
    n_frustrated : int
        Number of residues in frustrated regions.
    max_stress_label : str
        Label of the most stressed bond.
    min_stress_label : str
        Label of the most relaxed bond.
    explanation : str
        Markdown explanation for Streamlit expanders.
    """

    molecule_name: str = ""
    stress_result: Optional[StressResult] = None
    bond_labels: List[str] = field(default_factory=list)
    n_frustrated: int = 0
    max_stress_label: str = ""
    min_stress_label: str = ""
    explanation: str = ""


@dataclass
class BondStretchAnalysis:
    """Bond-stretch "Break a Bond" experiment analysis.

    Attributes
    ----------
    molecule_name : str
        Name of the molecular system.
    stretch_result : BondStretchResult
        Raw stretch data.
    bond_label : str
        Label of the stretched bond.
    energy_increase : float
        Energy difference from equilibrium to max displacement (kcal/mol).
    max_force_approx : float
        Approximate maximum restoring force dU/dr (kcal/(mol·Å)).
    explanation : str
        Markdown explanation for Streamlit expanders.
    """

    molecule_name: str = ""
    stretch_result: Optional[BondStretchResult] = None
    bond_label: str = ""
    energy_increase: float = 0.0
    max_force_approx: float = 0.0
    explanation: str = ""


@dataclass
class ParameterScanAnalysis:
    """Parameter sensitivity scan analysis.

    Attributes
    ----------
    molecule_name : str
        Name of the molecular system.
    param_name : str
        Scanned parameter name.
    param_values : List[float]
        Scanned values.
    total_energies : List[float]
        Total energy at each value.
    bond_energies : List[float]
        Bond energy component at each value.
    vdw_energies : List[float]
        VdW energy component at each value.
    sensitivity : float
        Approximate dU/dp slope (kcal/mol per unit).
    explanation : str
        Markdown explanation for Streamlit expanders.
    """

    molecule_name: str = ""
    param_name: str = ""
    param_values: List[float] = field(default_factory=list)
    total_energies: List[float] = field(default_factory=list)
    bond_energies: List[float] = field(default_factory=list)
    vdw_energies: List[float] = field(default_factory=list)
    sensitivity: float = 0.0
    explanation: str = ""


@dataclass
class PerturbationAnalysis:
    """Coordinate perturbation robustness analysis.

    Attributes
    ----------
    molecule_name : str
        Name of the molecular system.
    scales : List[float]
        Perturbation magnitudes (Å).
    mean_energies : List[float]
        Mean energy at each perturbation scale.
    std_energies : List[float]
        Std-dev of energy at each scale.
    baseline_energy : float
        Energy of the unperturbed structure.
    robustness_score : float
        Ratio of baseline energy to mean energy at largest perturbation.
    explanation : str
        Markdown explanation for Streamlit expanders.
    """

    molecule_name: str = ""
    scales: List[float] = field(default_factory=list)
    mean_energies: List[float] = field(default_factory=list)
    std_energies: List[float] = field(default_factory=list)
    baseline_energy: float = 0.0
    robustness_score: float = 0.0
    explanation: str = ""


# ═══════════════════════════════════════════════════════════════════════
# Analysis Functions
# ═══════════════════════════════════════════════════════════════════════

def analyze_energy(ff: ForceField) -> FullEnergyAnalysis:
    """Full energy decomposition pipeline.

    Parameters
    ----------
    ff : ForceField
        The molecular system.

    Returns
    -------
    FullEnergyAnalysis
        Comprehensive energy breakdown.
    """
    result = compute_energy(ff)
    decomp = energy_decomposition_dict(result)
    # Exclude "Total" for dominant-term analysis
    terms = {k: v for k, v in decomp.items() if k != "Total"}
    dominant = max(terms, key=lambda k: abs(terms[k]))
    bonded = result.bond_energy + result.angle_energy + result.dihedral_energy
    nonbonded = result.vdw_energy + result.electrostatic_energy
    total_abs = abs(bonded) + abs(nonbonded)
    bonded_frac = abs(bonded) / total_abs if total_abs > 1e-12 else 0.5

    explanation = _build_energy_explanation(ff, result, decomp, dominant, bonded, nonbonded)

    return FullEnergyAnalysis(
        molecule_name=ff.name,
        topology=topology_summary(ff),
        energy_result=result,
        decomposition=decomp,
        dominant_term=dominant,
        dominant_value=terms[dominant],
        bonded_total=bonded,
        nonbonded_total=nonbonded,
        bonded_fraction=bonded_frac,
        explanation=explanation,
    )


def analyze_stress(ff: ForceField) -> StressAnalysis:
    """Bond-by-bond stress analysis pipeline.

    Parameters
    ----------
    ff : ForceField
        The molecular system.

    Returns
    -------
    StressAnalysis
        Stress mapping for frustrated-region visualization.
    """
    stress = compute_stress(ff)
    labels = get_bond_labels(ff)
    max_label = labels[stress.max_stress_bond] if labels else ""
    min_label = labels[stress.min_stress_bond] if labels else ""

    explanation = _build_stress_explanation(ff, stress, labels, max_label, min_label)

    return StressAnalysis(
        molecule_name=ff.name,
        stress_result=stress,
        bond_labels=labels,
        n_frustrated=len(stress.frustrated_regions),
        max_stress_label=max_label,
        min_stress_label=min_label,
        explanation=explanation,
    )


def analyze_bond_stretch(
    ff: ForceField,
    bond_index: int = 0,
    max_displacement: float = 3.0,
    n_steps: int = 60,
) -> BondStretchAnalysis:
    """Bond-stretch "Break a Bond" experiment pipeline.

    Parameters
    ----------
    ff : ForceField
        The molecular system.
    bond_index : int
        Index of the bond to stretch.
    max_displacement : float
        Maximum displacement beyond equilibrium (Å).
    n_steps : int
        Number of steps.

    Returns
    -------
    BondStretchAnalysis
        Energy response curve data.
    """
    stretch = stretch_bond(ff, bond_index, max_displacement, n_steps)
    labels = get_bond_labels(ff)
    label = labels[bond_index] if bond_index < len(labels) else f"Bond {bond_index}"
    energy_inc = stretch.breaking_energy - stretch.equilibrium_energy

    # Numerical derivative for approximate max force
    if len(stretch.energies) > 1:
        de = np.diff(stretch.energies)
        dd = np.diff(stretch.displacements)
        forces = np.abs(de / (np.array(dd) + 1e-15))
        max_force = float(np.max(forces))
    else:
        max_force = 0.0

    explanation = _build_stretch_explanation(label, energy_inc, max_force, stretch)

    return BondStretchAnalysis(
        molecule_name=ff.name,
        stretch_result=stretch,
        bond_label=label,
        energy_increase=energy_inc,
        max_force_approx=max_force,
        explanation=explanation,
    )


def analyze_parameter_scan(
    ff: ForceField,
    param_name: str = "bond_k",
    values: Optional[List[float]] = None,
    bond_index: int = 0,
) -> ParameterScanAnalysis:
    """Parameter sensitivity scan pipeline.

    Parameters
    ----------
    ff : ForceField
        The molecular system.
    param_name : str
        Parameter to scan.
    values : list of float, optional
        Values to test.
    bond_index : int
        Bond/angle index to perturb.

    Returns
    -------
    ParameterScanAnalysis
        Energy sensitivity data.
    """
    results = scan_parameter(ff, param_name, values, bond_index)
    pvals = [r[0] for r in results]
    totals = [r[1].total_energy for r in results]
    bonds = [r[1].bond_energy for r in results]
    vdws = [r[1].vdw_energy for r in results]

    # Linear sensitivity estimate
    if len(pvals) > 1:
        slope = (totals[-1] - totals[0]) / (pvals[-1] - pvals[0] + 1e-15)
    else:
        slope = 0.0

    explanation = _build_scan_explanation(param_name, slope, pvals, totals)

    return ParameterScanAnalysis(
        molecule_name=ff.name,
        param_name=param_name,
        param_values=pvals,
        total_energies=totals,
        bond_energies=bonds,
        vdw_energies=vdws,
        sensitivity=slope,
        explanation=explanation,
    )


def analyze_perturbation(
    ff: ForceField,
    scales: Optional[List[float]] = None,
    n_samples: int = 5,
    seed: int = 42,
) -> PerturbationAnalysis:
    """Coordinate perturbation robustness pipeline.

    Parameters
    ----------
    ff : ForceField
        The molecular system.
    scales : list of float, optional
        Perturbation magnitudes in Å.
    n_samples : int
        Number of random samples per scale.
    seed : int
        Base random seed.

    Returns
    -------
    PerturbationAnalysis
        Robustness data across perturbation levels.
    """
    if scales is None:
        scales = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    baseline = compute_energy(ff).total_energy
    means: List[float] = []
    stds: List[float] = []

    for sc in scales:
        energies: List[float] = []
        for s in range(n_samples):
            pff = perturb_coordinates(ff, scale=sc, seed=seed + s)
            e = compute_energy(pff).total_energy
            energies.append(e)
        means.append(float(np.mean(energies)))
        stds.append(float(np.std(energies)))

    # Robustness: ratio of baseline to highest perturbation mean
    last_mean = means[-1] if means else baseline
    robustness = baseline / last_mean if abs(last_mean) > 1e-12 else 1.0

    explanation = _build_perturbation_explanation(scales, means, stds, baseline, robustness)

    return PerturbationAnalysis(
        molecule_name=ff.name,
        scales=scales,
        mean_energies=means,
        std_energies=stds,
        baseline_energy=baseline,
        robustness_score=robustness,
        explanation=explanation,
    )


# ═══════════════════════════════════════════════════════════════════════
# Explanation Builders
# ═══════════════════════════════════════════════════════════════════════

def _build_energy_explanation(
    ff: ForceField,
    result: EnergyResult,
    decomp: Dict[str, float],
    dominant: str,
    bonded: float,
    nonbonded: float,
) -> str:
    """Build Markdown explanation for the energy decomposition."""
    lines = [
        f"**{ff.name}** — Energy Decomposition",
        "",
        f"The total potential energy is **{result.total_energy:.4f} kcal/mol**, "
        f"computed from {ff.n_bonds} bonds, {ff.n_angles} angles, "
        f"{ff.n_dihedrals} dihedrals, and all non-bonded pairs within "
        f"the {ff.cutoff:.1f} Å cutoff.",
        "",
        "| Term | Energy (kcal/mol) |",
        "|------|------------------|",
    ]
    for term, val in decomp.items():
        lines.append(f"| {term} | {val:.4f} |")
    lines += [
        "",
        f"The **dominant contribution** is **{dominant}** "
        f"({decomp[dominant]:.4f} kcal/mol).",
        "",
        f"- Bonded terms (bonds + angles + dihedrals): {bonded:.4f} kcal/mol",
        f"- Non-bonded terms (VdW + electrostatics): {nonbonded:.4f} kcal/mol",
        "",
        "Bonded terms reflect covalent-geometry strain (deviations from "
        "equilibrium bond lengths and angles), while non-bonded terms capture "
        "through-space steric repulsion/attraction (Lennard-Jones) and "
        "electrostatic interactions (Coulomb's law).",
    ]
    return "\n".join(lines)


def _build_stress_explanation(
    ff: ForceField,
    stress: StressResult,
    labels: List[str],
    max_label: str,
    min_label: str,
) -> str:
    """Build Markdown explanation for the stress visualizer."""
    lines = [
        f"**{ff.name}** — Bond Stress Analysis",
        "",
        f"Each bond is colored by its **normalized stress** — the ratio of "
        f"its harmonic energy to the maximum bond energy in the system. "
        f"Bonds stretched far beyond their equilibrium length appear **red** "
        f"(high stress), while relaxed bonds appear **blue** (low stress).",
        "",
        f"- Most stressed bond: **{max_label}** (stress = "
        f"{stress.bond_stresses[stress.max_stress_bond]:.4f})" if stress.bond_stresses else "",
        f"- Most relaxed bond: **{min_label}** (stress = "
        f"{stress.bond_stresses[stress.min_stress_bond]:.4f})" if stress.bond_stresses else "",
        f"- Mean stress: **{stress.mean_stress:.4f}**",
        f"- Frustrated residues (above-average stress): "
        f"**{', '.join(str(r) for r in stress.frustrated_regions) if stress.frustrated_regions else 'none'}**",
        "",
        "Frustrated regions often correspond to active sites, binding "
        "pockets, or structurally strained loops where conformational "
        "flexibility is functionally important.",
    ]
    return "\n".join(lines)


def _build_stretch_explanation(
    label: str,
    energy_inc: float,
    max_force: float,
    stretch: BondStretchResult,
) -> str:
    """Build Markdown explanation for the bond-stretch experiment."""
    lines = [
        f"**Break a Bond: {label}**",
        "",
        f"Displacing atom_j along the bond axis from equilibrium to "
        f"{stretch.displacements[-1]:.1f} Å beyond r₀ shows how the "
        f"potential energy surface responds to bond stretching.",
        "",
        f"- **Energy increase:** {energy_inc:.4f} kcal/mol",
        f"- **Approximate max force:** {max_force:.2f} kcal/(mol·Å)",
        f"- **Equilibrium energy:** {stretch.equilibrium_energy:.4f} kcal/mol",
        f"- **Breaking energy:** {stretch.breaking_energy:.4f} kcal/mol",
        "",
        "The harmonic approximation U = ½k(r − r₀)² grows quadratically, so "
        "the energy rises steeply. In reality, bonds have a finite dissociation "
        "energy (Morse potential), but the harmonic model is accurate near "
        "equilibrium and clearly shows the energetic cost of bond stretching.",
    ]
    return "\n".join(lines)


def _build_scan_explanation(
    param_name: str,
    slope: float,
    pvals: List[float],
    totals: List[float],
) -> str:
    """Build Markdown explanation for the parameter scan."""
    lines = [
        f"**Parameter Sensitivity: {param_name}**",
        "",
        f"Scanning `{param_name}` from {pvals[0]:.2f} to {pvals[-1]:.2f} "
        f"reveals how the total energy responds to parameter variation.",
        "",
        f"- **Linear sensitivity (dU/dp):** {slope:.4f} kcal/mol per unit",
        f"- **Energy range:** {min(totals):.4f} to {max(totals):.4f} kcal/mol",
        "",
        "High sensitivity indicates that the energy landscape is strongly "
        "dependent on this parameter, meaning small errors in parameterization "
        "can significantly affect computed energies. Force-field developers "
        "must calibrate these parameters carefully against quantum-mechanical "
        "reference data or experimental observables.",
    ]
    return "\n".join(lines)


def _build_perturbation_explanation(
    scales: List[float],
    means: List[float],
    stds: List[float],
    baseline: float,
    robustness: float,
) -> str:
    """Build Markdown explanation for perturbation analysis."""
    lines = [
        "**Coordinate Perturbation Robustness**",
        "",
        f"Adding Gaussian noise (σ = {scales[0]:.2f} to {scales[-1]:.2f} Å) "
        f"to atomic coordinates tests the ruggedness of the energy surface "
        f"around the current conformation.",
        "",
        f"- **Baseline energy:** {baseline:.4f} kcal/mol",
        f"- **Robustness score:** {robustness:.4f} (ratio of baseline to "
        f"mean energy at σ = {scales[-1]:.2f} Å)",
        "",
        "| σ (Å) | Mean Energy | Std Dev |",
        "|-------|-------------|---------|",
    ]
    for sc, mn, sd in zip(scales, means, stds):
        lines.append(f"| {sc:.3f} | {mn:.4f} | {sd:.4f} |")
    lines += [
        "",
        "A robustness score near 1.0 indicates the structure sits in a "
        "broad, flat energy basin (stable). A score far from 1.0 suggests "
        "the structure is near a steep energy wall or saddle point.",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Summary Builder (CLI)
# ═══════════════════════════════════════════════════════════════════════

def forcefield_summary(analysis: FullEnergyAnalysis) -> str:
    """Formatted CLI summary of a full energy analysis.

    Parameters
    ----------
    analysis : FullEnergyAnalysis
        Energy analysis result.

    Returns
    -------
    str
        Multi-line plain-text summary.
    """
    lines = [
        f"System: {analysis.molecule_name}",
        f"Topology: {analysis.topology}",
        "",
        "Energy Decomposition (kcal/mol):",
        "-" * 40,
    ]
    for term, val in analysis.decomposition.items():
        lines.append(f"  {term:<20s} {val:>12.4f}")
    lines += [
        "-" * 40,
        f"  Bonded total:       {analysis.bonded_total:>12.4f}",
        f"  Non-bonded total:   {analysis.nonbonded_total:>12.4f}",
        f"  Bonded fraction:    {analysis.bonded_fraction:>12.2%}",
        f"  Dominant term:      {analysis.dominant_term} ({analysis.dominant_value:.4f})",
    ]
    return "\n".join(lines)
