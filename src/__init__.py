"""
The Force Field Engine — Molecular Mechanics Energy Calculator.

Modules
-------
forcefield_engine
    Core force field: atoms, bonds, angles, dihedrals, energy calculators,
    stress analysis, bond stretching, preset molecule builders.
analysis
    Higher-level analysis pipelines and summary builders.
visualization
    Plotly (interactive) and Matplotlib (publication) renderers.
"""

from src.forcefield_engine import (
    Atom,
    BondParam,
    AngleParam,
    DihedralParam,
    EnergyResult,
    StressResult,
    BondStretchResult,
    ForceField,
    compute_energy,
    compute_bond_energy,
    compute_angle_energy,
    compute_dihedral_energy,
    compute_nonbonded_energy,
    compute_stress,
    stretch_bond,
    perturb_coordinates,
    scan_parameter,
    build_alanine_dipeptide,
    build_glycine_tripeptide,
    build_alpha_helix_5,
    build_beta_hairpin,
    build_salt_bridge,
    build_disulfide_bond,
    build_molecule,
    energy_decomposition_dict,
    get_bond_labels,
    get_atom_labels,
    topology_summary,
    PRESET_MOLECULES,
    MOLECULE_DESCRIPTIONS,
    ENERGY_TERM_COLORS,
    STRESS_COLORSCALE,
    COULOMB_CONSTANT,
    DEFAULT_DIELECTRIC,
    CUTOFF_DISTANCE,
    LJ_PARAMS,
    PARTIAL_CHARGES,
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

from src.visualization import (
    PlotlyRenderer,
    MatplotlibRenderer,
)

__all__ = [
    # Engine — Data classes
    "Atom",
    "BondParam",
    "AngleParam",
    "DihedralParam",
    "EnergyResult",
    "StressResult",
    "BondStretchResult",
    "ForceField",
    # Engine — Energy calculators
    "compute_energy",
    "compute_bond_energy",
    "compute_angle_energy",
    "compute_dihedral_energy",
    "compute_nonbonded_energy",
    "compute_stress",
    "stretch_bond",
    "perturb_coordinates",
    "scan_parameter",
    # Engine — Builders
    "build_alanine_dipeptide",
    "build_glycine_tripeptide",
    "build_alpha_helix_5",
    "build_beta_hairpin",
    "build_salt_bridge",
    "build_disulfide_bond",
    "build_molecule",
    # Engine — Utilities
    "energy_decomposition_dict",
    "get_bond_labels",
    "get_atom_labels",
    "topology_summary",
    # Engine — Constants
    "PRESET_MOLECULES",
    "MOLECULE_DESCRIPTIONS",
    "ENERGY_TERM_COLORS",
    "STRESS_COLORSCALE",
    "COULOMB_CONSTANT",
    "DEFAULT_DIELECTRIC",
    "CUTOFF_DISTANCE",
    "LJ_PARAMS",
    "PARTIAL_CHARGES",
    # Analysis
    "FullEnergyAnalysis",
    "StressAnalysis",
    "BondStretchAnalysis",
    "ParameterScanAnalysis",
    "PerturbationAnalysis",
    "analyze_energy",
    "analyze_stress",
    "analyze_bond_stretch",
    "analyze_parameter_scan",
    "analyze_perturbation",
    "forcefield_summary",
    # Visualization
    "PlotlyRenderer",
    "MatplotlibRenderer",
]
