"""
The Force Field Engine — Molecular Mechanics Potential Energy Calculator.

Implements the classical molecular mechanics energy function U(r) used in
computational biochemistry to evaluate and minimize protein structures.  The
total potential energy decomposes into bonded (bonds, angles, dihedrals) and
non-bonded (Lennard-Jones 12-6, Coulomb electrostatics) contributions:

    U = U_bonds + U_angles + U_dihedrals + U_VdW + U_electrostatics

Each term uses physically meaningful AMBER-family parameters so that the
energy values are reported in kcal/mol, distances in angstroms, and angles
in radians.

References
----------
Cornell et al. (1995). "A Second Generation Force Field for the Simulation
    of Proteins, Nucleic Acids, and Organic Molecules." JACS 117, 5179-5197.
Weiner et al. (1984). "A New Force Field for Molecular Mechanical
    Simulation of Nucleic Acids and Proteins." JACS 106, 765-784.
Ponder & Case (2003). "Force Fields for Protein Simulations."
    Adv. Protein Chem. 66, 27-85.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# --- Physical constants ---
COULOMB_CONSTANT: float = 332.0636  # kcal·Å/(mol·e²)  Coulomb prefactor
DEFAULT_DIELECTRIC: float = 1.0     # vacuum relative permittivity
CUTOFF_DISTANCE: float = 12.0       # Å  non-bonded cutoff

# --- Default AMBER-like parameters ---
DEFAULT_BOND_K: float = 340.0       # kcal/(mol·Å²)  C-C bond
DEFAULT_BOND_R0: float = 1.526      # Å  equilibrium C-C bond length
DEFAULT_ANGLE_K: float = 63.0       # kcal/(mol·rad²)  C-C-C angle
DEFAULT_ANGLE_THETA0: float = 1.9106  # rad  ~109.5° tetrahedral

DEFAULT_DIHEDRAL_VN: float = 1.4    # kcal/mol  barrier height
DEFAULT_DIHEDRAL_GAMMA: float = 0.0  # rad  phase
DEFAULT_DIHEDRAL_N: int = 3         # periodicity

# --- Lennard-Jones σ and ε for common atom types (AMBER99) ---
LJ_PARAMS: Dict[str, Tuple[float, float]] = {
    # atom_type: (sigma Å, epsilon kcal/mol)
    "C":  (1.9080, 0.0860),  # sp3 carbon
    "CT": (1.9080, 0.1094),  # aliphatic sp3 C
    "CA": (1.9080, 0.0860),  # aromatic C
    "N":  (1.8240, 0.1700),  # sp2 nitrogen
    "O":  (1.6612, 0.2100),  # carbonyl oxygen
    "OH": (1.7210, 0.2104),  # hydroxyl oxygen
    "S":  (2.0000, 0.2500),  # sulfur
    "H":  (0.6000, 0.0157),  # hydrogen
    "HA": (1.4590, 0.0150),  # aromatic H
    "HC": (1.4870, 0.0157),  # aliphatic H
    "HN": (0.6000, 0.0157),  # amide H
}

# --- Partial charges for backbone & common side-chain (AMBER) ---
PARTIAL_CHARGES: Dict[str, float] = {
    # atom_name: charge (elementary charge units)
    "N":   -0.4157,
    "H":    0.2719,
    "CA":   0.0337,
    "HA":   0.0823,
    "C":    0.5973,
    "O":   -0.5679,
    "CB":  -0.1825,
    "HB":   0.0603,
    "CG":  -0.0103,
    "HG":   0.0317,
    "CD":   0.0305,
    "HD":   0.0366,
    "CE":  -0.0187,
    "HE":   0.0360,
    "NZ":  -0.3854,
    "HZ":   0.3400,
    "OG":  -0.6546,
    "OD":  -0.5819,
    "ND":  -0.1520,
    "NE":  -0.2637,
    "SD":  -0.2737,
    "SG":  -0.3119,
}

# --- Preset molecule names ---
PRESET_MOLECULES: List[str] = [
    "alanine_dipeptide",
    "glycine_tripeptide",
    "alpha_helix_5",
    "beta_hairpin",
    "salt_bridge",
    "disulfide_bond",
]

MOLECULE_DESCRIPTIONS: Dict[str, str] = {
    "alanine_dipeptide": "Ace-Ala-Nme — minimal model for backbone conformations",
    "glycine_tripeptide": "Ace-Gly-Gly-Gly-Nme — flexible backbone chain",
    "alpha_helix_5": "Five-residue poly-alanine α-helix with hydrogen bonds",
    "beta_hairpin": "Four-residue β-hairpin with a type-I turn",
    "salt_bridge": "Lys-Asp side-chain ion pair at ~2.8 Å",
    "disulfide_bond": "Two Cys residues forming a covalent S─S bridge",
}

# --- Color maps for stress visualization ---
STRESS_COLORSCALE: List[List] = [
    [0.0, "#2166ac"],   # deep blue  — relaxed
    [0.25, "#67a9cf"],
    [0.5, "#f7f7f7"],   # white      — neutral
    [0.75, "#ef8a62"],
    [1.0, "#b2182b"],   # deep red   — stressed
]

ENERGY_TERM_COLORS: Dict[str, str] = {
    "bonds": "#1f77b4",
    "angles": "#ff7f0e",
    "dihedrals": "#2ca02c",
    "vdw": "#d62728",
    "electrostatics": "#9467bd",
    "total": "#17becf",
}

ATOM_TYPE_COLORS: Dict[str, str] = {
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "S": "#FFFF30",
    "H": "#FFFFFF",
}


# ═══════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Atom:
    """A single atom in the molecular system.

    Attributes
    ----------
    index : int
        Zero-based atom index.
    name : str
        PDB atom name (e.g. ``"CA"``).
    element : str
        Chemical element symbol.
    atom_type : str
        Force-field atom type for LJ parameters.
    residue : str
        Three-letter residue name.
    residue_id : int
        Residue sequence number.
    x : float
        Cartesian x-coordinate in angstroms.
    y : float
        Cartesian y-coordinate in angstroms.
    z : float
        Cartesian z-coordinate in angstroms.
    charge : float
        Partial charge in elementary charge units.
    mass : float
        Atomic mass in daltons.
    """

    index: int
    name: str
    element: str
    atom_type: str = "C"
    residue: str = "ALA"
    residue_id: int = 1
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    charge: float = 0.0
    mass: float = 12.011

    @property
    def position(self) -> np.ndarray:
        """Return (x, y, z) as a NumPy array."""
        return np.array([self.x, self.y, self.z])


@dataclass
class BondParam:
    """Harmonic bond potential parameters.

    Attributes
    ----------
    atom_i : int
        Index of first atom.
    atom_j : int
        Index of second atom.
    k : float
        Force constant in kcal/(mol·Å²).
    r0 : float
        Equilibrium bond length in angstroms.
    """

    atom_i: int
    atom_j: int
    k: float = DEFAULT_BOND_K
    r0: float = DEFAULT_BOND_R0


@dataclass
class AngleParam:
    """Harmonic angle potential parameters.

    Attributes
    ----------
    atom_i : int
        Index of first atom.
    atom_j : int
        Index of vertex atom.
    atom_k : int
        Index of third atom.
    k : float
        Force constant in kcal/(mol·rad²).
    theta0 : float
        Equilibrium angle in radians.
    """

    atom_i: int
    atom_j: int
    atom_k: int
    k: float = DEFAULT_ANGLE_K
    theta0: float = DEFAULT_ANGLE_THETA0


@dataclass
class DihedralParam:
    """Periodic torsional (dihedral) potential parameters.

    Attributes
    ----------
    atom_i : int
        Index of first atom.
    atom_j : int
        Index of second atom.
    atom_k : int
        Index of third atom.
    atom_l : int
        Index of fourth atom.
    Vn : float
        Barrier height in kcal/mol.
    gamma : float
        Phase offset in radians.
    n : int
        Periodicity (number of minima per full rotation).
    """

    atom_i: int
    atom_j: int
    atom_k: int
    atom_l: int
    Vn: float = DEFAULT_DIHEDRAL_VN
    gamma: float = DEFAULT_DIHEDRAL_GAMMA
    n: int = DEFAULT_DIHEDRAL_N


@dataclass
class EnergyResult:
    """Decomposed potential energy evaluation.

    Attributes
    ----------
    bond_energy : float
        Sum of harmonic bond energies (kcal/mol).
    angle_energy : float
        Sum of harmonic angle energies (kcal/mol).
    dihedral_energy : float
        Sum of periodic dihedral energies (kcal/mol).
    vdw_energy : float
        Sum of Lennard-Jones 12-6 energies (kcal/mol).
    electrostatic_energy : float
        Sum of Coulombic energies (kcal/mol).
    total_energy : float
        Sum of all energy terms (kcal/mol).
    bond_energies : List[float]
        Per-bond energy contributions.
    angle_energies : List[float]
        Per-angle energy contributions.
    dihedral_energies : List[float]
        Per-dihedral energy contributions.
    vdw_pairs : List[Tuple[int, int, float]]
        Non-bonded VdW pair energies (i, j, energy).
    electrostatic_pairs : List[Tuple[int, int, float]]
        Non-bonded electrostatic pair energies (i, j, energy).
    status : str
        Computation status (``"success"`` or error description).
    """

    bond_energy: float = 0.0
    angle_energy: float = 0.0
    dihedral_energy: float = 0.0
    vdw_energy: float = 0.0
    electrostatic_energy: float = 0.0
    total_energy: float = 0.0
    bond_energies: List[float] = field(default_factory=list)
    angle_energies: List[float] = field(default_factory=list)
    dihedral_energies: List[float] = field(default_factory=list)
    vdw_pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    electrostatic_pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    status: str = "success"


@dataclass
class StressResult:
    """Per-bond stress analysis for the stress visualizer.

    Attributes
    ----------
    bond_stresses : List[float]
        Normalized stress per bond (0 = relaxed, 1 = max stress).
    bond_energies : List[float]
        Raw energy per bond (kcal/mol).
    bond_deviations : List[float]
        Deviation |r − r₀| per bond (Å).
    max_stress_bond : int
        Index of the most stressed bond.
    min_stress_bond : int
        Index of the most relaxed bond.
    mean_stress : float
        Mean normalized stress across all bonds.
    frustrated_regions : List[int]
        Residue IDs with above-average stress.
    """

    bond_stresses: List[float] = field(default_factory=list)
    bond_energies: List[float] = field(default_factory=list)
    bond_deviations: List[float] = field(default_factory=list)
    max_stress_bond: int = 0
    min_stress_bond: int = 0
    mean_stress: float = 0.0
    frustrated_regions: List[int] = field(default_factory=list)


@dataclass
class BondStretchResult:
    """Result of the interactive "Break a Bond" experiment.

    Attributes
    ----------
    displacements : List[float]
        Displacement values tested (Å from equilibrium).
    energies : List[float]
        Total energy at each displacement (kcal/mol).
    bond_energies_curve : List[float]
        Bond energy component at each displacement.
    vdw_energies_curve : List[float]
        VdW energy component at each displacement.
    elec_energies_curve : List[float]
        Electrostatic energy component at each displacement.
    equilibrium_energy : float
        Energy at equilibrium (kcal/mol).
    breaking_energy : float
        Energy at maximum displacement (kcal/mol).
    bond_index : int
        Index of the stretched bond.
    """

    displacements: List[float] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    bond_energies_curve: List[float] = field(default_factory=list)
    vdw_energies_curve: List[float] = field(default_factory=list)
    elec_energies_curve: List[float] = field(default_factory=list)
    equilibrium_energy: float = 0.0
    breaking_energy: float = 0.0
    bond_index: int = 0


# ═══════════════════════════════════════════════════════════════════════
# Core Class — ForceField
# ═══════════════════════════════════════════════════════════════════════

class ForceField:
    """Molecular mechanics force field for protein energy evaluation.

    Holds a molecular topology (atoms, bonds, angles, dihedrals) together
    with the non-bonded parameter tables and computes U(r) decomposed into
    five standard terms.

    Attributes
    ----------
    name : str
        Human-readable name for this system.
    atoms : list of Atom
        All atoms in the system.
    bonds : list of BondParam
        Bonded harmonic terms.
    angles : list of AngleParam
        Angle harmonic terms.
    dihedrals : list of DihedralParam
        Periodic torsion terms.
    dielectric : float
        Relative permittivity for Coulomb calculation.
    cutoff : float
        Non-bonded interaction cutoff in angstroms.
    """

    def __init__(
        self,
        name: str = "unnamed",
        dielectric: float = DEFAULT_DIELECTRIC,
        cutoff: float = CUTOFF_DISTANCE,
    ) -> None:
        self.name = name
        self.atoms: List[Atom] = []
        self.bonds: List[BondParam] = []
        self.angles: List[AngleParam] = []
        self.dihedrals: List[DihedralParam] = []
        self.dielectric = dielectric
        self.cutoff = cutoff
        self._exclusion_set: Set[Tuple[int, int]] = set()

    # ── Topology builders ────────────────────────────────────────

    def add_atom(self, atom: Atom) -> None:
        """Append an *atom* to the system."""
        self.atoms.append(atom)

    def add_bond(self, bond: BondParam) -> None:
        """Register a harmonic *bond* and mark the pair as excluded from non-bonded."""
        self.bonds.append(bond)
        i, j = bond.atom_i, bond.atom_j
        self._exclusion_set.add((min(i, j), max(i, j)))

    def add_angle(self, angle: AngleParam) -> None:
        """Register a harmonic *angle* and exclude the 1-3 pair."""
        self.angles.append(angle)
        i, k = angle.atom_i, angle.atom_k
        self._exclusion_set.add((min(i, k), max(i, k)))

    def add_dihedral(self, dihedral: DihedralParam) -> None:
        """Register a periodic *dihedral*."""
        self.dihedrals.append(dihedral)

    def build_exclusions(self) -> None:
        """Rebuild exclusion set from current bonds and angles."""
        self._exclusion_set.clear()
        for b in self.bonds:
            i, j = b.atom_i, b.atom_j
            self._exclusion_set.add((min(i, j), max(i, j)))
        for a in self.angles:
            i, k = a.atom_i, a.atom_k
            self._exclusion_set.add((min(i, k), max(i, k)))

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the system."""
        return len(self.atoms)

    @property
    def n_bonds(self) -> int:
        """Number of bonds."""
        return len(self.bonds)

    @property
    def n_angles(self) -> int:
        """Number of angle terms."""
        return len(self.angles)

    @property
    def n_dihedrals(self) -> int:
        """Number of dihedral terms."""
        return len(self.dihedrals)

    @property
    def positions(self) -> np.ndarray:
        """Return Nx3 coordinate matrix."""
        return np.array([[a.x, a.y, a.z] for a in self.atoms])

    def set_positions(self, coords: np.ndarray) -> None:
        """Update atom coordinates from an Nx3 array."""
        for i, atom in enumerate(self.atoms):
            atom.x, atom.y, atom.z = float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])

    def copy(self) -> "ForceField":
        """Return a deep copy of the force field."""
        return copy.deepcopy(self)


# ═══════════════════════════════════════════════════════════════════════
# Energy Calculators
# ═══════════════════════════════════════════════════════════════════════

def _distance(a: Atom, b: Atom) -> float:
    """Euclidean distance between two atoms in angstroms."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _angle(a: Atom, b: Atom, c: Atom) -> float:
    """Angle a-b-c in radians via the dot-product formula."""
    v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-15)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def _dihedral(a: Atom, b: Atom, c: Atom, d: Atom) -> float:
    """Dihedral angle a-b-c-d in radians via the atan2 formula."""
    b1 = np.array([b.x - a.x, b.y - a.y, b.z - a.z])
    b2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    b3 = np.array([d.x - c.x, d.y - c.y, d.z - c.z])
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-15))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return float(np.arctan2(y, x))


def compute_bond_energy(ff: ForceField) -> Tuple[float, List[float]]:
    """Compute harmonic bond energies: U = ½k(r − r₀)².

    Parameters
    ----------
    ff : ForceField
        The molecular system.

    Returns
    -------
    Tuple[float, List[float]]
        Total bond energy and per-bond energies in kcal/mol.
    """
    per_bond: List[float] = []
    total = 0.0
    for bond in ff.bonds:
        r = _distance(ff.atoms[bond.atom_i], ff.atoms[bond.atom_j])
        e = 0.5 * bond.k * (r - bond.r0) ** 2
        per_bond.append(e)
        total += e
    return total, per_bond


def compute_angle_energy(ff: ForceField) -> Tuple[float, List[float]]:
    """Compute harmonic angle energies: U = ½k(θ − θ₀)².

    Parameters
    ----------
    ff : ForceField
        The molecular system.

    Returns
    -------
    Tuple[float, List[float]]
        Total angle energy and per-angle energies in kcal/mol.
    """
    per_angle: List[float] = []
    total = 0.0
    for ang in ff.angles:
        theta = _angle(ff.atoms[ang.atom_i], ff.atoms[ang.atom_j], ff.atoms[ang.atom_k])
        e = 0.5 * ang.k * (theta - ang.theta0) ** 2
        per_angle.append(e)
        total += e
    return total, per_angle


def compute_dihedral_energy(ff: ForceField) -> Tuple[float, List[float]]:
    """Compute periodic dihedral energies: U = (Vn/2)[1 + cos(nφ − γ)].

    Parameters
    ----------
    ff : ForceField
        The molecular system.

    Returns
    -------
    Tuple[float, List[float]]
        Total dihedral energy and per-dihedral energies in kcal/mol.
    """
    per_dih: List[float] = []
    total = 0.0
    for dih in ff.dihedrals:
        phi = _dihedral(
            ff.atoms[dih.atom_i], ff.atoms[dih.atom_j],
            ff.atoms[dih.atom_k], ff.atoms[dih.atom_l],
        )
        e = (dih.Vn / 2.0) * (1.0 + math.cos(dih.n * phi - dih.gamma))
        per_dih.append(e)
        total += e
    return total, per_dih


def _lj_combine(type_i: str, type_j: str) -> Tuple[float, float]:
    """Lorentz–Berthelot combining rules for LJ parameters.

    Returns
    -------
    Tuple[float, float]
        Combined (sigma, epsilon) in (Å, kcal/mol).
    """
    si, ei = LJ_PARAMS.get(type_i, (1.9080, 0.0860))
    sj, ej = LJ_PARAMS.get(type_j, (1.9080, 0.0860))
    sigma = (si + sj) / 2.0
    epsilon = math.sqrt(ei * ej)
    return sigma, epsilon


def compute_nonbonded_energy(
    ff: ForceField,
) -> Tuple[float, float, List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    """Compute Lennard-Jones 12-6 and Coulomb non-bonded energies.

    LJ:  U = 4ε[(σ/r)¹² − (σ/r)⁶]
    Coulomb:  U = (332.0636 · q_i · q_j) / (ε_r · r)

    Pairs where both atoms are within 3 bonds (1-2 and 1-3 exclusions)
    are skipped.  A distance cutoff is applied.

    Parameters
    ----------
    ff : ForceField
        The molecular system.

    Returns
    -------
    Tuple[float, float, List, List]
        (vdw_total, elec_total, vdw_pairs, elec_pairs).
    """
    vdw_total = 0.0
    elec_total = 0.0
    vdw_pairs: List[Tuple[int, int, float]] = []
    elec_pairs: List[Tuple[int, int, float]] = []
    n = len(ff.atoms)
    for i in range(n):
        for j in range(i + 1, n):
            pair = (i, j)
            if pair in ff._exclusion_set:
                continue
            ai, aj = ff.atoms[i], ff.atoms[j]
            r = _distance(ai, aj)
            if r < 0.01:
                r = 0.01  # avoid singularity
            if r > ff.cutoff:
                continue
            # Lennard-Jones
            sigma, epsilon = _lj_combine(ai.atom_type, aj.atom_type)
            sr6 = (sigma / r) ** 6
            e_lj = 4.0 * epsilon * (sr6 ** 2 - sr6)
            vdw_total += e_lj
            vdw_pairs.append((i, j, e_lj))
            # Coulomb
            e_coul = COULOMB_CONSTANT * ai.charge * aj.charge / (ff.dielectric * r)
            elec_total += e_coul
            elec_pairs.append((i, j, e_coul))
    return vdw_total, elec_total, vdw_pairs, elec_pairs


def compute_energy(ff: ForceField) -> EnergyResult:
    """Compute the full potential energy U(r) decomposed into five terms.

    Parameters
    ----------
    ff : ForceField
        The molecular system with coordinates and parameters.

    Returns
    -------
    EnergyResult
        Decomposed energy evaluation.
    """
    bond_e, bond_list = compute_bond_energy(ff)
    angle_e, angle_list = compute_angle_energy(ff)
    dih_e, dih_list = compute_dihedral_energy(ff)
    vdw_e, elec_e, vdw_pairs, elec_pairs = compute_nonbonded_energy(ff)
    total = bond_e + angle_e + dih_e + vdw_e + elec_e
    return EnergyResult(
        bond_energy=bond_e,
        angle_energy=angle_e,
        dihedral_energy=dih_e,
        vdw_energy=vdw_e,
        electrostatic_energy=elec_e,
        total_energy=total,
        bond_energies=bond_list,
        angle_energies=angle_list,
        dihedral_energies=dih_list,
        vdw_pairs=vdw_pairs,
        electrostatic_pairs=elec_pairs,
        status="success",
    )


# ═══════════════════════════════════════════════════════════════════════
# Stress Analyzer
# ═══════════════════════════════════════════════════════════════════════

def compute_stress(ff: ForceField) -> StressResult:
    """Per-bond stress analysis: normalized deviation from equilibrium.

    Bonds stretched far beyond r₀ receive high stress (→ red); bonds
    close to equilibrium have low stress (→ blue).

    Parameters
    ----------
    ff : ForceField
        The molecular system.

    Returns
    -------
    StressResult
        Stress data for visualization.
    """
    _, per_bond = compute_bond_energy(ff)
    deviations: List[float] = []
    for bond in ff.bonds:
        r = _distance(ff.atoms[bond.atom_i], ff.atoms[bond.atom_j])
        deviations.append(abs(r - bond.r0))

    if not per_bond:
        return StressResult()

    max_e = max(per_bond) if max(per_bond) > 1e-12 else 1.0
    stresses = [e / max_e for e in per_bond]
    mean_s = sum(stresses) / len(stresses)

    # Identify frustrated residues (above-average stress)
    frustrated: Set[int] = set()
    for idx, s in enumerate(stresses):
        if s > mean_s:
            b = ff.bonds[idx]
            frustrated.add(ff.atoms[b.atom_i].residue_id)
            frustrated.add(ff.atoms[b.atom_j].residue_id)

    return StressResult(
        bond_stresses=stresses,
        bond_energies=per_bond,
        bond_deviations=deviations,
        max_stress_bond=int(np.argmax(stresses)),
        min_stress_bond=int(np.argmin(stresses)),
        mean_stress=mean_s,
        frustrated_regions=sorted(frustrated),
    )


# ═══════════════════════════════════════════════════════════════════════
# Bond Stretching — "Break a Bond" Experiment
# ═══════════════════════════════════════════════════════════════════════

def stretch_bond(
    ff: ForceField,
    bond_index: int = 0,
    max_displacement: float = 3.0,
    n_steps: int = 60,
) -> BondStretchResult:
    """Stretch a single bond and record the energy response.

    Displaces *atom_j* of the chosen bond along the bond axis from
    equilibrium to *max_displacement* angstroms beyond r₀.

    Parameters
    ----------
    ff : ForceField
        The molecular system (will be deep-copied).
    bond_index : int
        Index of the bond to stretch.
    max_displacement : float
        Maximum displacement in angstroms beyond r₀.
    n_steps : int
        Number of displacement steps.

    Returns
    -------
    BondStretchResult
        Energy vs. displacement data.

    Raises
    ------
    ValueError
        If *bond_index* is out of range.
    """
    if bond_index < 0 or bond_index >= len(ff.bonds):
        raise ValueError(
            f"Bond index {bond_index} out of range. "
            f"System has {len(ff.bonds)} bonds (0–{len(ff.bonds) - 1})."
        )

    bond = ff.bonds[bond_index]
    ai, aj = bond.atom_i, bond.atom_j
    # Direction vector from atom_i to atom_j
    dx = ff.atoms[aj].x - ff.atoms[ai].x
    dy = ff.atoms[aj].y - ff.atoms[ai].y
    dz = ff.atoms[aj].z - ff.atoms[ai].z
    length = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) + 1e-15
    ux, uy, uz = dx / length, dy / length, dz / length

    displacements: List[float] = []
    energies: List[float] = []
    bond_es: List[float] = []
    vdw_es: List[float] = []
    elec_es: List[float] = []

    for step in range(n_steps + 1):
        d = (max_displacement / n_steps) * step
        test_ff = ff.copy()
        # Displace atom_j along bond axis
        test_ff.atoms[aj].x = ff.atoms[aj].x + ux * d
        test_ff.atoms[aj].y = ff.atoms[aj].y + uy * d
        test_ff.atoms[aj].z = ff.atoms[aj].z + uz * d
        result = compute_energy(test_ff)
        displacements.append(d)
        energies.append(result.total_energy)
        bond_es.append(result.bond_energy)
        vdw_es.append(result.vdw_energy)
        elec_es.append(result.electrostatic_energy)

    return BondStretchResult(
        displacements=displacements,
        energies=energies,
        bond_energies_curve=bond_es,
        vdw_energies_curve=vdw_es,
        elec_energies_curve=elec_es,
        equilibrium_energy=energies[0] if energies else 0.0,
        breaking_energy=energies[-1] if energies else 0.0,
        bond_index=bond_index,
    )


# ═══════════════════════════════════════════════════════════════════════
# Parameter Perturbation
# ═══════════════════════════════════════════════════════════════════════

def perturb_coordinates(
    ff: ForceField,
    scale: float = 0.1,
    seed: Optional[int] = None,
) -> ForceField:
    """Return a copy of *ff* with Gaussian noise added to coordinates.

    Parameters
    ----------
    ff : ForceField
        Original system.
    scale : float
        Standard deviation of perturbation in angstroms.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ForceField
        Perturbed copy.
    """
    rng = np.random.default_rng(seed)
    perturbed = ff.copy()
    for atom in perturbed.atoms:
        atom.x += float(rng.normal(0.0, scale))
        atom.y += float(rng.normal(0.0, scale))
        atom.z += float(rng.normal(0.0, scale))
    return perturbed


def scan_parameter(
    ff: ForceField,
    param_name: str = "bond_k",
    values: Optional[List[float]] = None,
    bond_index: int = 0,
) -> List[Tuple[float, EnergyResult]]:
    """Scan one force-field parameter and record energy at each value.

    Parameters
    ----------
    ff : ForceField
        Molecular system.
    param_name : str
        One of ``"bond_k"``, ``"bond_r0"``, ``"angle_k"``, ``"dielectric"``.
    values : list of float, optional
        Parameter values to scan.  Defaults to a sensible range.
    bond_index : int
        Bond or angle index to modify (for bond/angle parameters).

    Returns
    -------
    List[Tuple[float, EnergyResult]]
        (parameter_value, energy_result) pairs.

    Raises
    ------
    ValueError
        If *param_name* is unrecognized.
    """
    valid_params = {"bond_k", "bond_r0", "angle_k", "dielectric"}
    if param_name not in valid_params:
        raise ValueError(
            f"Unknown parameter '{param_name}'. Choose from: {', '.join(sorted(valid_params))}"
        )

    if values is None:
        if param_name == "bond_k":
            values = [v for v in np.linspace(100, 600, 20)]
        elif param_name == "bond_r0":
            values = [v for v in np.linspace(1.0, 2.5, 20)]
        elif param_name == "angle_k":
            values = [v for v in np.linspace(20, 120, 20)]
        else:  # dielectric
            values = [v for v in np.linspace(1.0, 80.0, 20)]

    results: List[Tuple[float, EnergyResult]] = []
    for val in values:
        test_ff = ff.copy()
        if param_name == "bond_k" and test_ff.bonds:
            idx = min(bond_index, len(test_ff.bonds) - 1)
            test_ff.bonds[idx].k = val
        elif param_name == "bond_r0" and test_ff.bonds:
            idx = min(bond_index, len(test_ff.bonds) - 1)
            test_ff.bonds[idx].r0 = val
        elif param_name == "angle_k" and test_ff.angles:
            idx = min(bond_index, len(test_ff.angles) - 1)
            test_ff.angles[idx].k = val
        elif param_name == "dielectric":
            test_ff.dielectric = val
        energy = compute_energy(test_ff)
        results.append((val, energy))
    return results


# ═══════════════════════════════════════════════════════════════════════
# Preset Builders
# ═══════════════════════════════════════════════════════════════════════

def build_alanine_dipeptide() -> ForceField:
    """Build Ace-Ala-Nme (alanine dipeptide) with AMBER parameters.

    This 22-atom system is the standard minimal model for backbone
    φ/ψ torsional landscapes.

    Returns
    -------
    ForceField
        Parameterized alanine dipeptide.
    """
    ff = ForceField(name="Alanine Dipeptide (Ace-Ala-Nme)")

    # ── Atom definitions (AMBER atom types + coordinates) ──
    atoms_data = [
        # idx, name, elem, type, res, resid,   x,      y,      z,    charge, mass
        (0,  "CH3", "C", "CT", "ACE", 1,  -2.000,  0.000,  0.000, -0.3662, 12.011),
        (1,  "HH31","H", "HC", "ACE", 1,  -2.400,  1.020,  0.000,  0.1123,  1.008),
        (2,  "HH32","H", "HC", "ACE", 1,  -2.400, -0.510,  0.884,  0.1123,  1.008),
        (3,  "HH33","H", "HC", "ACE", 1,  -2.400, -0.510, -0.884,  0.1123,  1.008),
        (4,  "C",   "C", "C",  "ACE", 1,  -0.500,  0.000,  0.000,  0.5972, 12.011),
        (5,  "O",   "O", "O",  "ACE", 1,   0.100,  1.050,  0.000, -0.5679, 15.999),
        (6,  "N",   "N", "N",  "ALA", 2,   0.200, -1.100,  0.000, -0.4157, 14.007),
        (7,  "H",   "H", "HN", "ALA", 2,  -0.350, -1.970,  0.000,  0.2719,  1.008),
        (8,  "CA",  "C", "CT", "ALA", 2,   1.650, -1.100,  0.000,  0.0337, 12.011),
        (9,  "HA",  "H", "HA", "ALA", 2,   2.050, -0.580,  0.880,  0.0823,  1.008),
        (10, "CB",  "C", "CT", "ALA", 2,   2.100, -2.560,  0.000, -0.1825, 12.011),
        (11, "HB1", "H", "HC", "ALA", 2,   1.700, -3.080,  0.880,  0.0603,  1.008),
        (12, "HB2", "H", "HC", "ALA", 2,   3.190, -2.560,  0.000,  0.0603,  1.008),
        (13, "HB3", "H", "HC", "ALA", 2,   1.700, -3.080, -0.880,  0.0603,  1.008),
        (14, "C",   "C", "C",  "ALA", 2,   2.200, -0.350, -1.220,  0.5973, 12.011),
        (15, "O",   "O", "O",  "ALA", 2,   1.630,  0.600, -1.750, -0.5679, 15.999),
        (16, "N",   "N", "N",  "NME", 3,   3.300, -0.750, -1.850, -0.4157, 14.007),
        (17, "H",   "H", "HN", "NME", 3,   3.750, -1.550, -1.400,  0.2719,  1.008),
        (18, "CH3", "C", "CT", "NME", 3,   3.850, -0.100, -3.050, -0.1490, 12.011),
        (19, "HH31","H", "HC", "NME", 3,   3.420,  0.880, -3.250,  0.0976,  1.008),
        (20, "HH32","H", "HC", "NME", 3,   4.940, -0.020, -2.970,  0.0976,  1.008),
        (21, "HH33","H", "HC", "NME", 3,   3.600, -0.720, -3.910,  0.0976,  1.008),
    ]

    for data in atoms_data:
        ff.add_atom(Atom(
            index=data[0], name=data[1], element=data[2],
            atom_type=data[3], residue=data[4], residue_id=data[5],
            x=data[6], y=data[7], z=data[8],
            charge=data[9], mass=data[10],
        ))

    # ── Bonds (AMBER harmonic) ──
    bond_defs = [
        # (i, j, k kcal/(mol·Å²), r0 Å)
        (0, 1, 340.0, 1.090),  (0, 2, 340.0, 1.090),  (0, 3, 340.0, 1.090),
        (0, 4, 317.0, 1.522),  (4, 5, 570.0, 1.229),  (4, 6, 490.0, 1.335),
        (6, 7, 434.0, 1.010),  (6, 8, 337.0, 1.449),  (8, 9, 340.0, 1.090),
        (8, 10, 310.0, 1.526), (8, 14, 317.0, 1.522), (10, 11, 340.0, 1.090),
        (10, 12, 340.0, 1.090),(10, 13, 340.0, 1.090), (14, 15, 570.0, 1.229),
        (14, 16, 490.0, 1.335),(16, 17, 434.0, 1.010), (16, 18, 337.0, 1.449),
        (18, 19, 340.0, 1.090),(18, 20, 340.0, 1.090), (18, 21, 340.0, 1.090),
    ]
    for i, j, k, r0 in bond_defs:
        ff.add_bond(BondParam(atom_i=i, atom_j=j, k=k, r0=r0))

    # ── Angles ──
    angle_defs = [
        (1, 0, 2, 35.0, 1.9106),  (1, 0, 3, 35.0, 1.9106),
        (2, 0, 3, 35.0, 1.9106),  (1, 0, 4, 50.0, 1.9106),
        (2, 0, 4, 50.0, 1.9106),  (3, 0, 4, 50.0, 1.9106),
        (0, 4, 5, 80.0, 2.1012),  (0, 4, 6, 70.0, 2.0350),
        (5, 4, 6, 80.0, 2.1450),  (4, 6, 7, 50.0, 2.0857),
        (4, 6, 8, 50.0, 2.1240),  (7, 6, 8, 50.0, 2.0420),
        (6, 8, 9, 50.0, 1.9106),  (6, 8, 10, 80.0, 1.9340),
        (6, 8, 14, 63.0, 1.9106), (9, 8, 10, 50.0, 1.9106),
        (9, 8, 14, 50.0, 1.9106), (10, 8, 14, 63.0, 1.9106),
        (8, 10, 11, 50.0, 1.9106),(8, 10, 12, 50.0, 1.9106),
        (8, 10, 13, 50.0, 1.9106),(11, 10, 12, 35.0, 1.9106),
        (11, 10, 13, 35.0, 1.9106),(12, 10, 13, 35.0, 1.9106),
        (8, 14, 15, 80.0, 2.1012),(8, 14, 16, 70.0, 2.0350),
        (15, 14, 16, 80.0, 2.1450),(14, 16, 17, 50.0, 2.0857),
        (14, 16, 18, 50.0, 2.1240),(17, 16, 18, 50.0, 2.0420),
        (16, 18, 19, 50.0, 1.9106),(16, 18, 20, 50.0, 1.9106),
        (16, 18, 21, 50.0, 1.9106),(19, 18, 20, 35.0, 1.9106),
        (19, 18, 21, 35.0, 1.9106),(20, 18, 21, 35.0, 1.9106),
    ]
    for i, j, k_idx, k_val, t0 in angle_defs:
        ff.add_angle(AngleParam(atom_i=i, atom_j=j, atom_k=k_idx, k=k_val, theta0=t0))

    # ── Dihedrals (backbone φ/ψ + methyl rotors) ──
    dih_defs = [
        (0, 4, 6, 8,  2.50, math.pi, 2),   # ω (Ace C-N)
        (5, 4, 6, 8,  2.50, math.pi, 2),   # ω (O=C-N)
        (4, 6, 8, 14, 0.50, math.pi, 2),   # φ
        (4, 6, 8, 10, 0.15, 0.0, 3),       # φ-CB
        (6, 8, 14, 16, 0.20, math.pi, 2),  # ψ
        (6, 8, 14, 15, 0.07, 0.0, 2),      # ψ-O
        (8, 14, 16, 18, 2.50, math.pi, 2), # ω (Ala C-N)
        (15, 14, 16,18, 2.50, math.pi, 2), # ω (O=C-N)
        (6, 8, 10, 11, 0.16, 0.0, 3),      # χ methyl
        (6, 8, 10, 12, 0.16, 0.0, 3),
        (6, 8, 10, 13, 0.16, 0.0, 3),
        (14, 16, 18, 19, 0.16, 0.0, 3),    # Nme methyl
        (14, 16, 18, 20, 0.16, 0.0, 3),
        (14, 16, 18, 21, 0.16, 0.0, 3),
    ]
    for i, j, k, l, vn, gamma, n in dih_defs:
        ff.add_dihedral(DihedralParam(atom_i=i, atom_j=j, atom_k=k, atom_l=l,
                                      Vn=vn, gamma=gamma, n=n))

    return ff


def build_glycine_tripeptide() -> ForceField:
    """Build Ace-Gly-Gly-Gly-Nme with AMBER parameters.

    Glycine lacks a side chain, making this chain highly flexible —
    useful for testing dihedral scan landscapes.

    Returns
    -------
    ForceField
        Parameterized glycine tripeptide.
    """
    ff = ForceField(name="Glycine Tripeptide (Ace-Gly₃-Nme)")

    # Extended backbone, ~3.8 Å per residue along x
    # Ace cap
    atoms = [
        (0,  "CH3","C","CT","ACE",1, -3.80, 0.00, 0.00, -0.3662, 12.011),
        (1,  "HH31","H","HC","ACE",1, -4.20, 1.02, 0.00, 0.1123, 1.008),
        (2,  "HH32","H","HC","ACE",1, -4.20,-0.51, 0.88, 0.1123, 1.008),
        (3,  "HH33","H","HC","ACE",1, -4.20,-0.51,-0.88, 0.1123, 1.008),
        (4,  "C","C","C","ACE",1, -2.30, 0.00, 0.00, 0.5972, 12.011),
        (5,  "O","O","O","ACE",1, -1.70, 1.10, 0.00, -0.5679, 15.999),
        # Gly 1
        (6,  "N","N","N","GLY",2, -1.60,-1.10, 0.00, -0.4157, 14.007),
        (7,  "H","H","HN","GLY",2, -2.10,-1.97, 0.00,  0.2719, 1.008),
        (8,  "CA","C","CT","GLY",2, -0.15,-1.10, 0.00, -0.0252, 12.011),
        (9,  "HA2","H","HC","GLY",2,  0.25,-0.58, 0.88,  0.0698, 1.008),
        (10, "HA3","H","HC","GLY",2,  0.25,-0.58,-0.88,  0.0698, 1.008),
        (11, "C","C","C","GLY",2,  0.40,-2.50, 0.00,  0.5973, 12.011),
        (12, "O","O","O","GLY",2, -0.20,-3.55, 0.00, -0.5679, 15.999),
        # Gly 2
        (13, "N","N","N","GLY",3,  1.70,-2.50, 0.00, -0.4157, 14.007),
        (14, "H","H","HN","GLY",3,  2.15,-1.60, 0.00,  0.2719, 1.008),
        (15, "CA","C","CT","GLY",3,  2.45,-3.70, 0.00, -0.0252, 12.011),
        (16, "HA2","H","HC","GLY",3,  2.00,-4.25, 0.88,  0.0698, 1.008),
        (17, "HA3","H","HC","GLY",3,  2.00,-4.25,-0.88,  0.0698, 1.008),
        (18, "C","C","C","GLY",3,  3.95,-3.55, 0.00,  0.5973, 12.011),
        (19, "O","O","O","GLY",3,  4.55,-2.50, 0.00, -0.5679, 15.999),
        # Gly 3
        (20, "N","N","N","GLY",4,  4.55,-4.70, 0.00, -0.4157, 14.007),
        (21, "H","H","HN","GLY",4,  4.00,-5.55, 0.00,  0.2719, 1.008),
        (22, "CA","C","CT","GLY",4,  5.95,-4.80, 0.00, -0.0252, 12.011),
        (23, "HA2","H","HC","GLY",4,  6.40,-4.30, 0.88,  0.0698, 1.008),
        (24, "HA3","H","HC","GLY",4,  6.40,-4.30,-0.88,  0.0698, 1.008),
        (25, "C","C","C","GLY",4,  6.50,-6.20, 0.00,  0.5973, 12.011),
        (26, "O","O","O","GLY",4,  5.90,-7.25, 0.00, -0.5679, 15.999),
        # Nme cap
        (27, "N","N","N","NME",5,  7.80,-6.20, 0.00, -0.4157, 14.007),
        (28, "H","H","HN","NME",5,  8.30,-5.33, 0.00,  0.2719, 1.008),
        (29, "CH3","C","CT","NME",5,  8.55,-7.40, 0.00, -0.1490, 12.011),
        (30, "HH31","H","HC","NME",5,  8.10,-7.94, 0.88,  0.0976, 1.008),
        (31, "HH32","H","HC","NME",5,  9.63,-7.28, 0.00,  0.0976, 1.008),
        (32, "HH33","H","HC","NME",5,  8.10,-7.94,-0.88,  0.0976, 1.008),
    ]
    for d in atoms:
        ff.add_atom(Atom(index=d[0], name=d[1], element=d[2], atom_type=d[3],
                         residue=d[4], residue_id=d[5], x=d[6], y=d[7], z=d[8],
                         charge=d[9], mass=d[10]))

    # Backbone bonds for Ace-Gly₃-Nme
    bond_list = [
        (0,1,340,1.09),(0,2,340,1.09),(0,3,340,1.09),(0,4,317,1.522),
        (4,5,570,1.229),(4,6,490,1.335),(6,7,434,1.010),(6,8,337,1.449),
        (8,9,340,1.09),(8,10,340,1.09),(8,11,317,1.522),
        (11,12,570,1.229),(11,13,490,1.335),(13,14,434,1.01),(13,15,337,1.449),
        (15,16,340,1.09),(15,17,340,1.09),(15,18,317,1.522),
        (18,19,570,1.229),(18,20,490,1.335),(20,21,434,1.01),(20,22,337,1.449),
        (22,23,340,1.09),(22,24,340,1.09),(22,25,317,1.522),
        (25,26,570,1.229),(25,27,490,1.335),(27,28,434,1.01),(27,29,337,1.449),
        (29,30,340,1.09),(29,31,340,1.09),(29,32,340,1.09),
    ]
    for i, j, k, r0 in bond_list:
        ff.add_bond(BondParam(atom_i=i, atom_j=j, k=float(k), r0=r0))

    # Representative angles (backbone heavy-atom)
    angle_list = [
        (0,4,5,80,2.10),(0,4,6,70,2.03),(5,4,6,80,2.14),
        (4,6,7,50,2.09),(4,6,8,50,2.12),(7,6,8,50,2.04),
        (6,8,9,50,1.91),(6,8,10,50,1.91),(6,8,11,63,1.91),
        (9,8,10,35,1.91),(9,8,11,50,1.91),(10,8,11,50,1.91),
        (8,11,12,80,2.10),(8,11,13,70,2.03),(12,11,13,80,2.14),
        (11,13,14,50,2.09),(11,13,15,50,2.12),(14,13,15,50,2.04),
        (13,15,16,50,1.91),(13,15,17,50,1.91),(13,15,18,63,1.91),
        (16,15,17,35,1.91),(16,15,18,50,1.91),(17,15,18,50,1.91),
        (15,18,19,80,2.10),(15,18,20,70,2.03),(19,18,20,80,2.14),
        (18,20,21,50,2.09),(18,20,22,50,2.12),(21,20,22,50,2.04),
        (20,22,23,50,1.91),(20,22,24,50,1.91),(20,22,25,63,1.91),
        (23,22,24,35,1.91),(23,22,25,50,1.91),(24,22,25,50,1.91),
        (22,25,26,80,2.10),(22,25,27,70,2.03),(26,25,27,80,2.14),
        (25,27,28,50,2.09),(25,27,29,50,2.12),(28,27,29,50,2.04),
    ]
    for i, j, k, kv, t0 in angle_list:
        ff.add_angle(AngleParam(atom_i=i, atom_j=j, atom_k=k, k=float(kv), theta0=t0))

    # Representative backbone dihedrals
    dih_list = [
        (0,4,6,8, 2.5,math.pi,2),   (4,6,8,11, 0.5,math.pi,2),
        (6,8,11,13, 0.2,math.pi,2), (8,11,13,15, 2.5,math.pi,2),
        (11,13,15,18, 0.5,math.pi,2),(13,15,18,20, 0.2,math.pi,2),
        (15,18,20,22, 2.5,math.pi,2),(18,20,22,25, 0.5,math.pi,2),
        (20,22,25,27, 0.2,math.pi,2),(22,25,27,29, 2.5,math.pi,2),
    ]
    for i, j, k, l, vn, g, n in dih_list:
        ff.add_dihedral(DihedralParam(atom_i=i, atom_j=j, atom_k=k,
                                      atom_l=l, Vn=vn, gamma=g, n=n))

    return ff


def build_alpha_helix_5() -> ForceField:
    """Build a five-residue poly-alanine α-helix.

    Uses explicit Cartesian coordinates derived from ideal helical
    backbone geometry (φ = −57°, ψ = −47°, ω = 180°) with proper
    bond lengths and tetrahedral hydrogen placement.  All inter-atomic
    distances respect van der Waals radii to avoid steric clashes.

    Returns
    -------
    ForceField
        Parameterized α-helix fragment (50 atoms).
    """
    ff = ForceField(name="α-Helix (5-residue poly-Ala)")

    # ── Helper: place atom at distance *d* from *origin* along *direction* ──
    def _place(origin: np.ndarray, direction: np.ndarray, d: float) -> np.ndarray:
        n = np.linalg.norm(direction)
        if n < 1e-12:
            return origin + np.array([d, 0.0, 0.0])
        return origin + direction / n * d

    def _perp(v: np.ndarray) -> np.ndarray:
        """Return a unit vector perpendicular to *v*."""
        if abs(v[0]) < 0.9:
            candidate = np.array([1.0, 0.0, 0.0])
        else:
            candidate = np.array([0.0, 1.0, 0.0])
        p = np.cross(v, candidate)
        return p / np.linalg.norm(p)

    # ── Build backbone via torsion propagation ──
    # Ideal helix: φ = −57°, ψ = −47°, ω = 180°
    # Bond lengths: N-CA 1.449, CA-C 1.522, C-N 1.335
    # Bond angle at each backbone junction ≈ 111° (1.937 rad)

    phi = math.radians(-57.0)
    psi = math.radians(-47.0)
    omega = math.pi  # trans peptide bond

    bl_n_ca = 1.449
    bl_ca_c = 1.522
    bl_c_n = 1.335

    # Backbone bond angles (vary by vertex atom type)
    ba_at_ca = math.radians(111.2)   # N-CA-C angle
    ba_at_c = math.radians(116.2)    # CA-C-N angle
    ba_at_n = math.radians(121.7)    # C-N-CA angle

    def _next_atom(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                   bond_len: float, bond_angle: float, torsion: float) -> np.ndarray:
        """Place atom D given A-B-C chain, B-C-D angle, and A-B-C-D torsion."""
        bc = c - b
        bc_n = bc / np.linalg.norm(bc)
        ab = b - a
        n_vec = np.cross(ab, bc)
        n_norm = np.linalg.norm(n_vec)
        if n_norm < 1e-12:
            n_vec = _perp(bc_n)
        else:
            n_vec = n_vec / n_norm
        m = np.cross(n_vec, bc_n)
        # NeRF local-frame coordinates (note: NO leading minus on dx)
        dx = bond_len * math.cos(math.pi - bond_angle)
        dy = bond_len * math.sin(math.pi - bond_angle) * math.cos(torsion)
        dz = bond_len * math.sin(math.pi - bond_angle) * math.sin(torsion)
        return c + dx * bc_n + dy * m + dz * n_vec

    # Seed first three backbone atoms along +x
    backbone: List[np.ndarray] = []
    backbone.append(np.array([0.0, 0.0, 0.0]))         # N1
    backbone.append(np.array([bl_n_ca, 0.0, 0.0]))     # CA1
    backbone.append(_next_atom(
        np.array([-bl_c_n, 0.0, 0.1]),  # virtual predecessor
        backbone[0], backbone[1],
        bl_ca_c, ba_at_ca, math.radians(120.0),
    ))  # C1 — initial torsion is arbitrary

    # Propagation sequences (after C1 is placed):
    #   step 0: place N2  from [N1, CA1, C1]  using ψ, bond C-N, angle at C
    #   step 1: place CA2 from [CA1, C1, N2]  using ω, bond N-CA, angle at N
    #   step 2: place C2  from [C1, N2, CA2]  using φ, bond CA-C, angle at CA
    torsion_seq = [psi, omega, phi]
    bond_seq = [bl_c_n, bl_n_ca, bl_ca_c]
    angle_seq = [ba_at_c, ba_at_n, ba_at_ca]
    # We need 5 residues × 3 backbone atoms = 15; we have 3, need 12 more
    for step in range(12):
        a, b, c = backbone[-3], backbone[-2], backbone[-1]
        tor = torsion_seq[step % 3]
        bl = bond_seq[step % 3]
        ba = angle_seq[step % 3]
        backbone.append(_next_atom(a, b, c, bl, ba, tor))

    # backbone indices:  0,1,2 = N1,CA1,C1;  3,4,5 = N2,CA2,C2; ...
    # Atom layout per residue (10 atoms each): N H CA HA CB HB1 HB2 HB3 C O
    atoms_data: List[tuple] = []
    bond_defs: List[tuple] = []
    angle_defs: List[tuple] = []
    dihedral_defs: List[tuple] = []
    atom_idx = 0

    prev_c_idx = -1

    for res in range(5):
        base = res * 3  # index into backbone[]
        n_pos = backbone[base]
        ca_pos = backbone[base + 1]
        c_pos = backbone[base + 2]
        rid = res + 1

        # ── N ──
        n_idx = atom_idx
        atoms_data.append((atom_idx, "N", "N", "N", "ALA", rid,
                           n_pos[0], n_pos[1], n_pos[2], -0.4157, 14.007))
        atom_idx += 1

        # ── H on N (opposite to CA direction, in plane, 1.01 Å) ──
        h_idx = atom_idx
        h_dir = n_pos - ca_pos
        if prev_c_idx >= 0:
            prev_c_pos = np.array([atoms_data[prev_c_idx][6],
                                   atoms_data[prev_c_idx][7],
                                   atoms_data[prev_c_idx][8]])
            h_dir = n_pos - (prev_c_pos + ca_pos) / 2.0
        h_pos = _place(n_pos, h_dir, 1.01)
        atoms_data.append((atom_idx, "H", "H", "HN", "ALA", rid,
                           h_pos[0], h_pos[1], h_pos[2], 0.2719, 1.008))
        atom_idx += 1

        # ── CA ──
        ca_idx = atom_idx
        atoms_data.append((atom_idx, "CA", "C", "CT", "ALA", rid,
                           ca_pos[0], ca_pos[1], ca_pos[2], 0.0337, 12.011))
        atom_idx += 1

        # ── Tetrahedral placement of HA and CB around CA ──
        # CA has four sp3 substituents: N, C, HA, CB
        d1 = (n_pos - ca_pos)
        d1 = d1 / np.linalg.norm(d1)
        d2 = (c_pos - ca_pos)
        d2 = d2 / np.linalg.norm(d2)
        d_mid = -(d1 + d2)
        d_mid_norm = np.linalg.norm(d_mid)
        if d_mid_norm < 1e-12:
            d_mid = _perp(d1)
        else:
            d_mid = d_mid / d_mid_norm
        d_perp = np.cross(d1, d2)
        d_perp_norm = np.linalg.norm(d_perp)
        if d_perp_norm < 1e-12:
            d_perp = _perp(d_mid)
        else:
            d_perp = d_perp / d_perp_norm
        # Tetrahedral split angle from midpoint axis
        half_ncc = math.acos(max(-1.0, min(1.0, float(np.dot(d1, d2))))) / 2.0
        cos_half = math.cos(half_ncc)
        cos_alpha = (1.0 / 3.0) / max(cos_half, 0.01)
        cos_alpha = min(cos_alpha, 0.999)
        sin_alpha = math.sqrt(max(0.0, 1.0 - cos_alpha ** 2))
        ha_direction = cos_alpha * d_mid + sin_alpha * d_perp
        cb_direction = cos_alpha * d_mid - sin_alpha * d_perp

        # ── HA ──
        ha_idx = atom_idx
        ha_pos = ca_pos + 1.09 * ha_direction
        atoms_data.append((atom_idx, "HA", "H", "HA", "ALA", rid,
                           ha_pos[0], ha_pos[1], ha_pos[2], 0.0823, 1.008))
        atom_idx += 1

        # ── CB ──
        cb_idx = atom_idx
        cb_pos = ca_pos + 1.526 * cb_direction
        atoms_data.append((atom_idx, "CB", "C", "CT", "ALA", rid,
                           cb_pos[0], cb_pos[1], cb_pos[2], -0.1825, 12.011))
        atom_idx += 1

        # ── HB1, HB2, HB3 (methyl H, 1.09 Å from CB, tetrahedral) ──
        cb_ca_dir = ca_pos - cb_pos
        cb_ca_n = cb_ca_dir / np.linalg.norm(cb_ca_dir)
        p1 = _perp(cb_ca_n)
        p2 = np.cross(cb_ca_n, p1)
        tet_angle = math.radians(109.5)
        # Stagger methyl H by 60° relative to the HA-CA-CB plane
        stagger = math.pi / 3.0
        hb_indices = []
        for k in range(3):
            rot = stagger + k * (2.0 * math.pi / 3.0)
            hx = -math.cos(tet_angle)
            hy = math.sin(tet_angle) * math.cos(rot)
            hz = math.sin(tet_angle) * math.sin(rot)
            hb_dir = hx * cb_ca_n + hy * p1 + hz * p2
            hb_pos = cb_pos + 1.09 * hb_dir
            hb_name = f"HB{k + 1}"
            atoms_data.append((atom_idx, hb_name, "H", "HC", "ALA", rid,
                               hb_pos[0], hb_pos[1], hb_pos[2], 0.0603, 1.008))
            hb_indices.append(atom_idx)
            bond_defs.append((cb_idx, atom_idx, 340.0, 1.09))
            atom_idx += 1

        # ── C ──
        c_idx_cur = atom_idx
        atoms_data.append((atom_idx, "C", "C", "C", "ALA", rid,
                           c_pos[0], c_pos[1], c_pos[2], 0.5973, 12.011))
        atom_idx += 1

        # ── O (carbonyl, 1.229 Å from C, sp2 in peptide plane) ──
        # Place O using NeRF: it's in the peptide plane approximately
        # opposite N(i+1). The torsion N-CA-C-O ≈ ψ + 180° with a
        # correction for sp2 asymmetry (CA-C-O ≈ 120.5°, CA-C-N ≈ 116.2°).
        o_idx = atom_idx
        torsion_o = psi - math.radians(123.3)  # opposite N(i+1) in sp2 plane
        o_pos = _next_atom(n_pos, ca_pos, c_pos, 1.229,
                           math.radians(120.5), torsion_o)
        atoms_data.append((atom_idx, "O", "O", "O", "ALA", rid,
                           o_pos[0], o_pos[1], o_pos[2], -0.5679, 15.999))
        atom_idx += 1

        # ── Intra-residue bonds ──
        bond_defs.append((n_idx, h_idx, 434.0, 1.010))
        bond_defs.append((n_idx, ca_idx, 337.0, 1.449))
        bond_defs.append((ca_idx, ha_idx, 340.0, 1.090))
        bond_defs.append((ca_idx, cb_idx, 310.0, 1.526))
        bond_defs.append((ca_idx, c_idx_cur, 317.0, 1.522))
        bond_defs.append((c_idx_cur, o_idx, 570.0, 1.229))

        # ── Inter-residue peptide bond ──
        if prev_c_idx >= 0:
            prev_o_idx = prev_c_idx + 1  # O always follows C
            bond_defs.append((prev_c_idx, n_idx, 490.0, 1.335))
            angle_defs.append((prev_c_idx, n_idx, ca_idx, 50.0, 2.124))
            angle_defs.append((prev_c_idx, n_idx, h_idx, 35.0, 2.042))
            angle_defs.append((prev_o_idx, prev_c_idx, n_idx, 80.0, 2.101))

        # ── Intra-residue angles ──
        angle_defs.append((h_idx, n_idx, ca_idx, 50.0, 2.042))
        angle_defs.append((n_idx, ca_idx, c_idx_cur, 63.0, 1.911))
        angle_defs.append((n_idx, ca_idx, cb_idx, 80.0, 1.934))
        angle_defs.append((n_idx, ca_idx, ha_idx, 50.0, 1.911))
        angle_defs.append((ha_idx, ca_idx, cb_idx, 35.0, 1.911))
        angle_defs.append((ha_idx, ca_idx, c_idx_cur, 50.0, 1.911))
        angle_defs.append((cb_idx, ca_idx, c_idx_cur, 63.0, 1.911))
        angle_defs.append((ca_idx, c_idx_cur, o_idx, 80.0, 2.101))
        angle_defs.append((ca_idx, cb_idx, hb_indices[0], 50.0, 1.911))
        angle_defs.append((ca_idx, cb_idx, hb_indices[1], 50.0, 1.911))
        angle_defs.append((ca_idx, cb_idx, hb_indices[2], 50.0, 1.911))

        # ── Dihedral: φ = C(i−1)-N-CA-C ──
        if prev_c_idx >= 0:
            dihedral_defs.append((prev_c_idx, n_idx, ca_idx, c_idx_cur,
                                  0.5, math.pi, 2))

        prev_c_idx = c_idx_cur

    # ── Add all atoms ──
    for d in atoms_data:
        ff.add_atom(Atom(index=d[0], name=d[1], element=d[2], atom_type=d[3],
                         residue=d[4], residue_id=d[5],
                         x=d[6], y=d[7], z=d[8], charge=d[9], mass=d[10]))

    # ── Add all bonds, angles, dihedrals ──
    for i, j, k, r0 in bond_defs:
        ff.add_bond(BondParam(atom_i=i, atom_j=j, k=k, r0=r0))
    for i, j, k_idx, k_val, t0 in angle_defs:
        ff.add_angle(AngleParam(atom_i=i, atom_j=j, atom_k=k_idx,
                                k=k_val, theta0=t0))
    for i, j, k, l, vn, g, n in dihedral_defs:
        ff.add_dihedral(DihedralParam(atom_i=i, atom_j=j, atom_k=k,
                                       atom_l=l, Vn=vn, gamma=g, n=n))

    return ff


def build_beta_hairpin() -> ForceField:
    """Build a four-residue β-hairpin with a type-I turn.

    Two extended strands connected by a tight turn, demonstrating
    anti-parallel β-sheet hydrogen-bonding geometry.

    Returns
    -------
    ForceField
        Parameterized β-hairpin.
    """
    ff = ForceField(name="β-Hairpin (4-residue type-I turn)")

    # Simplified 4-residue backbone: Val-Asn-Gly-Thr
    # Extended along x for strand 1, turn, then back for strand 2
    backbone = [
        # res 1: strand
        (0, "N","N","N","VAL",1, 0.0, 0.0, 0.0, -0.4157),
        (1, "CA","C","CT","VAL",1, 1.45, 0.0, 0.0, 0.0337),
        (2, "C","C","C","VAL",1, 2.0, 1.4, 0.0, 0.5973),
        (3, "O","O","O","VAL",1, 1.3, 2.4, 0.0, -0.5679),
        # res 2: turn start
        (4, "N","N","N","ASN",2, 3.3, 1.5, 0.0, -0.4157),
        (5, "CA","C","CT","ASN",2, 4.0, 2.7, 0.0, 0.0337),
        (6, "C","C","C","ASN",2, 3.5, 3.5, 1.2, 0.5973),
        (7, "O","O","O","ASN",2, 4.1, 4.5, 1.5, -0.5679),
        # res 3: turn
        (8, "N","N","N","GLY",3, 2.4, 3.0, 1.9, -0.4157),
        (9, "CA","C","CT","GLY",3, 1.7, 3.5, 3.1, -0.0252),
        (10,"C","C","C","GLY",3, 0.5, 2.6, 3.2, 0.5973),
        (11,"O","O","O","GLY",3, 0.5, 1.5, 2.7, -0.5679),
        # res 4: strand 2
        (12,"N","N","N","THR",4, -0.5, 3.1, 3.8, -0.4157),
        (13,"CA","C","CT","THR",4, -1.7, 2.4, 4.0, 0.0337),
        (14,"C","C","C","THR",4, -2.2, 1.2, 3.2, 0.5973),
        (15,"O","O","O","THR",4, -1.8, 0.1, 3.5, -0.5679),
    ]
    for d in backbone:
        ff.add_atom(Atom(index=d[0], name=d[1], element=d[2], atom_type=d[3],
                         residue=d[4], residue_id=d[5], x=d[6], y=d[7], z=d[8],
                         charge=d[9], mass=12.011 if d[2] in ("C","N") else 15.999))

    # Bonds
    bb_bonds = [
        (0,1,337,1.449),(1,2,317,1.522),(2,3,570,1.229),(2,4,490,1.335),
        (4,5,337,1.449),(5,6,317,1.522),(6,7,570,1.229),(6,8,490,1.335),
        (8,9,337,1.449),(9,10,317,1.522),(10,11,570,1.229),(10,12,490,1.335),
        (12,13,337,1.449),(13,14,317,1.522),(14,15,570,1.229),
    ]
    for i, j, k, r0 in bb_bonds:
        ff.add_bond(BondParam(atom_i=i, atom_j=j, k=float(k), r0=r0))

    # Angles
    bb_angles = [
        (0,1,2,63,1.911),(1,2,3,80,2.101),(1,2,4,70,2.035),(3,2,4,80,2.145),
        (2,4,5,50,2.124),(4,5,6,63,1.911),(5,6,7,80,2.101),(5,6,8,70,2.035),
        (7,6,8,80,2.145),(6,8,9,50,2.124),(8,9,10,63,1.911),(9,10,11,80,2.101),
        (9,10,12,70,2.035),(11,10,12,80,2.145),(10,12,13,50,2.124),
        (12,13,14,63,1.911),(13,14,15,80,2.101),
    ]
    for i, j, k, kv, t0 in bb_angles:
        ff.add_angle(AngleParam(atom_i=i, atom_j=j, atom_k=k, k=float(kv), theta0=t0))

    # Dihedrals
    bb_dihedrals = [
        (0,1,2,4,0.5,math.pi,2),(1,2,4,5,2.5,math.pi,2),
        (2,4,5,6,0.5,math.pi,2),(4,5,6,8,0.2,math.pi,2),
        (5,6,8,9,2.5,math.pi,2),(6,8,9,10,0.5,math.pi,2),
        (8,9,10,12,0.2,math.pi,2),(9,10,12,13,2.5,math.pi,2),
        (10,12,13,14,0.5,math.pi,2),
    ]
    for i, j, k, l, vn, g, n in bb_dihedrals:
        ff.add_dihedral(DihedralParam(atom_i=i, atom_j=j, atom_k=k,
                                      atom_l=l, Vn=vn, gamma=g, n=n))

    return ff


def build_salt_bridge() -> ForceField:
    """Build a Lys-Asp salt bridge model.

    A lysine ammonium (NH₃⁺) paired with an aspartate carboxylate (COO⁻)
    at ~2.8 Å, demonstrating strong electrostatic stabilization.

    Returns
    -------
    ForceField
        Parameterized salt bridge system.
    """
    ff = ForceField(name="Salt Bridge (Lys⁺ · · · Asp⁻)")

    # Lysine side-chain (simplified: Cα-Cβ-Cγ-Cδ-Cε-Nζ + 3H on Nζ)
    lys_atoms = [
        (0, "CA","C","CT","LYS",1, 0.0, 0.0, 0.0,  0.0337, 12.011),
        (1, "CB","C","CT","LYS",1, 1.53, 0.0, 0.0, -0.0482, 12.011),
        (2, "CG","C","CT","LYS",1, 2.30, 1.30, 0.0,  0.0659, 12.011),
        (3, "CD","C","CT","LYS",1, 3.80, 1.30, 0.0, -0.0137, 12.011),
        (4, "CE","C","CT","LYS",1, 4.55, 2.60, 0.0, -0.0187, 12.011),
        (5, "NZ","N","N",  "LYS",1, 6.00, 2.60, 0.0, -0.3854, 14.007),
        (6, "HZ1","H","HN","LYS",1, 6.40, 1.70, 0.0,  0.3400, 1.008),
        (7, "HZ2","H","HN","LYS",1, 6.40, 3.10, 0.87, 0.3400, 1.008),
        (8, "HZ3","H","HN","LYS",1, 6.40, 3.10,-0.87, 0.3400, 1.008),
    ]
    # Aspartate side-chain (Cα-Cβ-Cγ-Oδ1-Oδ2)
    asp_atoms = [
        (9,  "CA","C","CT","ASP",2, 10.0, 0.0, 0.0,  0.0337, 12.011),
        (10, "CB","C","CT","ASP",2,  8.80, 1.00, 0.0, -0.0303, 12.011),
        (11, "CG","C","C", "ASP",2,  8.80, 2.50, 0.0,  0.7994, 12.011),
        (12, "OD1","O","O","ASP",2,  8.00, 2.80, 0.0, -0.8014, 15.999),
        (13, "OD2","O","O","ASP",2,  8.00, 3.20, 0.0, -0.8014, 15.999),
    ]

    for d in lys_atoms + asp_atoms:
        ff.add_atom(Atom(index=d[0], name=d[1], element=d[2], atom_type=d[3],
                         residue=d[4], residue_id=d[5], x=d[6], y=d[7], z=d[8],
                         charge=d[9], mass=d[10]))

    # Bonds
    lys_bonds = [
        (0,1,310,1.526),(1,2,310,1.526),(2,3,310,1.526),
        (3,4,310,1.526),(4,5,337,1.471),(5,6,434,1.010),
        (5,7,434,1.010),(5,8,434,1.010),
    ]
    asp_bonds = [
        (9,10,310,1.526),(10,11,317,1.522),(11,12,570,1.250),(11,13,570,1.250),
    ]
    for i, j, k, r0 in lys_bonds + asp_bonds:
        ff.add_bond(BondParam(atom_i=i, atom_j=j, k=float(k), r0=r0))

    # Angles
    angle_defs = [
        (0,1,2,63,1.911),(1,2,3,63,1.911),(2,3,4,63,1.911),
        (3,4,5,80,1.934),(4,5,6,50,1.911),(4,5,7,50,1.911),
        (4,5,8,50,1.911),(6,5,7,35,1.911),(6,5,8,35,1.911),
        (7,5,8,35,1.911),(9,10,11,63,1.911),(10,11,12,70,2.035),
        (10,11,13,70,2.035),(12,11,13,80,2.145),
    ]
    for i, j, k, kv, t0 in angle_defs:
        ff.add_angle(AngleParam(atom_i=i, atom_j=j, atom_k=k, k=float(kv), theta0=t0))

    # Dihedrals
    dih_defs = [
        (0,1,2,3,0.16,0.0,3),(1,2,3,4,0.16,0.0,3),
        (2,3,4,5,0.16,0.0,3),(3,4,5,6,0.16,0.0,3),
        (9,10,11,12,0.20,math.pi,2),(9,10,11,13,0.20,math.pi,2),
    ]
    for i, j, k, l, vn, g, n in dih_defs:
        ff.add_dihedral(DihedralParam(atom_i=i, atom_j=j, atom_k=k,
                                      atom_l=l, Vn=vn, gamma=g, n=n))

    return ff


def build_disulfide_bond() -> ForceField:
    """Build a disulfide bridge between two cysteine residues.

    The S─S bond (~2.05 Å, k ≈ 166 kcal/(mol·Å²)) is modeled with
    AMBER parameters.  This system demonstrates covalent cross-linking.

    Returns
    -------
    ForceField
        Parameterized disulfide system.
    """
    ff = ForceField(name="Disulfide Bond (Cys-S-S-Cys)")

    atoms_data = [
        # Cys 1
        (0, "N","N","N","CYS",1,  0.00, 0.00, 0.00, -0.4157, 14.007),
        (1, "CA","C","CT","CYS",1, 1.45, 0.00, 0.00,  0.0213, 12.011),
        (2, "CB","C","CT","CYS",1, 2.00, 1.40, 0.00, -0.1231, 12.011),
        (3, "SG","S","S","CYS",1,  3.70, 1.50, 0.00, -0.3119, 32.065),
        (4, "C","C","C","CYS",1,   2.00,-1.20, 0.50,  0.5973, 12.011),
        (5, "O","O","O","CYS",1,   1.30,-2.20, 0.50, -0.5679, 15.999),
        # Cys 2
        (6, "SG","S","S","CYS",2,  5.75, 1.50, 0.00, -0.3119, 32.065),
        (7, "CB","C","CT","CYS",2, 6.50, 0.10, 0.00, -0.1231, 12.011),
        (8, "CA","C","CT","CYS",2, 8.00, 0.00, 0.00,  0.0213, 12.011),
        (9, "N","N","N","CYS",2,   8.50,-1.20, 0.50, -0.4157, 14.007),
        (10,"C","C","C","CYS",2,   8.50, 1.20, 0.00,  0.5973, 12.011),
        (11,"O","O","O","CYS",2,   9.70, 1.30, 0.00, -0.5679, 15.999),
    ]
    for d in atoms_data:
        ff.add_atom(Atom(index=d[0], name=d[1], element=d[2], atom_type=d[3],
                         residue=d[4], residue_id=d[5], x=d[6], y=d[7], z=d[8],
                         charge=d[9], mass=d[10]))

    bond_defs = [
        (0,1,337,1.449),(1,2,310,1.526),(2,3,222,1.810),
        (3,6,166,2.050),  # disulfide S-S
        (6,7,222,1.810),(7,8,310,1.526),(8,9,337,1.449),
        (1,4,317,1.522),(4,5,570,1.229),(8,10,317,1.522),(10,11,570,1.229),
    ]
    for i, j, k, r0 in bond_defs:
        ff.add_bond(BondParam(atom_i=i, atom_j=j, k=float(k), r0=r0))

    angle_defs = [
        (0,1,2,63,1.911),(1,2,3,50,1.911),(2,3,6,68,1.822),
        (3,6,7,68,1.822),(6,7,8,50,1.911),(7,8,9,63,1.911),
        (0,1,4,63,1.911),(1,4,5,80,2.101),(7,8,10,63,1.911),(8,10,11,80,2.101),
    ]
    for i, j, k, kv, t0 in angle_defs:
        ff.add_angle(AngleParam(atom_i=i, atom_j=j, atom_k=k, k=float(kv), theta0=t0))

    dih_defs = [
        (0,1,2,3,0.16,0.0,3),(1,2,3,6,3.50,0.0,2),
        (2,3,6,7,3.50,0.0,2),(3,6,7,8,0.16,0.0,3),
        (6,7,8,9,0.16,0.0,3),
    ]
    for i, j, k, l, vn, g, n in dih_defs:
        ff.add_dihedral(DihedralParam(atom_i=i, atom_j=j, atom_k=k,
                                      atom_l=l, Vn=vn, gamma=g, n=n))

    return ff


# ═══════════════════════════════════════════════════════════════════════
# Builder Registry
# ═══════════════════════════════════════════════════════════════════════

_BUILDERS: Dict[str, object] = {
    "alanine_dipeptide": build_alanine_dipeptide,
    "glycine_tripeptide": build_glycine_tripeptide,
    "alpha_helix_5": build_alpha_helix_5,
    "beta_hairpin": build_beta_hairpin,
    "salt_bridge": build_salt_bridge,
    "disulfide_bond": build_disulfide_bond,
}


def build_molecule(name: str = "alanine_dipeptide") -> ForceField:
    """Dispatch to a preset molecule builder by *name*.

    Parameters
    ----------
    name : str
        One of the keys in ``PRESET_MOLECULES``.

    Returns
    -------
    ForceField
        The fully parameterized system.

    Raises
    ------
    ValueError
        If *name* is not recognized.
    """
    if name not in _BUILDERS:
        raise ValueError(
            f"Unknown molecule '{name}'. Choose from: {', '.join(sorted(_BUILDERS))}"
        )
    builder = _BUILDERS[name]
    return builder()


# ═══════════════════════════════════════════════════════════════════════
# Utility Helpers
# ═══════════════════════════════════════════════════════════════════════

def energy_decomposition_dict(result: EnergyResult) -> Dict[str, float]:
    """Return the five energy terms plus total as a flat dictionary."""
    return {
        "Bonds": result.bond_energy,
        "Angles": result.angle_energy,
        "Dihedrals": result.dihedral_energy,
        "Van der Waals": result.vdw_energy,
        "Electrostatics": result.electrostatic_energy,
        "Total": result.total_energy,
    }


def get_bond_labels(ff: ForceField) -> List[str]:
    """Human-readable labels for each bond: ``'Atom_i — Atom_j'``."""
    labels: List[str] = []
    for b in ff.bonds:
        ai = ff.atoms[b.atom_i]
        aj = ff.atoms[b.atom_j]
        labels.append(f"{ai.name}({ai.residue}{ai.residue_id})—{aj.name}({aj.residue}{aj.residue_id})")
    return labels


def get_atom_labels(ff: ForceField) -> List[str]:
    """Human-readable labels for each atom."""
    return [f"{a.name}({a.residue}{a.residue_id})" for a in ff.atoms]


def topology_summary(ff: ForceField) -> str:
    """One-line topology summary string."""
    return (
        f"{ff.name}: {ff.n_atoms} atoms, {ff.n_bonds} bonds, "
        f"{ff.n_angles} angles, {ff.n_dihedrals} dihedrals"
    )
