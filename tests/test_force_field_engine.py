"""
The Force Field Engine — Comprehensive Test Suite.

20 test classes, 100+ test methods covering:
    - Engine: Atom, BondParam, AngleParam, DihedralParam dataclasses
    - ForceField construction and topology
    - Energy calculators: bond, angle, dihedral, non-bonded
    - Stress analysis and frustrated regions
    - Bond-stretch experiment
    - Parameter perturbation and scanning
    - Preset molecule builders and registry
    - Analysis pipelines
    - Plotly + Matplotlib renderers
    - CLI argument parsing
    - Integration end-to-end
    - Edge cases
"""

from __future__ import annotations

import math
import sys
import os

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go

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
    COULOMB_CONSTANT,
    LJ_PARAMS,
    _distance,
    _angle,
    _dihedral,
    _lj_combine,
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
# Module-Level Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def dipeptide():
    return build_alanine_dipeptide()


@pytest.fixture
def tripeptide():
    return build_glycine_tripeptide()


@pytest.fixture
def helix():
    return build_alpha_helix_5()


@pytest.fixture
def dipeptide_energy(dipeptide):
    return analyze_energy(dipeptide)


@pytest.fixture
def dipeptide_stress(dipeptide):
    return analyze_stress(dipeptide)


@pytest.fixture
def simple_ff():
    """Minimal two-atom system for unit tests."""
    ff = ForceField(name="H-H test")
    ff.add_atom(Atom(index=0, name="H1", element="H", atom_type="H",
                      x=0.0, y=0.0, z=0.0, charge=0.0))
    ff.add_atom(Atom(index=1, name="H2", element="H", atom_type="H",
                      x=1.5, y=0.0, z=0.0, charge=0.0))
    ff.add_bond(BondParam(atom_i=0, atom_j=1, k=340.0, r0=1.5))
    return ff


# ═══════════════════════════════════════════════════════════════════════
# 1. Atom Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAtom:
    """Unit tests for the Atom dataclass."""

    def test_creation(self):
        a = Atom(index=0, name="CA", element="C")
        assert a.index == 0
        assert a.name == "CA"
        assert a.element == "C"

    def test_default_values(self):
        a = Atom(index=0, name="X", element="X")
        assert a.x == 0.0
        assert a.charge == 0.0
        assert a.mass == 12.011

    def test_position_property(self):
        a = Atom(index=0, name="X", element="X", x=1.0, y=2.0, z=3.0)
        pos = a.position
        np.testing.assert_array_almost_equal(pos, [1.0, 2.0, 3.0])

    def test_atom_type_default(self):
        a = Atom(index=0, name="X", element="X")
        assert a.atom_type == "C"

    def test_residue_default(self):
        a = Atom(index=0, name="X", element="X")
        assert a.residue == "ALA"
        assert a.residue_id == 1


# ═══════════════════════════════════════════════════════════════════════
# 2. BondParam Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBondParam:
    """Unit tests for BondParam dataclass."""

    def test_creation(self):
        b = BondParam(atom_i=0, atom_j=1)
        assert b.atom_i == 0
        assert b.atom_j == 1
        assert b.k == 340.0
        assert b.r0 == 1.526

    def test_custom_params(self):
        b = BondParam(atom_i=2, atom_j=3, k=500.0, r0=1.10)
        assert b.k == 500.0
        assert b.r0 == 1.10


# ═══════════════════════════════════════════════════════════════════════
# 3. AngleParam Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAngleParam:
    """Unit tests for AngleParam dataclass."""

    def test_creation(self):
        a = AngleParam(atom_i=0, atom_j=1, atom_k=2)
        assert a.atom_i == 0
        assert a.atom_j == 1
        assert a.atom_k == 2

    def test_defaults(self):
        a = AngleParam(atom_i=0, atom_j=1, atom_k=2)
        assert a.k == 63.0
        assert abs(a.theta0 - 1.9106) < 0.001


# ═══════════════════════════════════════════════════════════════════════
# 4. DihedralParam Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDihedralParam:
    """Unit tests for DihedralParam dataclass."""

    def test_creation(self):
        d = DihedralParam(atom_i=0, atom_j=1, atom_k=2, atom_l=3)
        assert d.Vn == 1.4
        assert d.n == 3

    def test_custom_params(self):
        d = DihedralParam(atom_i=0, atom_j=1, atom_k=2, atom_l=3,
                          Vn=2.5, gamma=math.pi, n=2)
        assert d.Vn == 2.5
        assert abs(d.gamma - math.pi) < 1e-10


# ═══════════════════════════════════════════════════════════════════════
# 5. ForceField Construction Tests
# ═══════════════════════════════════════════════════════════════════════

class TestForceFieldConstruction:
    """Unit tests for ForceField topology building."""

    def test_empty_system(self):
        ff = ForceField(name="empty")
        assert ff.n_atoms == 0
        assert ff.n_bonds == 0

    def test_add_atom(self, simple_ff):
        assert simple_ff.n_atoms == 2

    def test_add_bond_creates_exclusion(self, simple_ff):
        assert (0, 1) in simple_ff._exclusion_set

    def test_add_angle_creates_exclusion(self):
        ff = ForceField()
        ff.add_atom(Atom(index=0, name="A", element="C", x=0, y=0, z=0))
        ff.add_atom(Atom(index=1, name="B", element="C", x=1, y=0, z=0))
        ff.add_atom(Atom(index=2, name="C", element="C", x=2, y=0, z=0))
        ff.add_angle(AngleParam(atom_i=0, atom_j=1, atom_k=2))
        assert (0, 2) in ff._exclusion_set

    def test_positions_property(self, simple_ff):
        pos = simple_ff.positions
        assert pos.shape == (2, 3)
        np.testing.assert_almost_equal(pos[1, 0], 1.5)

    def test_set_positions(self, simple_ff):
        new_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        simple_ff.set_positions(new_coords)
        assert simple_ff.atoms[1].x == 2.0

    def test_copy_is_independent(self, simple_ff):
        copy = simple_ff.copy()
        copy.atoms[0].x = 999.0
        assert simple_ff.atoms[0].x != 999.0

    def test_build_exclusions(self, simple_ff):
        simple_ff._exclusion_set.clear()
        simple_ff.build_exclusions()
        assert (0, 1) in simple_ff._exclusion_set


# ═══════════════════════════════════════════════════════════════════════
# 6. Distance and Geometry Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGeometry:
    """Unit tests for distance, angle, and dihedral calculations."""

    def test_distance_same_point(self):
        a = Atom(index=0, name="A", element="C", x=0, y=0, z=0)
        assert _distance(a, a) == 0.0

    def test_distance_x_axis(self):
        a = Atom(index=0, name="A", element="C", x=0, y=0, z=0)
        b = Atom(index=1, name="B", element="C", x=3, y=4, z=0)
        assert abs(_distance(a, b) - 5.0) < 1e-10

    def test_angle_90_degrees(self):
        a = Atom(index=0, name="A", element="C", x=1, y=0, z=0)
        b = Atom(index=1, name="B", element="C", x=0, y=0, z=0)
        c = Atom(index=2, name="C", element="C", x=0, y=1, z=0)
        angle = _angle(a, b, c)
        assert abs(angle - math.pi / 2) < 1e-10

    def test_angle_180_degrees(self):
        a = Atom(index=0, name="A", element="C", x=-1, y=0, z=0)
        b = Atom(index=1, name="B", element="C", x=0, y=0, z=0)
        c = Atom(index=2, name="C", element="C", x=1, y=0, z=0)
        angle = _angle(a, b, c)
        assert abs(angle - math.pi) < 1e-6

    def test_dihedral_returns_float(self):
        a = Atom(index=0, name="A", element="C", x=1, y=0, z=0)
        b = Atom(index=1, name="B", element="C", x=0, y=0, z=0)
        c = Atom(index=2, name="C", element="C", x=0, y=1, z=0)
        d = Atom(index=3, name="D", element="C", x=0, y=1, z=1)
        phi = _dihedral(a, b, c, d)
        assert isinstance(phi, float)


# ═══════════════════════════════════════════════════════════════════════
# 7. Bond Energy Calculator Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBondEnergy:
    """Unit tests for harmonic bond energy calculation."""

    def test_equilibrium_is_zero(self, simple_ff):
        total, per_bond = compute_bond_energy(simple_ff)
        assert abs(total) < 1e-10

    def test_stretched_bond_positive(self):
        ff = ForceField()
        ff.add_atom(Atom(index=0, name="A", element="C", x=0, y=0, z=0))
        ff.add_atom(Atom(index=1, name="B", element="C", x=2.0, y=0, z=0))
        ff.add_bond(BondParam(atom_i=0, atom_j=1, k=340.0, r0=1.5))
        total, _ = compute_bond_energy(ff)
        expected = 0.5 * 340.0 * (2.0 - 1.5) ** 2
        assert abs(total - expected) < 1e-10

    def test_compressed_bond_positive(self):
        ff = ForceField()
        ff.add_atom(Atom(index=0, name="A", element="C", x=0, y=0, z=0))
        ff.add_atom(Atom(index=1, name="B", element="C", x=1.0, y=0, z=0))
        ff.add_bond(BondParam(atom_i=0, atom_j=1, k=340.0, r0=1.5))
        total, _ = compute_bond_energy(ff)
        assert total > 0

    def test_per_bond_length_matches(self, dipeptide):
        _, per_bond = compute_bond_energy(dipeptide)
        assert len(per_bond) == dipeptide.n_bonds


# ═══════════════════════════════════════════════════════════════════════
# 8. Angle Energy Calculator Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAngleEnergy:
    """Unit tests for harmonic angle energy calculation."""

    def test_angle_at_equilibrium(self):
        ff = ForceField()
        # Place atoms at exact tetrahedral angle (109.47°)
        ff.add_atom(Atom(index=0, name="A", element="C", x=1, y=0, z=0))
        ff.add_atom(Atom(index=1, name="B", element="C", x=0, y=0, z=0))
        # cos(109.47) ≈ -0.3338 → y ~= sin(109.47) ≈ 0.9426
        theta0 = math.radians(109.47)
        ff.add_atom(Atom(index=2, name="C", element="C",
                         x=math.cos(theta0), y=math.sin(theta0), z=0))
        ff.add_angle(AngleParam(atom_i=0, atom_j=1, atom_k=2, k=63.0, theta0=theta0))
        total, _ = compute_angle_energy(ff)
        assert abs(total) < 1e-6

    def test_per_angle_length(self, dipeptide):
        _, per_angle = compute_angle_energy(dipeptide)
        assert len(per_angle) == dipeptide.n_angles


# ═══════════════════════════════════════════════════════════════════════
# 9. Dihedral Energy Calculator Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDihedralEnergy:
    """Unit tests for periodic dihedral energy calculation."""

    def test_dihedral_bounded(self, dipeptide):
        total, per_dih = compute_dihedral_energy(dipeptide)
        for e in per_dih:
            # Max possible = Vn (when cos = 1, + 1 = 2, / 2 = Vn)
            assert e >= 0

    def test_per_dihedral_length(self, dipeptide):
        _, per_dih = compute_dihedral_energy(dipeptide)
        assert len(per_dih) == dipeptide.n_dihedrals


# ═══════════════════════════════════════════════════════════════════════
# 10. Non-Bonded Energy Tests
# ═══════════════════════════════════════════════════════════════════════

class TestNonBondedEnergy:
    """Unit tests for LJ and Coulomb energy calculations."""

    def test_lj_combine_same_type(self):
        sigma, epsilon = _lj_combine("C", "C")
        assert abs(sigma - 1.9080) < 1e-4
        assert abs(epsilon - 0.0860) < 1e-4

    def test_lj_combine_different_types(self):
        sigma, epsilon = _lj_combine("C", "N")
        expected_s = (1.9080 + 1.8240) / 2
        expected_e = math.sqrt(0.0860 * 0.1700)
        assert abs(sigma - expected_s) < 1e-4
        assert abs(epsilon - expected_e) < 1e-4

    def test_exclusions_skip_bonded(self, simple_ff):
        vdw, elec, vdw_pairs, elec_pairs = compute_nonbonded_energy(simple_ff)
        # Only 2 atoms, bonded → excluded → no non-bonded
        assert len(vdw_pairs) == 0
        assert len(elec_pairs) == 0

    def test_charged_atoms_coulomb(self):
        ff = ForceField()
        ff.add_atom(Atom(index=0, name="A", element="N", atom_type="N",
                         x=0, y=0, z=0, charge=1.0))
        ff.add_atom(Atom(index=1, name="B", element="O", atom_type="O",
                         x=3.0, y=0, z=0, charge=-1.0))
        _, elec, _, elec_pairs = compute_nonbonded_energy(ff)
        expected_elec = COULOMB_CONSTANT * 1.0 * (-1.0) / (1.0 * 3.0)
        assert abs(elec - expected_elec) < 1e-3

    def test_cutoff_excludes_distant(self):
        ff = ForceField(cutoff=5.0)
        ff.add_atom(Atom(index=0, name="A", element="C", x=0, y=0, z=0))
        ff.add_atom(Atom(index=1, name="B", element="C", x=10.0, y=0, z=0))
        vdw, elec, _, _ = compute_nonbonded_energy(ff)
        assert abs(vdw) < 1e-15
        assert abs(elec) < 1e-15


# ═══════════════════════════════════════════════════════════════════════
# 11. Full Energy Computation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestComputeEnergy:
    """Unit tests for compute_energy()."""

    def test_returns_energy_result(self, dipeptide):
        result = compute_energy(dipeptide)
        assert isinstance(result, EnergyResult)
        assert result.status == "success"

    def test_total_is_sum(self, dipeptide):
        result = compute_energy(dipeptide)
        expected = (result.bond_energy + result.angle_energy +
                    result.dihedral_energy + result.vdw_energy +
                    result.electrostatic_energy)
        assert abs(result.total_energy - expected) < 1e-10

    def test_energy_decomposition_dict(self, dipeptide):
        result = compute_energy(dipeptide)
        d = energy_decomposition_dict(result)
        assert "Bonds" in d
        assert "Total" in d
        assert len(d) == 6


# ═══════════════════════════════════════════════════════════════════════
# 12. Stress Analysis Tests
# ═══════════════════════════════════════════════════════════════════════

class TestStressAnalysis:
    """Unit tests for compute_stress()."""

    def test_returns_stress_result(self, dipeptide):
        result = compute_stress(dipeptide)
        assert isinstance(result, StressResult)

    def test_stresses_normalized(self, dipeptide):
        result = compute_stress(dipeptide)
        for s in result.bond_stresses:
            assert 0.0 <= s <= 1.0 + 1e-10

    def test_max_stress_is_one(self, dipeptide):
        result = compute_stress(dipeptide)
        assert abs(max(result.bond_stresses) - 1.0) < 1e-10

    def test_frustrated_regions_type(self, dipeptide):
        result = compute_stress(dipeptide)
        assert isinstance(result.frustrated_regions, list)


# ═══════════════════════════════════════════════════════════════════════
# 13. Bond Stretch Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBondStretch:
    """Unit tests for stretch_bond()."""

    def test_returns_result(self, dipeptide):
        result = stretch_bond(dipeptide, bond_index=0, max_displacement=1.0, n_steps=10)
        assert isinstance(result, BondStretchResult)
        assert len(result.displacements) == 11

    def test_energy_increases(self, dipeptide):
        result = stretch_bond(dipeptide, bond_index=0, max_displacement=2.0, n_steps=20)
        assert result.breaking_energy > result.equilibrium_energy

    def test_invalid_bond_index(self, dipeptide):
        with pytest.raises(ValueError, match="Bond index"):
            stretch_bond(dipeptide, bond_index=999)

    def test_negative_bond_index(self, dipeptide):
        with pytest.raises(ValueError, match="Bond index"):
            stretch_bond(dipeptide, bond_index=-1)


# ═══════════════════════════════════════════════════════════════════════
# 14. Perturbation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPerturbation:
    """Unit tests for perturb_coordinates()."""

    def test_returns_forcefield(self, simple_ff):
        result = perturb_coordinates(simple_ff, scale=0.1, seed=42)
        assert isinstance(result, ForceField)

    def test_positions_differ(self, simple_ff):
        result = perturb_coordinates(simple_ff, scale=1.0, seed=42)
        assert result.atoms[0].x != simple_ff.atoms[0].x

    def test_seed_reproducibility(self, simple_ff):
        r1 = perturb_coordinates(simple_ff, scale=0.5, seed=7)
        r2 = perturb_coordinates(simple_ff, scale=0.5, seed=7)
        assert r1.atoms[0].x == r2.atoms[0].x

    def test_zero_scale_unchanged(self, simple_ff):
        result = perturb_coordinates(simple_ff, scale=0.0, seed=42)
        assert abs(result.atoms[0].x - simple_ff.atoms[0].x) < 1e-15


# ═══════════════════════════════════════════════════════════════════════
# 15. Parameter Scan Tests
# ═══════════════════════════════════════════════════════════════════════

class TestParameterScan:
    """Unit tests for scan_parameter()."""

    def test_returns_list(self, dipeptide):
        results = scan_parameter(dipeptide, "bond_k", [100, 200, 300])
        assert len(results) == 3
        assert isinstance(results[0][1], EnergyResult)

    def test_invalid_param(self, dipeptide):
        with pytest.raises(ValueError, match="Unknown parameter"):
            scan_parameter(dipeptide, "invalid_param")

    def test_default_values_used(self, dipeptide):
        results = scan_parameter(dipeptide, "bond_k")
        assert len(results) == 20


# ═══════════════════════════════════════════════════════════════════════
# 16. Preset Builder Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPresetBuilders:
    """Unit tests for all preset molecule builders."""

    def test_alanine_dipeptide(self):
        ff = build_alanine_dipeptide()
        assert ff.n_atoms == 22
        assert ff.n_bonds > 0
        assert ff.n_angles > 0
        assert ff.n_dihedrals > 0

    def test_glycine_tripeptide(self):
        ff = build_glycine_tripeptide()
        assert ff.n_atoms == 33
        assert ff.n_bonds > 0

    def test_alpha_helix(self):
        ff = build_alpha_helix_5()
        assert ff.n_atoms > 0
        assert "helix" in ff.name.lower() or "Helix" in ff.name

    def test_beta_hairpin(self):
        ff = build_beta_hairpin()
        assert ff.n_atoms == 16
        assert ff.n_bonds > 0

    def test_salt_bridge(self):
        ff = build_salt_bridge()
        assert ff.n_atoms == 14

    def test_disulfide_bond(self):
        ff = build_disulfide_bond()
        assert ff.n_atoms == 12

    def test_build_molecule_dispatch(self):
        for name in PRESET_MOLECULES:
            ff = build_molecule(name)
            assert isinstance(ff, ForceField)
            assert ff.n_atoms > 0

    def test_build_molecule_invalid(self):
        with pytest.raises(ValueError, match="Unknown molecule"):
            build_molecule("nonexistent_molecule")

    def test_molecule_descriptions(self):
        for name in PRESET_MOLECULES:
            assert name in MOLECULE_DESCRIPTIONS


# ═══════════════════════════════════════════════════════════════════════
# 17. Analysis Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAnalysisPipelines:
    """Unit tests for analysis module functions."""

    def test_analyze_energy(self, dipeptide_energy):
        assert isinstance(dipeptide_energy, FullEnergyAnalysis)
        assert dipeptide_energy.molecule_name != ""
        assert dipeptide_energy.dominant_term != ""
        assert len(dipeptide_energy.explanation) > 0

    def test_analyze_stress(self, dipeptide_stress):
        assert isinstance(dipeptide_stress, StressAnalysis)
        assert len(dipeptide_stress.bond_labels) > 0

    def test_analyze_bond_stretch(self, dipeptide):
        result = analyze_bond_stretch(dipeptide, bond_index=0)
        assert isinstance(result, BondStretchAnalysis)
        assert result.energy_increase >= 0

    def test_analyze_parameter_scan(self, dipeptide):
        result = analyze_parameter_scan(dipeptide, "bond_k")
        assert isinstance(result, ParameterScanAnalysis)
        assert len(result.param_values) > 0

    def test_analyze_perturbation(self, dipeptide):
        result = analyze_perturbation(dipeptide, n_samples=3, seed=42)
        assert isinstance(result, PerturbationAnalysis)
        assert len(result.scales) > 0

    def test_forcefield_summary(self, dipeptide_energy):
        summary = forcefield_summary(dipeptide_energy)
        assert isinstance(summary, str)
        assert "Total" in summary or "Energy" in summary


# ═══════════════════════════════════════════════════════════════════════
# 18. Plotly Renderer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPlotlyRenderer:
    """Unit tests for PlotlyRenderer static methods."""

    def test_energy_decomposition(self, dipeptide_energy):
        fig = PlotlyRenderer.energy_decomposition(dipeptide_energy)
        assert isinstance(fig, go.Figure)

    def test_energy_pie(self, dipeptide_energy):
        fig = PlotlyRenderer.energy_pie(dipeptide_energy)
        assert isinstance(fig, go.Figure)

    def test_per_bond_energy(self, dipeptide_energy):
        fig = PlotlyRenderer.per_bond_energy(dipeptide_energy)
        assert isinstance(fig, go.Figure)

    def test_stress_heatmap(self, dipeptide_stress):
        fig = PlotlyRenderer.stress_heatmap(dipeptide_stress)
        assert isinstance(fig, go.Figure)

    def test_molecular_structure_3d(self, dipeptide, dipeptide_stress):
        fig = PlotlyRenderer.molecular_structure_3d(dipeptide, dipeptide_stress)
        assert isinstance(fig, go.Figure)

    def test_molecular_structure_3d_no_stress(self, dipeptide):
        fig = PlotlyRenderer.molecular_structure_3d(dipeptide)
        assert isinstance(fig, go.Figure)

    def test_bond_stretch_curve(self, dipeptide):
        stretch = analyze_bond_stretch(dipeptide, bond_index=0, n_steps=10)
        fig = PlotlyRenderer.bond_stretch_curve(stretch)
        assert isinstance(fig, go.Figure)

    def test_parameter_scan(self, dipeptide):
        scan = analyze_parameter_scan(dipeptide, "bond_k")
        fig = PlotlyRenderer.parameter_scan(scan)
        assert isinstance(fig, go.Figure)

    def test_perturbation_robustness(self, dipeptide):
        pert = analyze_perturbation(dipeptide, n_samples=3)
        fig = PlotlyRenderer.perturbation_robustness(pert)
        assert isinstance(fig, go.Figure)

    def test_molecule_comparison(self):
        analyses = {}
        for name in PRESET_MOLECULES[:2]:
            ff = build_molecule(name)
            analyses[name] = analyze_energy(ff)
        fig = PlotlyRenderer.molecule_comparison(analyses)
        assert isinstance(fig, go.Figure)

    def test_bonded_vs_nonbonded(self, dipeptide_energy):
        fig = PlotlyRenderer.bonded_vs_nonbonded(dipeptide_energy)
        assert isinstance(fig, go.Figure)


# ═══════════════════════════════════════════════════════════════════════
# 19. Matplotlib Renderer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMatplotlibRenderer:
    """Unit tests for MatplotlibRenderer static methods."""

    def test_energy_decomposition(self, dipeptide_energy):
        fig = MatplotlibRenderer.energy_decomposition(dipeptide_energy)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_energy_pie(self, dipeptide_energy):
        fig = MatplotlibRenderer.energy_pie(dipeptide_energy)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_per_bond_energy(self, dipeptide_energy):
        fig = MatplotlibRenderer.per_bond_energy(dipeptide_energy)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_stress_heatmap(self, dipeptide_stress):
        fig = MatplotlibRenderer.stress_heatmap(dipeptide_stress)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bond_stretch_curve(self, dipeptide):
        stretch = analyze_bond_stretch(dipeptide, bond_index=0, n_steps=10)
        fig = MatplotlibRenderer.bond_stretch_curve(stretch)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_parameter_scan(self, dipeptide):
        scan = analyze_parameter_scan(dipeptide, "bond_k")
        fig = MatplotlibRenderer.parameter_scan(scan)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_perturbation_robustness(self, dipeptide):
        pert = analyze_perturbation(dipeptide, n_samples=3)
        fig = MatplotlibRenderer.perturbation_robustness(pert)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_molecule_comparison(self):
        analyses = {}
        for name in PRESET_MOLECULES[:2]:
            ff = build_molecule(name)
            analyses[name] = analyze_energy(ff)
        fig = MatplotlibRenderer.molecule_comparison(analyses)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bonded_vs_nonbonded(self, dipeptide_energy):
        fig = MatplotlibRenderer.bonded_vs_nonbonded(dipeptide_energy)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 20. CLI Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCLI:
    """Unit tests for CLI argument parsing."""

    def test_build_parser(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args([])
        assert args.molecule == "alanine_dipeptide"

    def test_parser_molecule_flag(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--molecule", "salt_bridge"])
        assert args.molecule == "salt_bridge"

    def test_parser_save_flag(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--save"])
        assert args.save is True

    def test_parser_stress_mode(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--stress"])
        assert args.stress is True

    def test_parser_stretch_mode(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--stretch", "--bond", "3", "--displacement", "2.5"])
        assert args.stretch is True
        assert args.bond == 3
        assert args.displacement == 2.5


# ═══════════════════════════════════════════════════════════════════════
# 21. Utility Function Tests
# ═══════════════════════════════════════════════════════════════════════

class TestUtilities:
    """Unit tests for utility helper functions."""

    def test_get_bond_labels(self, dipeptide):
        labels = get_bond_labels(dipeptide)
        assert len(labels) == dipeptide.n_bonds
        assert all(isinstance(l, str) for l in labels)

    def test_get_atom_labels(self, dipeptide):
        labels = get_atom_labels(dipeptide)
        assert len(labels) == dipeptide.n_atoms

    def test_topology_summary(self, dipeptide):
        s = topology_summary(dipeptide)
        assert "22 atoms" in s
        assert isinstance(s, str)


# ═══════════════════════════════════════════════════════════════════════
# 22. Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_dipeptide(self):
        ff = build_molecule("alanine_dipeptide")
        energy_analysis = analyze_energy(ff)
        stress_analysis = analyze_stress(ff)
        stretch_analysis = analyze_bond_stretch(ff, bond_index=0, n_steps=5)

        assert energy_analysis.energy_result.status == "success"
        assert stress_analysis.stress_result.mean_stress >= 0
        assert stretch_analysis.energy_increase >= 0

        fig1 = PlotlyRenderer.energy_decomposition(energy_analysis)
        fig2 = PlotlyRenderer.stress_heatmap(stress_analysis)
        fig3 = PlotlyRenderer.bond_stretch_curve(stretch_analysis)
        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)
        assert isinstance(fig3, go.Figure)

    def test_all_molecules_compute(self):
        for name in PRESET_MOLECULES:
            ff = build_molecule(name)
            result = compute_energy(ff)
            assert result.status == "success"
            assert isinstance(result.total_energy, float)

    def test_perturbed_energy_changes(self):
        ff = build_molecule("alanine_dipeptide")
        baseline = compute_energy(ff).total_energy
        perturbed = perturb_coordinates(ff, scale=0.5, seed=42)
        perturbed_e = compute_energy(perturbed).total_energy
        assert perturbed_e != baseline


# ═══════════════════════════════════════════════════════════════════════
# 23. Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge-case and boundary tests."""

    def test_empty_forcefield_energy(self):
        ff = ForceField(name="empty")
        result = compute_energy(ff)
        assert result.total_energy == 0.0

    def test_empty_stress(self):
        ff = ForceField(name="empty")
        result = compute_stress(ff)
        assert result.mean_stress == 0.0

    def test_single_atom_energy(self):
        ff = ForceField()
        ff.add_atom(Atom(index=0, name="H", element="H"))
        result = compute_energy(ff)
        assert result.total_energy == 0.0

    def test_per_bond_empty_visualization(self):
        empty_analysis = FullEnergyAnalysis()
        fig = PlotlyRenderer.per_bond_energy(empty_analysis)
        assert isinstance(fig, go.Figure)

    def test_empty_stress_heatmap(self):
        empty_stress = StressAnalysis()
        fig = PlotlyRenderer.stress_heatmap(empty_stress)
        assert isinstance(fig, go.Figure)

    def test_empty_stretch_curve(self):
        empty_stretch = BondStretchAnalysis()
        fig = PlotlyRenderer.bond_stretch_curve(empty_stretch)
        assert isinstance(fig, go.Figure)

    def test_very_close_atoms(self):
        ff = ForceField()
        ff.add_atom(Atom(index=0, name="A", element="C", x=0, y=0, z=0))
        ff.add_atom(Atom(index=1, name="B", element="C", x=0.001, y=0, z=0))
        result = compute_energy(ff)
        assert isinstance(result.total_energy, float)
        assert not math.isnan(result.total_energy)

    def test_large_dielectric(self):
        ff = ForceField(dielectric=80.0)
        ff.add_atom(Atom(index=0, name="A", element="N", atom_type="N",
                         x=0, y=0, z=0, charge=1.0))
        ff.add_atom(Atom(index=1, name="B", element="O", atom_type="O",
                         x=5.0, y=0, z=0, charge=-1.0))
        _, elec, _, _ = compute_nonbonded_energy(ff)
        # Elec should be very small in water (ε=80)
        assert abs(elec) < 2.0
