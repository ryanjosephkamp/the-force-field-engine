#!/usr/bin/env python3
"""
The Force Field Engine — CLI Entry Point.

Modes:
    --analyze       Full energy decomposition (default)
    --stress        Bond stress analysis
    --stretch       Bond-stretch experiment
    --compare       Compare all preset molecules
    --scan          Parameter sensitivity scan

Usage:
    python main.py
    python main.py --analyze --molecule alanine_dipeptide
    python main.py --stress --save
    python main.py --stretch --bond 5 --displacement 4.0
    python main.py --compare --save
    python main.py --scan --param bond_k
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")

from src.forcefield_engine import (
    ForceField,
    build_molecule,
    compute_energy,
    topology_summary,
    PRESET_MOLECULES,
)
from src.analysis import (
    analyze_energy,
    analyze_stress,
    analyze_bond_stretch,
    analyze_parameter_scan,
    analyze_perturbation,
    forcefield_summary,
)
from src.visualization import MatplotlibRenderer


# ═══════════════════════════════════════════════════════════════════════
# CLI Helpers
# ═══════════════════════════════════════════════════════════════════════

def _ensure_figures() -> str:
    """Create figures/ directory if needed and return its path."""
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def _save_fig(fig, name: str) -> None:
    """Save a matplotlib figure to figures/."""
    fig_dir = _ensure_figures()
    path = os.path.join(fig_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="The Force Field Engine — Molecular Mechanics Energy Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--analyze", action="store_true", default=True,
                      help="Full energy decomposition (default)")
    mode.add_argument("--stress", action="store_true",
                      help="Bond stress analysis")
    mode.add_argument("--stretch", action="store_true",
                      help="Bond-stretch 'Break a Bond' experiment")
    mode.add_argument("--compare", action="store_true",
                      help="Compare all preset molecules")
    mode.add_argument("--scan", action="store_true",
                      help="Parameter sensitivity scan")

    parser.add_argument("--molecule", type=str, default="alanine_dipeptide",
                        choices=PRESET_MOLECULES,
                        help="Preset molecule to analyze")
    parser.add_argument("--bond", type=int, default=0,
                        help="Bond index for stretch/scan")
    parser.add_argument("--displacement", type=float, default=3.0,
                        help="Max displacement for bond stretch (Å)")
    parser.add_argument("--param", type=str, default="bond_k",
                        choices=["bond_k", "bond_r0", "angle_k", "dielectric"],
                        help="Parameter for sensitivity scan")
    parser.add_argument("--save", action="store_true",
                        help="Save figures to figures/")
    parser.add_argument("--verbose", action="store_true",
                        help="Extra output")
    return parser


# ═══════════════════════════════════════════════════════════════════════
# Command Functions
# ═══════════════════════════════════════════════════════════════════════

def cmd_analyze(args: argparse.Namespace) -> None:
    """Full energy decomposition."""
    t0 = time.time()
    print("═" * 60)
    print("  THE FORCE FIELD ENGINE — Energy Decomposition")
    print("═" * 60)

    ff = build_molecule(args.molecule)
    analysis = analyze_energy(ff)
    print(forcefield_summary(analysis))

    if args.save:
        fig = MatplotlibRenderer.energy_decomposition(analysis)
        _save_fig(fig, "energy_decomposition")
        fig2 = MatplotlibRenderer.energy_pie(analysis)
        _save_fig(fig2, "energy_pie")
        fig3 = MatplotlibRenderer.per_bond_energy(analysis)
        _save_fig(fig3, "per_bond_energy")
        fig4 = MatplotlibRenderer.bonded_vs_nonbonded(analysis)
        _save_fig(fig4, "bonded_vs_nonbonded")

    print(f"\n  Completed in {time.time() - t0:.3f}s")


def cmd_stress(args: argparse.Namespace) -> None:
    """Bond stress analysis."""
    t0 = time.time()
    print("═" * 60)
    print("  THE FORCE FIELD ENGINE — Stress Analysis")
    print("═" * 60)

    ff = build_molecule(args.molecule)
    stress = analyze_stress(ff)

    print(f"  Molecule: {ff.name}")
    print(f"  Mean stress: {stress.stress_result.mean_stress:.4f}")
    print(f"  Most stressed: {stress.max_stress_label}")
    print(f"  Most relaxed: {stress.min_stress_label}")
    print(f"  Frustrated residues: {stress.n_frustrated}")
    if stress.stress_result.frustrated_regions:
        print(f"  Residue IDs: {stress.stress_result.frustrated_regions}")

    if args.save:
        fig = MatplotlibRenderer.stress_heatmap(stress)
        _save_fig(fig, "stress_heatmap")

    print(f"\n  Completed in {time.time() - t0:.3f}s")


def cmd_stretch(args: argparse.Namespace) -> None:
    """Bond-stretch experiment."""
    t0 = time.time()
    print("═" * 60)
    print("  THE FORCE FIELD ENGINE — Break a Bond")
    print("═" * 60)

    ff = build_molecule(args.molecule)
    stretch = analyze_bond_stretch(ff, args.bond, args.displacement)

    print(f"  Bond: {stretch.bond_label}")
    print(f"  Energy increase: {stretch.energy_increase:.4f} kcal/mol")
    print(f"  Max force: {stretch.max_force_approx:.2f} kcal/(mol·Å)")
    print(f"  Equilibrium E: {stretch.stretch_result.equilibrium_energy:.4f}")
    print(f"  Breaking E:    {stretch.stretch_result.breaking_energy:.4f}")

    if args.save:
        fig = MatplotlibRenderer.bond_stretch_curve(stretch)
        _save_fig(fig, "bond_stretch")

    print(f"\n  Completed in {time.time() - t0:.3f}s")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare all preset molecules."""
    t0 = time.time()
    print("═" * 60)
    print("  THE FORCE FIELD ENGINE — Molecule Comparison")
    print("═" * 60)

    analyses = {}
    for name in PRESET_MOLECULES:
        ff = build_molecule(name)
        a = analyze_energy(ff)
        analyses[name] = a
        print(f"\n  {name}:")
        print(f"    Total: {a.energy_result.total_energy:.4f} kcal/mol")
        print(f"    Dominant: {a.dominant_term} ({a.dominant_value:.4f})")

    if args.save:
        fig = MatplotlibRenderer.molecule_comparison(analyses)
        _save_fig(fig, "molecule_comparison")

    print(f"\n  Completed in {time.time() - t0:.3f}s")


def cmd_scan(args: argparse.Namespace) -> None:
    """Parameter sensitivity scan."""
    t0 = time.time()
    print("═" * 60)
    print("  THE FORCE FIELD ENGINE — Parameter Scan")
    print("═" * 60)

    ff = build_molecule(args.molecule)
    scan = analyze_parameter_scan(ff, args.param, bond_index=args.bond)

    print(f"  Parameter: {args.param}")
    print(f"  Sensitivity: {scan.sensitivity:.4f}")
    print(f"  Energy range: {min(scan.total_energies):.4f} → {max(scan.total_energies):.4f}")

    if args.save:
        fig = MatplotlibRenderer.parameter_scan(scan)
        _save_fig(fig, f"param_scan_{args.param}")

    print(f"\n  Completed in {time.time() - t0:.3f}s")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    """Parse arguments and dispatch to command function."""
    parser = build_parser()
    args = parser.parse_args()

    if args.stress:
        cmd_stress(args)
    elif args.stretch:
        cmd_stretch(args)
    elif args.compare:
        cmd_compare(args)
    elif args.scan:
        cmd_scan(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()
