"""
The Force Field Engine — Plotly and Matplotlib Renderers.

Provides dual static-method renderer classes for interactive Streamlit
dashboards (PlotlyRenderer) and publication-quality CLI figures
(MatplotlibRenderer).  Every visualization method is duplicated across
both renderers with matching signatures where applicable.

Classes
-------
PlotlyRenderer
    Interactive Plotly figures for Streamlit dashboards.
MatplotlibRenderer
    Publication-quality Matplotlib figures for CLI / PDF export.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover
    go = None  # type: ignore
    make_subplots = None  # type: ignore

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
except ImportError:  # pragma: no cover
    plt = None  # type: ignore

from src.forcefield_engine import (
    ForceField,
    EnergyResult,
    StressResult,
    BondStretchResult,
    Atom,
    ENERGY_TERM_COLORS,
    STRESS_COLORSCALE,
    ATOM_TYPE_COLORS,
    get_bond_labels,
    get_atom_labels,
)
from src.analysis import (
    FullEnergyAnalysis,
    StressAnalysis,
    BondStretchAnalysis,
    ParameterScanAnalysis,
    PerturbationAnalysis,
)


# ═══════════════════════════════════════════════════════════════════════
# Color Constants
# ═══════════════════════════════════════════════════════════════════════

BOND_COLOR = "#636efa"
ANGLE_COLOR = "#EF553B"
DIHEDRAL_COLOR = "#00cc96"
VDW_COLOR = "#ab63fa"
ELEC_COLOR = "#FFA15A"
TOTAL_COLOR = "#19d3f3"

STRESS_HIGH = "#b2182b"
STRESS_MID = "#f7f7f7"
STRESS_LOW = "#2166ac"

BAR_COLORS = [BOND_COLOR, ANGLE_COLOR, DIHEDRAL_COLOR, VDW_COLOR, ELEC_COLOR]
TERM_NAMES = ["Bonds", "Angles", "Dihedrals", "Van der Waals", "Electrostatics"]


# ═══════════════════════════════════════════════════════════════════════
# Plotly Renderer
# ═══════════════════════════════════════════════════════════════════════

class PlotlyRenderer:
    """Interactive Plotly figures for Streamlit dashboards."""

    # ── Energy Decomposition Bar ─────────────────────────────────
    @staticmethod
    def energy_decomposition(analysis: FullEnergyAnalysis, height: int = 500) -> go.Figure:
        """Stacked bar chart of the five energy terms.

        Parameters
        ----------
        analysis : FullEnergyAnalysis
            Energy analysis result.
        height : int
            Figure height in pixels.

        Returns
        -------
        go.Figure
        """
        decomp = analysis.decomposition
        terms = [k for k in decomp if k != "Total"]
        values = [decomp[k] for k in terms]
        colors = BAR_COLORS[:len(terms)]

        fig = go.Figure()
        for t, v, c in zip(terms, values, colors):
            fig.add_trace(go.Bar(
                x=[t], y=[v], name=t,
                marker_color=c,
                text=[f"{v:.2f}"],
                textposition="outside",
            ))

        fig.update_layout(
            title=f"Energy Decomposition — {analysis.molecule_name}",
            yaxis_title="Energy (kcal/mol)",
            xaxis_title="Energy Term",
            height=height,
            showlegend=False,
            template="plotly_white",
        )
        return fig

    # ── Energy Pie Chart ─────────────────────────────────────────
    @staticmethod
    def energy_pie(analysis: FullEnergyAnalysis, height: int = 450) -> go.Figure:
        """Pie chart of absolute energy contributions.

        Parameters
        ----------
        analysis : FullEnergyAnalysis
            Energy analysis result.
        height : int
            Figure height in pixels.

        Returns
        -------
        go.Figure
        """
        decomp = analysis.decomposition
        terms = [k for k in decomp if k != "Total"]
        values = [abs(decomp[k]) for k in terms]
        if sum(values) < 1e-12:
            values = [1.0] * len(terms)

        fig = go.Figure(go.Pie(
            labels=terms,
            values=values,
            marker_colors=BAR_COLORS[:len(terms)],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.4f} kcal/mol<extra></extra>",
        ))
        fig.update_layout(
            title=f"Energy Composition — {analysis.molecule_name}",
            height=height,
            template="plotly_white",
        )
        return fig

    # ── Per-Bond Energy Bar ──────────────────────────────────────
    @staticmethod
    def per_bond_energy(analysis: FullEnergyAnalysis, height: int = 500) -> go.Figure:
        """Bar chart of individual bond energies.

        Parameters
        ----------
        analysis : FullEnergyAnalysis
            Energy analysis result.
        height : int
            Figure height in pixels.

        Returns
        -------
        go.Figure
        """
        if not analysis.energy_result or not analysis.energy_result.bond_energies:
            fig = go.Figure()
            fig.add_annotation(text="No bond data available", x=0.5, y=0.5,
                               showarrow=False, font_size=16)
            fig.update_layout(height=height)
            return fig

        energies = analysis.energy_result.bond_energies
        labels = [f"Bond {i}" for i in range(len(energies))]
        max_e = max(energies) if max(energies) > 1e-12 else 1.0
        colors = [f"rgb({int(255 * e / max_e)}, {int(50 + 150 * (1 - e / max_e))}, {int(255 * (1 - e / max_e))})"
                  for e in energies]

        fig = go.Figure(go.Bar(
            x=labels, y=energies,
            marker_color=colors,
            text=[f"{e:.3f}" for e in energies],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Per-Bond Energies — {analysis.molecule_name}",
            yaxis_title="Energy (kcal/mol)",
            xaxis_title="Bond",
            height=height,
            template="plotly_white",
        )
        return fig

    # ── Stress Heatmap ───────────────────────────────────────────
    @staticmethod
    def stress_heatmap(stress_analysis: StressAnalysis, height: int = 500) -> go.Figure:
        """Heatmap of per-bond stress values.

        Parameters
        ----------
        stress_analysis : StressAnalysis
            Stress analysis result.
        height : int
            Figure height in pixels.

        Returns
        -------
        go.Figure
        """
        sr = stress_analysis.stress_result
        if not sr or not sr.bond_stresses:
            fig = go.Figure()
            fig.add_annotation(text="No stress data", x=0.5, y=0.5,
                               showarrow=False, font_size=16)
            fig.update_layout(height=height)
            return fig

        stresses = sr.bond_stresses
        labels = stress_analysis.bond_labels or [f"Bond {i}" for i in range(len(stresses))]
        n = len(stresses)
        # Reshape into a 2D matrix for heatmap
        cols = max(1, int(np.ceil(np.sqrt(n))))
        rows = max(1, int(np.ceil(n / cols)))
        matrix = np.full((rows, cols), np.nan)
        label_matrix = [["" for _ in range(cols)] for _ in range(rows)]
        for idx, (s, lbl) in enumerate(zip(stresses, labels)):
            r, c = divmod(idx, cols)
            matrix[r][c] = s
            label_matrix[r][c] = lbl

        fig = go.Figure(go.Heatmap(
            z=matrix.tolist(),
            colorscale=STRESS_COLORSCALE,
            zmin=0, zmax=1,
            text=label_matrix,
            hovertemplate="Bond: %{text}<br>Stress: %{z:.4f}<extra></extra>",
            colorbar_title="Stress",
        ))
        fig.update_layout(
            title=f"Bond Stress Map — {stress_analysis.molecule_name}",
            height=height,
            template="plotly_white",
            xaxis_title="Bond Index (column)",
            yaxis_title="Bond Index (row)",
        )
        return fig

    # ── 3D Molecular Structure ───────────────────────────────────
    @staticmethod
    def molecular_structure_3d(
        ff: ForceField,
        stress_analysis: Optional[StressAnalysis] = None,
        height: int = 600,
    ) -> go.Figure:
        """3D scatter + line plot of the molecular structure colored by stress.

        Parameters
        ----------
        ff : ForceField
            Molecular system.
        stress_analysis : StressAnalysis, optional
            If provided, bonds are colored by stress.
        height : int
            Figure height in pixels.

        Returns
        -------
        go.Figure
        """
        fig = go.Figure()

        # Atom spheres
        xs = [a.x for a in ff.atoms]
        ys = [a.y for a in ff.atoms]
        zs = [a.z for a in ff.atoms]
        labels = get_atom_labels(ff)
        atom_colors = [ATOM_TYPE_COLORS.get(a.element, "#909090") for a in ff.atoms]
        sizes = [8 if a.element != "H" else 5 for a in ff.atoms]

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=sizes, color=atom_colors, line=dict(width=0.5, color="black")),
            text=labels,
            hovertemplate="Atom: %{text}<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
            name="Atoms",
        ))

        # Bonds as lines
        sr = stress_analysis.stress_result if stress_analysis else None
        for idx, bond in enumerate(ff.bonds):
            ai, aj = ff.atoms[bond.atom_i], ff.atoms[bond.atom_j]
            if sr and sr.bond_stresses:
                s = sr.bond_stresses[idx] if idx < len(sr.bond_stresses) else 0.0
                r_val = int(255 * s)
                b_val = int(255 * (1 - s))
                color = f"rgb({r_val}, 50, {b_val})"
            else:
                color = "#999999"
            fig.add_trace(go.Scatter3d(
                x=[ai.x, aj.x], y=[ai.y, aj.y], z=[ai.z, aj.z],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False,
                hoverinfo="skip",
            ))

        fig.update_layout(
            title=f"3D Structure — {ff.name}",
            height=height,
            template="plotly_white",
            scene=dict(
                xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)",
                aspectmode="data",
            ),
        )
        return fig

    # ── Bond Stretch Curve ───────────────────────────────────────
    @staticmethod
    def bond_stretch_curve(
        stretch_analysis: BondStretchAnalysis,
        height: int = 500,
    ) -> go.Figure:
        """Energy vs. displacement curve for bond stretching.

        Parameters
        ----------
        stretch_analysis : BondStretchAnalysis
            Bond stretch result.
        height : int
            Figure height in pixels.

        Returns
        -------
        go.Figure
        """
        sr = stretch_analysis.stretch_result
        if not sr or not sr.displacements:
            fig = go.Figure()
            fig.add_annotation(text="No stretch data", x=0.5, y=0.5,
                               showarrow=False, font_size=16)
            fig.update_layout(height=height)
            return fig

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sr.displacements, y=sr.energies,
            mode="lines+markers", name="Total Energy",
            line=dict(color=TOTAL_COLOR, width=3),
            marker=dict(size=4),
        ))
        fig.add_trace(go.Scatter(
            x=sr.displacements, y=sr.bond_energies_curve,
            mode="lines", name="Bond Energy",
            line=dict(color=BOND_COLOR, width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=sr.displacements, y=sr.vdw_energies_curve,
            mode="lines", name="VdW Energy",
            line=dict(color=VDW_COLOR, width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=sr.displacements, y=sr.elec_energies_curve,
            mode="lines", name="Electrostatic Energy",
            line=dict(color=ELEC_COLOR, width=2, dash="dashdot"),
        ))

        fig.update_layout(
            title=f"Break a Bond: {stretch_analysis.bond_label}",
            xaxis_title="Displacement beyond r₀ (Å)",
            yaxis_title="Energy (kcal/mol)",
            height=height,
            template="plotly_white",
            legend=dict(x=0.02, y=0.98),
        )
        return fig

    # ── Parameter Scan ───────────────────────────────────────────
    @staticmethod
    def parameter_scan(scan_analysis: ParameterScanAnalysis, height: int = 500) -> go.Figure:
        """Line plot of energy vs. parameter value.

        Parameters
        ----------
        scan_analysis : ParameterScanAnalysis
            Parameter scan result.
        height : int
            Figure height in pixels.

        Returns
        -------
        go.Figure
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=scan_analysis.param_values,
            y=scan_analysis.total_energies,
            mode="lines+markers", name="Total Energy",
            line=dict(color=TOTAL_COLOR, width=3),
        ))
        fig.add_trace(go.Scatter(
            x=scan_analysis.param_values,
            y=scan_analysis.bond_energies,
            mode="lines", name="Bond Energy",
            line=dict(color=BOND_COLOR, width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=scan_analysis.param_values,
            y=scan_analysis.vdw_energies,
            mode="lines", name="VdW Energy",
            line=dict(color=VDW_COLOR, width=2, dash="dot"),
        ))

        param_labels = {
            "bond_k": "k (kcal/(mol·Å²))",
            "bond_r0": "r₀ (Å)",
            "angle_k": "kθ (kcal/(mol·rad²))",
            "dielectric": "Dielectric constant εᵣ",
        }
        xlabel = param_labels.get(scan_analysis.param_name, scan_analysis.param_name)

        fig.update_layout(
            title=f"Parameter Sensitivity — {scan_analysis.param_name}",
            xaxis_title=xlabel,
            yaxis_title="Energy (kcal/mol)",
            height=height,
            template="plotly_white",
        )
        return fig

    # ── Perturbation Robustness ──────────────────────────────────
    @staticmethod
    def perturbation_robustness(
        pert_analysis: PerturbationAnalysis,
        height: int = 500,
    ) -> go.Figure:
        """Error-bar plot of energy vs. perturbation magnitude.

        Parameters
        ----------
        pert_analysis : PerturbationAnalysis
            Perturbation analysis result.
        height : int
            Figure height in pixels.

        Returns
        -------
        go.Figure
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pert_analysis.scales,
            y=pert_analysis.mean_energies,
            error_y=dict(type="data", array=pert_analysis.std_energies, visible=True),
            mode="lines+markers",
            name="Mean Energy",
            line=dict(color=TOTAL_COLOR, width=3),
            marker=dict(size=8),
        ))
        fig.add_hline(
            y=pert_analysis.baseline_energy,
            line_dash="dash", line_color="gray",
            annotation_text="Baseline",
        )
        fig.update_layout(
            title=f"Perturbation Robustness — {pert_analysis.molecule_name}",
            xaxis_title="Perturbation σ (Å)",
            yaxis_title="Energy (kcal/mol)",
            height=height,
            template="plotly_white",
        )
        return fig

    # ── Molecule Comparison ──────────────────────────────────────
    @staticmethod
    def molecule_comparison(
        analyses: Dict[str, FullEnergyAnalysis],
        height: int = 550,
    ) -> go.Figure:
        """Grouped bar chart comparing multiple molecules.

        Parameters
        ----------
        analyses : dict
            Mapping molecule name → FullEnergyAnalysis.
        height : int
            Figure height in pixels.

        Returns
        -------
        go.Figure
        """
        fig = go.Figure()
        names = list(analyses.keys())
        for idx, term in enumerate(TERM_NAMES):
            vals = [analyses[n].decomposition.get(term, 0.0) for n in names]
            fig.add_trace(go.Bar(
                x=names, y=vals, name=term,
                marker_color=BAR_COLORS[idx],
            ))
        fig.update_layout(
            barmode="group",
            title="Molecule Energy Comparison",
            yaxis_title="Energy (kcal/mol)",
            height=height,
            template="plotly_white",
        )
        return fig

    # ── Bonded vs Non-bonded ─────────────────────────────────────
    @staticmethod
    def bonded_vs_nonbonded(analysis: FullEnergyAnalysis, height: int = 400) -> go.Figure:
        """Side-by-side bar of bonded vs. non-bonded totals.

        Parameters
        ----------
        analysis : FullEnergyAnalysis
            Energy analysis result.
        height : int
            Figure height in pixels.

        Returns
        -------
        go.Figure
        """
        fig = go.Figure(go.Bar(
            x=["Bonded", "Non-bonded"],
            y=[analysis.bonded_total, analysis.nonbonded_total],
            marker_color=[BOND_COLOR, VDW_COLOR],
            text=[f"{analysis.bonded_total:.2f}", f"{analysis.nonbonded_total:.2f}"],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Bonded vs. Non-bonded — {analysis.molecule_name}",
            yaxis_title="Energy (kcal/mol)",
            height=height,
            template="plotly_white",
        )
        return fig


# ═══════════════════════════════════════════════════════════════════════
# Matplotlib Renderer
# ═══════════════════════════════════════════════════════════════════════

class MatplotlibRenderer:
    """Publication-quality Matplotlib figures for CLI and PDF export."""

    # ── Energy Decomposition Bar ─────────────────────────────────
    @staticmethod
    def energy_decomposition(analysis: FullEnergyAnalysis) -> plt.Figure:
        """Bar chart of the five energy terms.

        Parameters
        ----------
        analysis : FullEnergyAnalysis
            Energy analysis result.

        Returns
        -------
        plt.Figure
        """
        decomp = analysis.decomposition
        terms = [k for k in decomp if k != "Total"]
        values = [decomp[k] for k in terms]
        colors = BAR_COLORS[:len(terms)]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(terms, values, color=colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Energy (kcal/mol)")
        ax.set_title(f"Energy Decomposition — {analysis.molecule_name}")
        ax.axhline(0, color="black", linewidth=0.5)
        fig.tight_layout()
        return fig

    # ── Energy Pie Chart ─────────────────────────────────────────
    @staticmethod
    def energy_pie(analysis: FullEnergyAnalysis) -> plt.Figure:
        """Pie chart of absolute energy contributions.

        Parameters
        ----------
        analysis : FullEnergyAnalysis
            Energy analysis result.

        Returns
        -------
        plt.Figure
        """
        decomp = analysis.decomposition
        terms = [k for k in decomp if k != "Total"]
        values = [abs(decomp[k]) for k in terms]
        if sum(values) < 1e-12:
            values = [1.0] * len(terms)
        colors = BAR_COLORS[:len(terms)]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(values, labels=terms, colors=colors, autopct="%1.1f%%",
               startangle=90)
        ax.set_title(f"Energy Composition — {analysis.molecule_name}")
        fig.tight_layout()
        return fig

    # ── Per-Bond Energy Bar ──────────────────────────────────────
    @staticmethod
    def per_bond_energy(analysis: FullEnergyAnalysis) -> plt.Figure:
        """Bar chart of individual bond energies.

        Parameters
        ----------
        analysis : FullEnergyAnalysis
            Energy analysis result.

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        if not analysis.energy_result or not analysis.energy_result.bond_energies:
            ax.text(0.5, 0.5, "No bond data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            fig.tight_layout()
            return fig

        energies = analysis.energy_result.bond_energies
        x = range(len(energies))
        max_e = max(energies) if max(energies) > 1e-12 else 1.0
        norm = Normalize(vmin=0, vmax=max_e)
        cmap = cm.get_cmap("RdYlBu_r")
        colors = [cmap(norm(e)) for e in energies]

        ax.bar(x, energies, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Bond Index")
        ax.set_ylabel("Energy (kcal/mol)")
        ax.set_title(f"Per-Bond Energies — {analysis.molecule_name}")
        fig.tight_layout()
        return fig

    # ── Stress Heatmap ───────────────────────────────────────────
    @staticmethod
    def stress_heatmap(stress_analysis: StressAnalysis) -> plt.Figure:
        """Heatmap of per-bond stress values.

        Parameters
        ----------
        stress_analysis : StressAnalysis
            Stress analysis result.

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        sr = stress_analysis.stress_result
        if not sr or not sr.bond_stresses:
            ax.text(0.5, 0.5, "No stress data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            fig.tight_layout()
            return fig

        stresses = sr.bond_stresses
        n = len(stresses)
        cols = max(1, int(np.ceil(np.sqrt(n))))
        rows = max(1, int(np.ceil(n / cols)))
        matrix = np.full((rows, cols), np.nan)
        for idx, s in enumerate(stresses):
            r, c = divmod(idx, cols)
            matrix[r][c] = s

        im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Stress (normalized)")
        ax.set_xlabel("Bond Index (column)")
        ax.set_ylabel("Bond Index (row)")
        ax.set_title(f"Bond Stress Map — {stress_analysis.molecule_name}")
        fig.tight_layout()
        return fig

    # ── Bond Stretch Curve ───────────────────────────────────────
    @staticmethod
    def bond_stretch_curve(stretch_analysis: BondStretchAnalysis) -> plt.Figure:
        """Energy vs. displacement curve.

        Parameters
        ----------
        stretch_analysis : BondStretchAnalysis
            Bond stretch result.

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        sr = stretch_analysis.stretch_result
        if not sr or not sr.displacements:
            ax.text(0.5, 0.5, "No stretch data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            fig.tight_layout()
            return fig

        ax.plot(sr.displacements, sr.energies, "-o", color=TOTAL_COLOR,
                markersize=3, linewidth=2, label="Total")
        ax.plot(sr.displacements, sr.bond_energies_curve, "--",
                color=BOND_COLOR, linewidth=1.5, label="Bond")
        ax.plot(sr.displacements, sr.vdw_energies_curve, ":",
                color=VDW_COLOR, linewidth=1.5, label="VdW")
        ax.plot(sr.displacements, sr.elec_energies_curve, "-.",
                color=ELEC_COLOR, linewidth=1.5, label="Electrostatic")

        ax.set_xlabel("Displacement beyond r₀ (Å)")
        ax.set_ylabel("Energy (kcal/mol)")
        ax.set_title(f"Break a Bond: {stretch_analysis.bond_label}")
        ax.legend()
        fig.tight_layout()
        return fig

    # ── Parameter Scan ───────────────────────────────────────────
    @staticmethod
    def parameter_scan(scan_analysis: ParameterScanAnalysis) -> plt.Figure:
        """Line plot of energy vs. parameter value.

        Parameters
        ----------
        scan_analysis : ParameterScanAnalysis
            Parameter scan result.

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(scan_analysis.param_values, scan_analysis.total_energies,
                "-o", color=TOTAL_COLOR, markersize=4, linewidth=2, label="Total")
        ax.plot(scan_analysis.param_values, scan_analysis.bond_energies,
                "--", color=BOND_COLOR, linewidth=1.5, label="Bond")
        ax.plot(scan_analysis.param_values, scan_analysis.vdw_energies,
                ":", color=VDW_COLOR, linewidth=1.5, label="VdW")

        param_labels = {
            "bond_k": "k (kcal/(mol·Å²))",
            "bond_r0": "r₀ (Å)",
            "angle_k": "kθ (kcal/(mol·rad²))",
            "dielectric": "Dielectric constant εᵣ",
        }
        xlabel = param_labels.get(scan_analysis.param_name, scan_analysis.param_name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Energy (kcal/mol)")
        ax.set_title(f"Parameter Sensitivity — {scan_analysis.param_name}")
        ax.legend()
        fig.tight_layout()
        return fig

    # ── Perturbation Robustness ──────────────────────────────────
    @staticmethod
    def perturbation_robustness(pert_analysis: PerturbationAnalysis) -> plt.Figure:
        """Error-bar plot of energy vs. perturbation magnitude.

        Parameters
        ----------
        pert_analysis : PerturbationAnalysis
            Perturbation analysis result.

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            pert_analysis.scales, pert_analysis.mean_energies,
            yerr=pert_analysis.std_energies,
            fmt="-o", color=TOTAL_COLOR, markersize=6, linewidth=2,
            capsize=4, label="Mean Energy",
        )
        ax.axhline(pert_analysis.baseline_energy, color="gray",
                    linestyle="--", linewidth=1, label="Baseline")
        ax.set_xlabel("Perturbation σ (Å)")
        ax.set_ylabel("Energy (kcal/mol)")
        ax.set_title(f"Perturbation Robustness — {pert_analysis.molecule_name}")
        ax.legend()
        fig.tight_layout()
        return fig

    # ── Molecule Comparison ──────────────────────────────────────
    @staticmethod
    def molecule_comparison(analyses: Dict[str, FullEnergyAnalysis]) -> plt.Figure:
        """Grouped bar chart comparing multiple molecules.

        Parameters
        ----------
        analyses : dict
            Mapping molecule name → FullEnergyAnalysis.

        Returns
        -------
        plt.Figure
        """
        names = list(analyses.keys())
        n_terms = len(TERM_NAMES)
        x = np.arange(len(names))
        width = 0.15

        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, term in enumerate(TERM_NAMES):
            vals = [analyses[n].decomposition.get(term, 0.0) for n in names]
            ax.bar(x + idx * width - width * 2, vals, width,
                   label=term, color=BAR_COLORS[idx], edgecolor="black", linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("Energy (kcal/mol)")
        ax.set_title("Molecule Energy Comparison")
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        fig.tight_layout()
        return fig

    # ── Bonded vs Non-bonded ─────────────────────────────────────
    @staticmethod
    def bonded_vs_nonbonded(analysis: FullEnergyAnalysis) -> plt.Figure:
        """Side-by-side bar of bonded vs. non-bonded totals.

        Parameters
        ----------
        analysis : FullEnergyAnalysis
            Energy analysis result.

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(["Bonded", "Non-bonded"],
               [analysis.bonded_total, analysis.nonbonded_total],
               color=[BOND_COLOR, VDW_COLOR], edgecolor="black")
        ax.set_ylabel("Energy (kcal/mol)")
        ax.set_title(f"Bonded vs. Non-bonded — {analysis.molecule_name}")
        ax.axhline(0, color="black", linewidth=0.5)
        fig.tight_layout()
        return fig
