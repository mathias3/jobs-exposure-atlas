#!/usr/bin/env python3
"""Generate publication-style chart snapshots for the README."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

BG = "#f6f1e8"
PANEL = "#fffaf2"
GRID = "#d8ccbb"
TEXT = "#1c2a38"
MUTED = "#6b7a8d"
TEAL = "#187a78"
CYAN = "#2f8fb7"
GOLD = "#c58b2e"
CORAL = "#cd5b45"
RED = "#a63f32"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 20,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.45,
    }
)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "unified_exposure.csv")
    numeric_cols = [
        "num_jobs_2024",
        "median_pay_annual",
        "karpathy_normalized",
        "openai_beta",
        "anthropic_observed",
        "frey_osborne_prob",
        "theory_reality_gap",
        "llm_shift",
        "consensus_score",
        "disagreement",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["coverage"] = df[
        ["karpathy_normalized", "openai_beta", "anthropic_observed", "frey_osborne_prob"]
    ].notna().sum(axis=1)
    return df


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fmt_jobs(x: float) -> str:
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x / 1_000:.0f}K"
    return str(int(x))


def build_workforce_frontier(df: pd.DataFrame) -> None:
    rows = df[df["coverage"] >= 3].copy()
    rows = rows.dropna(subset=["num_jobs_2024", "consensus_score", "disagreement", "median_pay_annual"])
    rows["impact"] = rows["num_jobs_2024"] * rows["consensus_score"]
    pay_root = np.sqrt(rows["median_pay_annual"].clip(lower=30_000))
    if np.isclose(pay_root.min(), pay_root.max()):
        rows["bubble_area"] = 700.0
    else:
        rows["bubble_area"] = np.interp(pay_root, (pay_root.min(), pay_root.max()), (180, 1800))

    fig, ax = plt.subplots(figsize=(12.5, 8))
    norm = Normalize(rows["disagreement"].min(), rows["disagreement"].max())
    cmap = LinearSegmentedColormap.from_list("impact", ["#4f9b8b", "#f0b44f", "#c5553f"])

    scatter = ax.scatter(
        rows["num_jobs_2024"],
        rows["consensus_score"],
        s=rows["bubble_area"],
        c=rows["disagreement"],
        cmap=cmap,
        norm=norm,
        alpha=0.78,
        edgecolors="#fffaf2",
        linewidths=1.0,
    )

    labels = rows.nlargest(8, "impact")
    right_label_threshold = rows["num_jobs_2024"].quantile(0.82)
    for _, row in labels.iterrows():
        on_right_edge = row["num_jobs_2024"] >= right_label_threshold
        ax.annotate(
            row["title"],
            (row["num_jobs_2024"], row["consensus_score"]),
            xytext=(-8, 6) if on_right_edge else (7, 7),
            textcoords="offset points",
            ha="right" if on_right_edge else "left",
            fontsize=9,
            color=TEXT,
        )

    ax.axhline(0.6, color=MUTED, linestyle="--", linewidth=1.0)
    ax.axvline(1_000_000, color=MUTED, linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlim(4_000, rows["num_jobs_2024"].max() * 1.35)
    ax.set_ylim(0.08, 0.94)
    ax.set_xticks([10_000, 100_000, 1_000_000])
    ax.set_xticklabels(["10K", "100K", "1M"])
    ax.set_xlabel("Occupation employment (log scale)")
    ax.set_ylabel("Consensus exposure score")
    ax.set_title("Workforce Exposure Frontier", loc="left", pad=18)
    ax.text(
        0,
        1.03,
        "Large bubbles combine large workforce footprint with strong multi-source exposure. Color shows disagreement.",
        transform=ax.transAxes,
        fontsize=11,
        color=MUTED,
    )
    style_axes(ax)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02, shrink=0.92)
    cbar.outline.set_edgecolor(GRID)
    cbar.ax.tick_params(length=0, colors=MUTED)
    cbar.set_label("Disagreement", color=TEXT)

    save(fig, "workforce_frontier.png")


def build_theory_gap(df: pd.DataFrame) -> None:
    rows = (
        df.dropna(subset=["openai_beta", "anthropic_observed", "theory_reality_gap"])
        .nlargest(12, "theory_reality_gap")
        .sort_values("theory_reality_gap")
    )

    fig, ax = plt.subplots(figsize=(12.5, 8))
    y = np.arange(len(rows))

    ax.hlines(y, rows["anthropic_observed"], rows["openai_beta"], color="#d4c6b4", linewidth=4, zorder=1)
    ax.scatter(rows["anthropic_observed"], y, s=110, color=CYAN, zorder=2, label="Anthropic observed")
    ax.scatter(rows["openai_beta"], y, s=110, color=CORAL, zorder=3, label="OpenAI beta")

    for idx, (_, row) in enumerate(rows.iterrows()):
        ax.text(row["openai_beta"] + 0.012, idx, f"+{row['theory_reality_gap']:.2f}", va="center", fontsize=9, color=RED)

    ax.set_yticks(y)
    ax.set_yticklabels(rows["title"])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Exposure score")
    ax.set_title("Theory vs Deployment Gap Leaders", loc="left", pad=18)
    ax.text(
        0,
        1.03,
        "These occupations look highly exposed in theory but materially less deployed in Anthropic's observed usage data.",
        transform=ax.transAxes,
        fontsize=11,
        color=MUTED,
    )
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)

    save(fig, "theory_reality_gap.png")


def build_llm_shift(df: pd.DataFrame) -> None:
    rows = (
        df.dropna(subset=["karpathy_normalized", "frey_osborne_prob", "llm_shift"])
        .nlargest(12, "llm_shift")
        .sort_values("llm_shift")
    )

    fig, ax = plt.subplots(figsize=(12.5, 8))
    y = np.arange(len(rows))

    ax.hlines(y, rows["frey_osborne_prob"], rows["karpathy_normalized"], color="#d7cab8", linewidth=4, zorder=1)
    ax.scatter(rows["frey_osborne_prob"], y, s=110, color=TEAL, zorder=2, label="Frey & Osborne")
    ax.scatter(rows["karpathy_normalized"], y, s=110, color=CORAL, zorder=3, label="Karpathy")

    for idx, (_, row) in enumerate(rows.iterrows()):
        ax.text(row["karpathy_normalized"] + 0.012, idx, f"+{row['llm_shift']:.2f}", va="center", fontsize=9, color=RED)

    ax.set_yticks(y)
    ax.set_yticklabels(rows["title"])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Exposure score")
    ax.set_title("LLM Disruption Delta Occupations", loc="left", pad=18)
    ax.text(
        0,
        1.03,
        "Top occupations newly exposed in the LLM era relative to the pre-LLM automation baseline.",
        transform=ax.transAxes,
        fontsize=11,
        color=MUTED,
    )
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)

    save(fig, "llm_shift.png")


def build_category_heatmap(df: pd.DataFrame) -> None:
    source_cols = {
        "Karpathy": "karpathy_normalized",
        "OpenAI": "openai_beta",
        "Anthropic": "anthropic_observed",
        "Frey": "frey_osborne_prob",
    }

    records = []
    for category, vals in df.groupby("category"):
        row = {"category": category}
        weights = vals["num_jobs_2024"].fillna(0)
        for label, col in source_cols.items():
            mask = vals[col].notna() & weights.gt(0)
            if mask.any():
                row[label] = np.average(vals.loc[mask, col], weights=weights.loc[mask])
            else:
                row[label] = np.nan
        records.append(row)

    cat = pd.DataFrame(records).sort_values("Karpathy", ascending=False)
    values = cat[list(source_cols.keys())].to_numpy()

    fig_h = max(8.5, 0.42 * len(cat) + 2.2)
    fig, ax = plt.subplots(figsize=(9.6, fig_h))
    cmap = LinearSegmentedColormap.from_list("heat", ["#dceee7", "#f4c05a", "#c5553f"])
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(source_cols)))
    ax.set_xticklabels(list(source_cols.keys()))
    ax.set_yticks(np.arange(len(cat)))
    ax.set_yticklabels([c.replace("-", " ").title().replace(" And ", " & ") for c in cat["category"]])
    ax.set_title("Category-Level Confrontation Map", loc="left", pad=18)
    ax.text(
        0,
        1.02,
        "Employment-weighted category averages across all four source frameworks.",
        transform=ax.transAxes,
        fontsize=11,
        color=MUTED,
    )

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(
                j,
                i,
                "NA" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color=TEXT,
            )

    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color=BG, linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    style_axes(ax)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.outline.set_edgecolor(GRID)
    cbar.ax.tick_params(length=0, colors=MUTED)
    cbar.set_label("Weighted score", color=TEXT)

    save(fig, "category_confrontation.png")


def main() -> None:
    df = load_data()
    build_workforce_frontier(df)
    build_theory_gap(df)
    build_llm_shift(df)
    build_category_heatmap(df)
    print(f"Wrote chart snapshots to {OUT}")


if __name__ == "__main__":
    main()
