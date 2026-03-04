#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import font_manager

font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
font_manager.fontManager.addfont(font_path)
sns.set_theme(
    style="whitegrid",
    font="Times New Roman",        # or "Times New Roman", "Comic Sans MS", etc.
    font_scale=2       # scales all font sizes by this factor
)
#sns.set_style("whitegrid")
#sns.set_context("notebook", font_scale=1.50)

def main(summary_file: str,
         location_tests: bool = False,
         failure_tests: bool = False) -> None:
    df = pd.read_csv(summary_file)
    df["sum_rate"] /= 10**6 # make Mbps

    # ─── Pre-processing depending on plot type ──────────────────────────────────
    if location_tests:
        df["location_error"] = df["location_error"].fillna(0.0)
        df = df.sort_values(by=["uav_count", "location_error"])
        x_col, hue_col = "uav_count", "location_error"
        placement_order = sorted(df[hue_col].unique())
        palette = sns.color_palette("viridis", n_colors=len(placement_order))
        legend_labels = None
        # Generate hatch patterns for location errors
        base_patterns = ["", "//", "\\\\", "xx", "--", "++", "||", "oo", "OO", "..", "**"]
        hatch_patterns = base_patterns[:len(placement_order)]

    elif failure_tests:
        # Treat random as "Terrestial only"
        df["placement"] = df["placement"].replace({"terrestial": "Terrestial Only"})
        df["placement"] = df["placement"].replace({"leo": "Traditional D2C"})
        df["placement"] = df["placement"].replace({"titan": "TITAN"})
        df["placement"] = df["placement"].replace({"bayesian_aoi": "SOTA [25]"})
        df = df.sort_values(by=["gnb_count", "placement"])
        x_col, hue_col = "gnb_count", "placement"
        palette = sns.color_palette("rocket", n_colors=df[hue_col].nunique())
        placement_order = ["Terrestial Only", "Traditional D2C", "SOTA [25]", "TITAN"]
        legend_labels = placement_order
        # Define hatch patterns for failure tests
        hatch_patterns = ["", "//", "\\\\", "xx"]

    else:
        x_col, hue_col = "uav_count", "placement"

        all_placements     = ["random", "bayesian_stochastic", "greedy", "bayesian_aoi",
                              "bayesian", "titan_low"]
        all_legend_labels  = ["Baseline-1", "Baseline-2", "Baseline-3", "SOTA [25]",
                              "TITAN", "TITAN (Low Fidelity)"]
        all_hatch_patterns = ["", "xx", "--", "//", "\\\\"]

        placement_order = [p for p in all_placements if p in df[hue_col].unique()]
        legend_labels   = [all_legend_labels[all_placements.index(p)]
                           for p in placement_order]
        hatch_patterns  = all_hatch_patterns[: len(placement_order)]

        palette = sns.color_palette("Set1", n_colors=len(placement_order))
        df = df.sort_values(by=x_col)

    metrics = [
        ("coverage_ratio", "Coverage ratio"),
        ("mean_sinr_db",   "Mean SINR (dB)"),
        ("sum_rate",       "Sum rate (Mbps)"),
        ("spectral_efficiency", "Spectral efficiency (bits/s/Hz)"),
        ("fairness_index", "Jain fairness"),
    ]

    # ─── Plot loop ──────────────────────────────────────────────────────────────
    for metric, ylabel in metrics:
        if not failure_tests and not location_tests:
            fig, ax = plt.subplots(figsize=(12, 9))
        else:
            fig, ax = plt.subplots(figsize=(12, 7))

        # dataframe that feeds the bars
        plot_df = df.copy()
        if (not location_tests and not failure_tests
                and "placement" in plot_df.columns):
            plot_df = plot_df[plot_df["placement"].isin(placement_order)]

        # pick up ALL 'leo' rows from the original df (not plot_df)
        d2c_df = df[df["placement"] == "leo"] \
                 if "placement" in df.columns and "leo" in df["placement"].values \
                 else pd.DataFrame()

        # draw the bars
        sns.barplot(
            data=plot_df,
            x=x_col,
            y=metric,
            hue=hue_col,
            hue_order=placement_order,
            errorbar=("ci", 95),
            palette=palette,
            ax=ax,
        )

        # add hatching to bars
        if hatch_patterns:
            for container, hatch in zip(ax.containers, hatch_patterns):
                for bar in container.patches:
                    bar.set_hatch(hatch)

        # D2C baseline
        if not d2c_df.empty and not failure_tests:  # skip in failure_tests to keep legend clean
            m  = d2c_df[metric].mean()
            se = d2c_df[metric].std(ddof=1) / len(d2c_df) ** 0.5
            ci = 1.96 * se
            xmin, xmax = ax.get_xlim()
            ax.hlines(m, xmin, xmax, linestyle="--", color="k")
            ax.fill_between([xmin, xmax], m - ci, m + ci, color="k", alpha=0.15)

        # labels & legend
        ax.set_xlabel("Number of UAVs")
        ax.set_ylabel(ylabel)

        if location_tests:
            patches = [
                mpatches.Patch(facecolor=palette[i],
                               hatch=hatch_patterns[i],
                               label=f"{placement_order[i]}")
                for i in range(len(placement_order))
            ]
            ax.legend(handles=patches, title="Location error (m)")
        elif failure_tests:
            ax.set_xlabel("Number of Active gNBs")
            patches = [
                mpatches.Patch(facecolor=palette[i],
                               hatch=hatch_patterns[i],
                               label=legend_labels[i])
                for i in range(len(placement_order))
            ]
            ax.legend(handles=patches, title="Method")
        else:
            patches = [
                mpatches.Patch(facecolor=palette[i],
                               hatch=hatch_patterns[i],
                               label=legend_labels[i])
                for i in range(len(placement_order))
            ]
            if not d2c_df.empty:
                patches.append(Line2D([0], [0],
                                      color="k", linestyle="--",
                                      label="Traditional D2C"))
                # Fairness için sol alta, diğerleri için default konum
            if metric == "fairness_index":
                ax.legend(handles=patches, title="Method")
                #ax.legend(handles=patches, title="Placement", loc='lower right')
            else:
                ax.legend(handles=patches, title="Placement")

        # save figure
        out_dir = os.path.dirname(summary_file)
        os.makedirs(out_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{metric}.pdf"), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("summary", help="CSV summary generated by run_case.py")
    p.add_argument(
        "--location-tests",
        action="store_true",
        help=("Plot location accuracy sweeps with UAV count on the x‑axis and "
              "location error as the hue"),
    )
    p.add_argument(
        "--failure-tests",
        action="store_true",
        help=("Plot base-station failure sweeps with UAV count on the x‑axis and "
              "number of active gNBs as the hue"),
    )
    args = p.parse_args()
    main(args.summary,
         location_tests=args.location_tests,
         failure_tests=args.failure_tests)
