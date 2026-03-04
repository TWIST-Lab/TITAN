import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Consistent plotting style
sns.set_style("whitegrid")
import os
import time
from SimulationEnvironment import SimulationEnvironment
from bayesian_optimizer import (
    optimize_uav_positions,
    optimize_uav_positions_uma,
    optimize_uav_positions_aoi,
)
from Utilities import plot_timeseries_from_df, plot_allocation_from_df


CANONICAL_GNB_POSITIONS = np.array([
    [608.0, 73.0, 150.0],
    [-140.0, -300.0, 110.0],
    [-220.0, 220.0, 40.0],
], dtype=float)


def _get_failed_gnb_aoi_centers(env, tol=1e-6):
    """Return failed gNB coordinates as AOI centers for failure analysis."""
    active_positions = np.array([
        np.array(tx.position).reshape(-1)[:3] for tx in env.base_stations
    ], dtype=float) if env.base_stations else np.empty((0, 3), dtype=float)

    failed = []
    for gnb in CANONICAL_GNB_POSITIONS:
        if active_positions.size == 0:
            failed.append(gnb)
            continue
        deltas = np.linalg.norm(active_positions - gnb[None, :], axis=1)
        if not np.any(deltas < tol):
            failed.append(gnb)

    if not failed:
        return np.array([[0.0, 0.0, 0.0]], dtype=float)

    return np.array(failed, dtype=float)

def evaluate_env(env, placement: str, uav_count: int, use_ue_offset: bool = False) -> dict:
    """
    Gather coverage, SINR and sum-rate statistics for the current state of `env`
    and return them in a uniform dictionary format understood by `main()`.
    All UAVs that were disabled (power_dbm < 0) are pushed below ground before
    link-level evaluation to remove any residual interference.
    """
    total_uavs = len(env.uavs)
    inactive_uavs = total_uavs - uav_count

    # Banish silent UAVs to “hell”
    for idx in range(inactive_uavs):
        env.uavs[idx].position[2] = -500

    # Link budget before teleporting inactive nodes
    tx_pos = [np.array(tx.position).flatten().tolist() for tx in env.txs]
    serving, sinr = env.check_sinr(tx_pos)

    # Full-stack evaluation
    env.build_rx_tx_association(serving)
    df_slot, df_alloc = env.evaluate_links(num_steps=100)

    if False:
        for bs_id, df_bs in df_slot.groupby('bs'):
            plot_timeseries_from_df(df_bs, title=f"Time-series metrics {bs_id}")

        for bs_id, df_bs in df_alloc.groupby('bs'):
            plot_allocation_from_df(df_bs,
                                    num_symbols=env.num_ofdm_symbols,
                                    num_subcarriers=env.num_subcarriers,
                                    title=f'OFDMA allocation – BS {bs_id}')
        plt.show()

    covered = serving != -1
    coverage_ratio = np.sum(covered) / len(serving)
    mean_sinr_db = np.mean(10 * np.log10(sinr[covered])) if np.any(covered) else float("-inf")

    if df_slot.empty or "bits" not in df_slot.columns:
        sum_rate = 0.0
        fairness_index = 0.0
    else:
        slot_count = df_slot["slot"].nunique()
        bits_total = df_slot["bits"].sum()
        slot_duration = env.num_ofdm_symbols / env.subcarrier_spacing
        sum_rate = bits_total * coverage_ratio / slot_count / slot_duration

        # Jain fairness across UT bitrates -------------------------------
        rate_per_ut = (
            df_slot.groupby("ue")["bits"].sum() / (slot_count * slot_duration)
        )
        rate_per_ut = rate_per_ut.reindex(range(len(env.ues)), fill_value=0.0)
        rates = rate_per_ut.to_numpy()
        numer = rates.sum() ** 2
        denom = len(rates) * np.sum(rates ** 2)
        fairness_index = float(numer / denom) if denom > 0 else 0.0

    bandwidth_hz = env.subcarrier_spacing * env.num_subcarriers

    spectral_efficiency = sum_rate / uav_count / bandwidth_hz 

    return {
        "placement": placement,
        "uav_count": uav_count,
        "coverage_ratio": coverage_ratio,
        "mean_sinr_db": mean_sinr_db,
        "sum_rate": sum_rate,
        "spectral_efficiency": spectral_efficiency,
        "fairness_index": fairness_index,
        "positions": [env.uavs[i].position for i in range(inactive_uavs, total_uavs)],
        "df_slot": df_slot,
        "df_alloc": df_alloc,
    }


def run_random_placement(env, uav_count, xy_extent=500, z_min=50, z_max=1000, seed=None, use_ue_offset=False):

    if seed is not None:
        np.random.seed(seed)

    total_uavs = len(env.uavs)
    inactive_uavs = total_uavs - uav_count

    # Randomize positions for all UAVs first
    for i, tx in enumerate(env.uavs):
        x = np.random.uniform(-xy_extent, xy_extent)
        y = np.random.uniform(-xy_extent, xy_extent)
        z = np.random.uniform(z_min, z_max)

        tx.position = np.array([x, y, z])
        tx.look_at(np.array([x, y, 0]))  # Point toward ground
        tx.power_dbm = 23  # Reset power
        env.bs_power_dbm = 23

    # Check SINR before moving inactive UAVs to "hell"
    tx_pos = [np.array(tx.position).flatten().tolist() for tx in env.txs]
    serving, sinr = env.check_sinr(tx_pos, use_ue_offset=use_ue_offset)

    return evaluate_env(env, "random", uav_count, use_ue_offset=use_ue_offset)

def run_leo(env, use_ue_offset=False):

    power_dbm = 43 + 40 - 5
    elevation = 10              # degrees
    z = 600000                 # meters
    env.bs_power_dbm = power_dbm

    uav_count = len(env.uavs)
    azimuths = np.linspace(0, 2 * np.pi, uav_count, endpoint=False)

    rng = np.random.default_rng()
    weights = np.array([0.6011, 0.3034, 0.0954])
    weights[2] = 1.0 - weights[1] - weights[0]
    edges = np.array([60.0, 70.0, 80.0, 90.0])
    bands = rng.choice(3, size=uav_count, p=weights)
    elevations = edges[bands] + 10.0 * rng.random(uav_count)

    for sat, azimuth, elevation in zip(env.uavs[-uav_count:], azimuths, elevations):
        radius = z / np.tan(np.deg2rad(elevation))
        sat.position = np.array([
            radius * np.cos(azimuth),
            radius * np.sin(azimuth),
            z,
        ])
        sat.look_at([0, 0, 0])
        sat.power_dbm = power_dbm


    return evaluate_env(env, "leo", uav_count=uav_count, use_ue_offset=use_ue_offset)

def run_bayesian_stochastic_placement(
    env,
    uav_count,
    xy_extent=500,
    z_min=50,
    z_max=1000,
    n_trials=150,
    use_ue_offset=False,
    eval_offset=None,
):
    """
    Bayesian optimisation using the TR 38.901 UMa backend, evaluation with ray-tracing.
    """
    total_uavs   = len(env.uavs)
    inactive_uavs= total_uavs - uav_count

    for tx in env.uavs:
        tx.power_dbm = 23
        env.bs_power_dbm     = 23

    for s in range(inactive_uavs):
        env.uavs[s].power_dbm = -200

    best_xyz, best_obj = optimize_uav_positions_uma(
        env,
        n_trials      = n_trials,
        xy_extent     = xy_extent,
        z_min         = z_min,
        z_max         = z_max,
        use_ue_offset = use_ue_offset)

    for i, tx in enumerate(env.uavs):
        tx.position = best_xyz[i]

    # Do the association with RT
    tx_pos = [np.array(tx.position).flatten().tolist() for tx in env.txs]
    serving, sinr = env.check_sinr(tx_pos, use_ue_offset=use_ue_offset)

    if eval_offset is None:
        eval_offset = use_ue_offset

    return evaluate_env(env, "bayesian_stochastic", uav_count, use_ue_offset=eval_offset)

def run_bayesian_placement(
    env,
    uav_count,
    xy_extent=500,
    z_min=50,
    z_max=1000,
    n_trials=150,
    use_ue_offset=False,
    eval_offset=None,
):
    total_uavs = len(env.uavs)
    inactive_uavs = total_uavs - uav_count

    # Enable all UAVs initially for optimization
    for tx in env.uavs:
        tx.power_dbm = 23  # Reset power
        env.bs_power_dbm = 23

    # Disable the first inactive_uavs UAVs
    for s in range(inactive_uavs):
        env.uavs[s].power_dbm = -200  # Very low power

    # Run Bayesian optimization
    best_xyz, best_obj = optimize_uav_positions(env,
                                               n_trials=n_trials,
                                               xy_extent=xy_extent,
                                               z_min=z_min,
                                               z_max=z_max,
                                               use_ue_offset=use_ue_offset)

    # Apply optimized positions to all UAVs
    for i, tx in enumerate(env.uavs):
        tx.position = best_xyz[i]
        tx.look_at(np.array([best_xyz[i][0], best_xyz[i][1], 0]))  # Point toward ground


    # Check SINR before moving inactive UAVs to "hell"
    tx_pos = [np.array(tx.position).flatten().tolist() for tx in env.txs]
    serving, sinr = env.check_sinr(tx_pos, use_ue_offset=use_ue_offset)

    if eval_offset is None:
        eval_offset = use_ue_offset

    return evaluate_env(env, "bayesian", uav_count, use_ue_offset=eval_offset)

def run_bayesian_aoi_placement(
    env,
    uav_count,
    xy_extent=500,
    z_min=50,
    z_max=1000,
    n_trials=150,
    use_ue_offset=False,
    eval_offset=None,
):
    """Bayesian optimisation with AOI-aware loss.

    In failure analysis, each failed gNB location is used as an AOI center.
    """
    total_uavs = len(env.uavs)
    inactive_uavs = total_uavs - uav_count

    for tx in env.uavs:
        tx.power_dbm = 23
        env.bs_power_dbm = 23

    for s in range(inactive_uavs):
        env.uavs[s].power_dbm = -200

    failed_aoi_centers = _get_failed_gnb_aoi_centers(env)

    best_xyz, best_obj = optimize_uav_positions_aoi(
        env,
        n_trials=n_trials,
        xy_extent=xy_extent,
        z_min=z_min,
        z_max=z_max,
        alpha=0.01,
        beta=1.0,
        gamma=0.8,
        d_min=400.0,
        aoi_centers=failed_aoi_centers,
        aoi_radii=np.full(failed_aoi_centers.shape[0], 800.0, dtype=float),
        use_ue_offset=use_ue_offset,
    )

    for i, tx in enumerate(env.uavs):
        tx.position = best_xyz[i]
        tx.look_at(np.array([best_xyz[i][0], best_xyz[i][1], 0]))

    tx_pos = [np.array(tx.position).flatten().tolist() for tx in env.txs]
    serving, sinr = env.check_sinr(tx_pos, use_ue_offset=use_ue_offset)

    if eval_offset is None:
        eval_offset = use_ue_offset

    return evaluate_env(env, "bayesian_aoi", uav_count, use_ue_offset=eval_offset)


def main():
    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Initialize environment
    env = SimulationEnvironment(seed=44)

    # Initialize solvers
    env.init_solvers()

    # Add a dummy transmitter to sample UE positions
    #env.add_tx("dummy", [0, 0, 1000], 23)

    env.add_base_station("tx1", [608, 73, 150], orientation=[0, 0, 0], color=(0, 0, 1))
    env.add_base_station("tx2", [-140, -300, 110], orientation=[0, 0, 0], color=(0.7, 0, 0.7))
    env.add_base_station("tx3", [-220, 220, 40], orientation=[0, 0, 0], color=(1, 0, 0))

    print("Deploying UEs...")
    env.deploy_ues(100, location_error=0.0)

    #env.scene.remove("dummy")
    env.scene.remove("tx1")
    env.scene.remove("tx2")
    env.scene.remove("tx3")
    env.txs.clear()
    env.uavs.clear()
    env.base_stations.clear()
    env.bs_positions.clear()
    env.bs_active.clear()

    env.add_base_station("tx1", [608, 73, 150], orientation=[0, 0, 0], color=(0, 0, 1))
    env.add_base_station("tx2", [-140, -300, 110], orientation=[0, 0, 0], color=(0.7, 0, 0.7))
    env.add_base_station("tx3", [-220, 220, 40], orientation=[0, 0, 0], color=(1, 0, 0))

    # Add all UAVs that will be used
    max_uavs = 3
    for i in range(max_uavs):
        env.add_tx(f"uav_{i+1}", [0, 0, 500], 23)

    # Number of Monte-Carlo trials
    n_trials = 1

    # Store results from all trials
    all_results = []

    # Run simulations for different UAV counts and placement methods
    for trial in range(n_trials):
        print(f"\n=== Trial {trial+1}/{n_trials} ===")
        for uav_count in range(1, max_uavs+1):
            if uav_count < 2:
                leo_results = run_leo(env, use_ue_offset=True)
                leo_results["trial"] = trial
                all_results.append(leo_results)

                env.call_rm_solver()
                env.render_paths()
                plt.savefig(f"results/leo_placement.png")
                plt.close()

            print(f"  Running simulations for {uav_count} UAV(s)...")

            # Random placement
            print("    Random placement...")
            random_results = run_random_placement(
                env, uav_count, seed=uav_count*100 + trial,
                use_ue_offset=True)
            random_results["trial"] = trial
            all_results.append(random_results)

            inactive_uavs = max_uavs - uav_count

            # Save visualization
            env.call_rm_solver()
            env.render_scene(metric="sinr")
            plt.savefig(f"results/random_placement_{uav_count}_uavs_trial{trial}.png")
            plt.close()

            # Bayesian UMa placement
            print("     Bayesian Stochastic placement")
            bayes_uma = run_bayesian_stochastic_placement(
                env, uav_count,
                xy_extent=500, z_min=50, z_max=1000,
                n_trials=100,
                use_ue_offset=True)
            bayes_uma["trial"] = trial
            all_results.append(bayes_uma)

            env.call_rm_solver()
            env.render_scene(metric="sinr")
            plt.savefig(f"results/bayesian_uma_{uav_count}uav_trial{trial}.png")
            plt.close()

            # Bayesian placement
            print("    Bayesian placement...")
            bayesian_results = run_bayesian_placement(
                env, uav_count, n_trials=100,
                use_ue_offset=True)
            bayesian_results["trial"] = trial
            all_results.append(bayesian_results)

            # Save visualization
            env.call_rm_solver()
            env.render_scene(metric="sinr")
            plt.savefig(f"results/bayesian_placement_{uav_count}_uavs_trial{trial}.png")
            plt.close()

            # Plot and save allocation results for both methods
            for result, method in [
                    (random_results, "random"),
                    (bayesian_results, "bayesian"),
                    (bayes_uma, "bayesian_uma"),
                    (leo_results, "leo")]:
                for bs_id, df_bs in result["df_alloc"].groupby('bs'):
                    if bs_id >= inactive_uavs:  # Only plot for active UAVs
                        plot_allocation_from_df(
                            df_bs,
                            num_symbols=env.num_ofdm_symbols,
                            num_subcarriers=env.num_subcarriers,
                            title=(f'OFDMA allocation - {method.capitalize()} - '
                                   f'{uav_count} UAVs - BS {bs_id}'))
                        plt.savefig(
                            f"results/{method}_allocation_{uav_count}_uavs_bs{bs_id}_trial{trial}.png")
                        plt.close()

    # Create summary dataframe
    results_df = pd.DataFrame(all_results)

    # Plot summary bar plots with seaborn using 95% confidence intervals
    metrics = [
        ("coverage_ratio", "Coverage ratio"),
        ("mean_sinr_db", "Mean SINR (dB)"),
        ("sum_rate", "Sum rate (bits/s)"),
        ("fairness_index", "Jain fairness")
    ]

    plt.figure(figsize=(10, 16))
    for idx, (metric, label) in enumerate(metrics, start=1):
        ax = plt.subplot(len(metrics), 1, idx)

        sns.barplot(
            data=results_df[results_df["placement"] != "leo"],
            x="uav_count",
            y=metric,
            hue="placement",
            errorbar=("ci", 95),
            ax=ax,
        )

        # Add LEO reference as horizontal line with confidence interval
        leo_df = results_df[results_df["placement"] == "leo"]
        if not leo_df.empty:
            mean_val = leo_df[metric].mean()
            ci = 1.96 * leo_df[metric].std(ddof=1) / np.sqrt(len(leo_df))
            xmin, xmax = ax.get_xlim()
            ax.hlines(mean_val, xmin, xmax, linestyle="--", color="k",
                      label="leo" if idx == 1 else None)
            ax.fill_between([xmin, xmax],
                            mean_val - ci,
                            mean_val + ci,
                            color="k", alpha=0.15)

        ax.set_xlabel("Number of UAVs")
        ax.set_ylabel(label)
        if idx == 1:
            ax.legend()

    plt.tight_layout()
    plt.savefig("results/summary_barplot.png")
    plt.close()

if __name__ == "__main__":
    main()
