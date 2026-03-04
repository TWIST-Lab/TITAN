import numpy as np
import matplotlib.pyplot as plt
from SimulationEnvironment import SimulationEnvironment
from bayesian_optimizer import optimize_uav_positions_uma
from Utilities import plot_timeseries_from_df, plot_allocation_from_df


def main():
    env = SimulationEnvironment(seed=44)
    env.add_tx("dummy", [0, 0, 1000], 23)
    env.init_solvers()
    env.deploy_ues(50, location_error=0.0)
    env.scene.remove("dummy")
    env.txs.clear()
    env.uavs.clear()
    env.base_stations.clear()
    env.bs_positions.clear()
    env.bs_active.clear()

    env.add_base_station("tx1", [608, 73, 150], orientation=[0, 0, 0], color=(0, 0, 1))
    env.add_base_station("tx2", [-140, -300, 110], orientation=[0, 0, 0], color=(0.7, 0, 0.7))
    env.add_base_station("tx3", [-220, 220, 40], orientation=[0, 0, 0], color=(1, 0, 0))

    env.add_tx("uav_1", [0, 0, 500], 23)

    env.call_rm_solver()

    best_xyz, _ = optimize_uav_positions_uma(
        env,
        n_trials=10,
        xy_extent=500,
        z_min=50,
        z_max=1000,
        use_ue_offset=False,
    )

    env.uavs[0].position = best_xyz[0]

    env.call_path_solver(use_perturbed=False)
    bs_xyz = [np.array(tx.position).flatten().tolist() for tx in env.base_stations]
    uav_xyz = [p.tolist() for p in best_xyz]
    serving, sinr = env.check_sinr(bs_xyz + uav_xyz, use_ue_offset=False)
    env.call_rm_solver()
    env.render_scene(metric="sinr")
    env.render_paths()

    env.build_rx_tx_association(serving)

    df_slot, df_alloc = env.evaluate_links(num_steps=100)

    covered = serving != -1
    coverage_ratio = np.sum(covered) / len(serving)
    mean_sinr_db = np.mean(10 * np.log10(sinr[covered])) if np.any(covered) else float("-inf")

    print(f"Coverage ratio: {coverage_ratio:.2f}")
    print(f"Mean SINR (dB): {mean_sinr_db:.2f}")

    for bs_id, df_bs in df_alloc.groupby('bs'):
        plot_allocation_from_df(df_bs,
                                num_symbols=env.num_ofdm_symbols,
                                num_subcarriers=env.num_subcarriers,
                                title=f'OFDMA allocation – BS {bs_id}')

    for bs_id, df_bs in df_slot.groupby('bs'):
        plot_timeseries_from_df(df_bs, title=f'Time-series metrics BS {bs_id}')

    plt.show()


if __name__ == "__main__":
    main()
