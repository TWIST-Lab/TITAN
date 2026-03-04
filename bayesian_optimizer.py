# ----------------------------------------------------
# Optuna-based optimiser for UAV locations
# ----------------------------------------------------
import optuna
import numpy as np


def _terrain_relative_z(env, x, y, z_rel):
    """Convert relative altitude to absolute altitude over terrain."""
    gnd_z = env.get_ground_height(x, y) if hasattr(env, "get_ground_height") else None
    if gnd_z is None:
        return float(z_rel)
    return float(gnd_z + z_rel)

def optimize_uav_positions(env,
                           n_trials=100,
                           xy_extent=2000,
                           z_min=50,
                           z_max=1000,
                           timeout=None,
                           random_seed=0,
                           use_ue_offset=False):
    """
    Runs Optuna to find UAV coordinates that maximise
       objective = Jain * mean_SINR_linear * coverage_ratio.

    Parameters
    ----------
    env         : SimulationEnvironment  (already initialised, UEs deployed,
                                          UAV TXs added)
    n_trials    : int         – number of Optuna evaluations
    xy_extent   : float       – search box is [-xy_extent, +xy_extent] (metres)
    z_min,z_max : float       – altitude limits (metres)
    timeout     : float | None – wall-clock limit in seconds
    random_seed : int         – reproducibility
    use_ue_offset : bool – If ``True``, perturbed UE locations are used during
        optimisation

    Returns
    -------
    best_xyz : ndarray (n_tx,3) – optimum UAV positions
    best_val : float            – best objective value
    """
    n_tx = len(env.uavs)
    n_bs = len(env.base_stations)

    def as_xyz(trial):
        """Decode trial parameters into (n_tx,3) array."""
        xs = [trial.suggest_float(f"x{i}", -xy_extent, xy_extent) for i in range(n_tx)]
        ys = [trial.suggest_float(f"y{i}", -xy_extent, xy_extent) for i in range(n_tx)]
        z_rel = [trial.suggest_float(f"z{i}", z_min, z_max) for i in range(n_tx)]
        zs = [_terrain_relative_z(env, xs[i], ys[i], z_rel[i]) for i in range(n_tx)]
        return np.stack([xs, ys, zs], axis=1)

    def objective(trial):
        xyz = as_xyz(trial)
        bs_xyz = [np.array(tx.position).flatten().tolist() for tx in env.base_stations]
        tx_locs = bs_xyz + [xyz[i].tolist() for i in range(n_tx)]

        serving, sinr_lin = env.check_sinr(tx_locs, use_ue_offset=use_ue_offset)

        served_mask = serving != -1
        if not served_mask.any():
            return 0.0                # no coverage → worst score

        # --- sumrate 
        B = 1 # This value doesn't matter for optimization
        sumrate = np.sum(B * np.log2(1 + sinr_lin[served_mask]))
        # --- Jain fairness over UE counts ----------------------------------
        ue_counts = np.bincount(serving[served_mask], minlength=n_tx)
        numer = (ue_counts.sum())**2
        denom = len(ue_counts) * np.sum(ue_counts**2)
        jain = numer / denom if denom > 0 else 0.0

        # --- coverage ratio -------------------------------------------------
        coverage = served_mask.mean()   # ∈ [0,1]

        return sumrate * coverage * jain # study direction = "maximize"

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best_x = [study.best_params[f"x{i}"] for i in range(n_tx)]
    best_y = [study.best_params[f"y{i}"] for i in range(n_tx)]
    best_z_rel = [study.best_params[f"z{i}"] for i in range(n_tx)]
    best_z = [_terrain_relative_z(env, best_x[i], best_y[i], best_z_rel[i])
              for i in range(n_tx)]
    best_xyz = np.stack([best_x, best_y, best_z], axis=1)
    return best_xyz, study.best_value


# ----------------------------------------------------
# TR 38.901 UMA based optimisation (UAV-to-UE compliant)
# ----------------------------------------------------

def _uma_los_probability(d2d, h_bs, h_ut):
    """UMi street Canyon LoS probability (TR 38.901 Eq. 7.4-1), no height augmentation."""
    p2d = np.where(d2d <= 18.0,
                   1.0,
                   18.0 / d2d + np.exp(-d2d / 63.0) * (1 - 18.0 / d2d))
    return np.clip(p2d, 0.0, 1.0)


def _uma_pathloss(fc_hz, d3d, h_bs, h_ut, los=True):
    """
    UMa path loss compliant with TR 38.901:
     - LoS: two-slope with breakpoint d_BP (Eqs. 7.4.3‑1, 7.4.3‑2)
     - NLoS: Eq. 7.4.4‑1 with height gain, and PL = max(LoS, NLoS)
    """
    fc_ghz = fc_hz / 1e9
    c = 3e8
    # breakpoint distance (m)
    d_bp = (4 * h_bs * h_ut * fc_hz) / c

    if los:
        # two-slope formula
        pl = np.where(
            d3d < d_bp,
            28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz),
            28.0 + 40.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
                   - 9.0 * np.log10(d_bp**2 + (h_bs - h_ut)**2)
        )
    else:
        # compute the LoS PL for comparison
        pl_los = _uma_pathloss(fc_hz, d3d, h_bs, h_ut, los=True)
        # NLoS formula with UE height gain
        pl_nlos = (13.54 +
                   39.08 * np.log10(d3d) +
                   20.0 * np.log10(fc_ghz) -
                   0.6 * (h_ut - 1.5))
        # take the maximum
        pl = np.maximum(pl_los, pl_nlos)

    return pl


def _check_sinr_uma(env, uav_locs, sinr_cutoff_db=5, use_ue_offset=False):
    """Estimate SINR using TR 38.901 UMa model.

    ``uav_locs`` contains positions for UAVs only. Base stations are
    automatically appended using their current coordinates in ``env``.
    """
    if use_ue_offset and env.ues_pert:
        ue_pos = np.array(env.ues_pert, dtype=float)
    else:
        ue_pos = np.array([np.array(rx.position).reshape(-1)[:3]
                           for rx in env.ues], dtype=float)

    bs_xyz = [np.array(tx.position).reshape(-1)[:3] for tx in env.base_stations]
    uav_xyz = np.array(uav_locs)
    tx_xyz = np.vstack([bs_xyz, uav_xyz]) if bs_xyz else uav_xyz
    ue_xyz = ue_pos[:, None, :]              # [n_ue, 1, 3]
    tx_xyz = tx_xyz[None, :, :]              # [1, n_tx, 3]

    diff = ue_xyz - tx_xyz                  # [n_ue, n_tx, 3]
    d2d = np.linalg.norm(diff[..., :2], axis=-1)
    d3d = np.linalg.norm(diff, axis=-1)

    h_bs = tx_xyz[..., 2]
    h_ut = ue_xyz[..., 2]
    p_los = _uma_los_probability(d2d, h_bs, h_ut)

    pl_los = _uma_pathloss(env.carrier_hz, d3d, h_bs, h_ut, los=True)
    pl_nlos = _uma_pathloss(env.carrier_hz, d3d, h_bs, h_ut, los=False)

    g_los = 10 ** (-pl_los / 10.0)
    g_nlos = 10 ** (-pl_nlos / 10.0)
    g_exp = p_los * g_los + (1.0 - p_los) * g_nlos

    p_tx_lin = 10 ** (env.bs_power_dbm / 10.0) / 1000.0
    p_rx = p_tx_lin * g_exp                     # [n_ue, n_tx]

    best_idx = np.argmax(p_rx, axis=1)
    sinr_lin = p_rx[np.arange(p_rx.shape[0]), best_idx] / env.no

    sinr_db = 10.0 * np.log10(sinr_lin)
    serving = best_idx.astype(int)
    serving[sinr_db < sinr_cutoff_db] = -1

    return serving, sinr_lin


def optimize_uav_positions_uma(env,
                               n_trials=100,
                               xy_extent=2000,
                               z_min=50,
                               z_max=1000,
                               timeout=None,
                               random_seed=0,
                               use_ue_offset=False):
    """Bayesian optimisation using TR 38.901 UMa model."""
    n_tx = len(env.uavs)

    def as_xyz(trial):
        xs = [trial.suggest_float(f"x{i}", -xy_extent, xy_extent) for i in range(n_tx)]
        ys = [trial.suggest_float(f"y{i}", -xy_extent, xy_extent) for i in range(n_tx)]
        z_rel = [trial.suggest_float(f"z{i}", z_min, z_max) for i in range(n_tx)]
        zs = [_terrain_relative_z(env, xs[i], ys[i], z_rel[i]) for i in range(n_tx)]
        return np.stack([xs, ys, zs], axis=1)

    def objective(trial):
        xyz = as_xyz(trial)
        serving, sinr_lin = _check_sinr_uma(env, xyz,
                                            sinr_cutoff_db=5,
                                            use_ue_offset=use_ue_offset)
        served_mask = serving != -1
        if not served_mask.any():
            return 0.0

        B = 1
        sumrate = np.sum(B * np.log2(1 + sinr_lin[served_mask]))
        ue_counts = np.bincount(serving[served_mask], minlength=n_tx)
        numer = (ue_counts.sum()) ** 2
        denom = len(ue_counts) * np.sum(ue_counts ** 2)
        jain = numer / denom if denom > 0 else 0.0
        coverage = served_mask.mean()
        return sumrate * coverage * jain

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best_x = [study.best_params[f"x{i}"] for i in range(n_tx)]
    best_y = [study.best_params[f"y{i}"] for i in range(n_tx)]
    best_z_rel = [study.best_params[f"z{i}"] for i in range(n_tx)]
    best_z = [_terrain_relative_z(env, best_x[i], best_y[i], best_z_rel[i])
              for i in range(n_tx)]
    best_xyz = np.stack([best_x, best_y, best_z], axis=1)
    return best_xyz, study.best_value


def _modified_sigmoid(z, t, kappa):
    """Modified sigmoid used by the AOI weighting term."""
    return 1.0 / (1.0 + np.exp(-kappa * (z - t)))


def optimize_uav_positions_aoi(env,
                               n_trials=100,
                               xy_extent=2000,
                               z_min=50,
                               z_max=1000,
                               timeout=None,
                               random_seed=0,
                               alpha=0.01,
                               beta=1.0,
                               gamma=0.8,
                               d_min=500.0,
                               aoi_center=(0.0, 0.0, 0.0),
                               aoi_radius=800.0,
                               aoi_centers=None,
                               aoi_radii=None,
                               kappa_a=0.02,
                               kappa_i=0.02,
                               grid_points_per_axis=5,
                               use_ue_offset=False):
    """Bayesian optimization with AOI-aware placement loss.

    Loss (minimize):
        Lp = -alpha * K + beta * Pa + gamma * Pu

    where:
        K  = coverage factor using evenly spaced XY grid support points.
        Pa = AOI attraction penalty for AOI centered at ``aoi_center``.
        Pu = repulsion penalty to keep UAVs apart by ``d_min``.
    """
    n_tx = len(env.uavs)
    if aoi_centers is None:
        aoi_centers = np.array([aoi_center], dtype=float)
    else:
        aoi_centers = np.array(aoi_centers, dtype=float)

    if aoi_centers.ndim == 1:
        aoi_centers = aoi_centers[None, :]

    if aoi_radii is None:
        aoi_radii = np.full(aoi_centers.shape[0], float(aoi_radius), dtype=float)
    else:
        aoi_radii = np.array(aoi_radii, dtype=float)

    if aoi_radii.shape[0] != aoi_centers.shape[0]:
        raise ValueError("aoi_radii length must match number of aoi_centers")

    grid_axis = np.linspace(-xy_extent, xy_extent, int(grid_points_per_axis))
    grid_xy = np.array(np.meshgrid(grid_axis, grid_axis), dtype=float).reshape(2, -1).T
    if hasattr(env, "get_ground_height"):
        grid_z = np.array([float(env.get_ground_height(gx, gy)) for gx, gy in grid_xy],
                          dtype=float)
    else:
        grid_z = np.zeros(grid_xy.shape[0], dtype=float)
    grid_xyz = np.column_stack((grid_xy, grid_z))

    def as_xyz(trial):
        xs = [trial.suggest_float(f"x{i}", -xy_extent, xy_extent) for i in range(n_tx)]
        ys = [trial.suggest_float(f"y{i}", -xy_extent, xy_extent) for i in range(n_tx)]
        z_rel = [trial.suggest_float(f"z{i}", z_min, z_max) for i in range(n_tx)]
        zs = [_terrain_relative_z(env, xs[i], ys[i], z_rel[i]) for i in range(n_tx)]
        return np.stack([xs, ys, zs], axis=1)

    def objective(trial):
        xyz = as_xyz(trial)

        # Coverage factor: distance from each ground-level grid point (z_rel=0) to the closest UAV
        grid_uav_dist = np.linalg.norm(grid_xyz[:, None, :] - xyz[None, :, :], axis=-1)
        min_dist_per_grid = np.min(grid_uav_dist, axis=1)
        K = float(np.sum(min_dist_per_grid))

        # AOI attraction penalty (multiple AOIs)
        Pa = 0.0
        for k, center in enumerate(aoi_centers):
            radius = float(aoi_radii[k])
            d_center = np.linalg.norm(xyz - center[None, :], axis=1)
            occupancy_sigmoid = _modified_sigmoid(
                d_center,
                (2.0 / 3.0) * radius,
                -kappa_i,
            )
            omega = float(np.exp(-np.sum(occupancy_sigmoid)))

            Pa += float(np.sum(
                omega * d_center
                - (1.0 - omega) * np.exp(-kappa_a * (d_center - radius))
            ))

        # Repulsion penalty
        Pu = 0.0
        for i in range(n_tx):
            for j in range(i + 1, n_tx):
                dij = np.linalg.norm(xyz[i] - xyz[j])
                Pu += max(0.0, d_min - float(dij))

        loss = -alpha * K + beta * Pa + gamma * Pu
        return float(loss)

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best_x = [study.best_params[f"x{i}"] for i in range(n_tx)]
    best_y = [study.best_params[f"y{i}"] for i in range(n_tx)]
    best_z_rel = [study.best_params[f"z{i}"] for i in range(n_tx)]
    best_z = [_terrain_relative_z(env, best_x[i], best_y[i], best_z_rel[i])
              for i in range(n_tx)]
    best_xyz = np.stack([best_x, best_y, best_z], axis=1)
    return best_xyz, study.best_value
