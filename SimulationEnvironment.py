import numpy as np
import pandas as pd
import mitsuba as mi
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
#mpl.use("Agg") # for running from ssh
import sionna as sn
import drjit as dr

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    if gpu_num!="":
        print(f'\nUsing GPU {gpu_num}\n')
    else:
        print('\nUsing CPU\n')
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from sionna.rt import (
    load_scene, Camera, Transmitter, PlanarArray,
    RadioMapSolver, transform_mesh, Receiver,
    PathSolver, subcarrier_frequencies 
)

from sionna.phy import config
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel, LMMSEPostEqualizationSINR
from sionna.phy.constants import BOLTZMANN_CONSTANT
from sionna.phy.nr.utils import decode_mcs_index
from sionna.phy.utils import log2, dbm_to_watt, lin_to_db
from sionna.sys import PHYAbstraction, OuterLoopLinkAdaptation, \
    PFSchedulerSUMIMO, downlink_fair_power_control
from coarse_pf_scheduler import CoarsePFSchedulerSUMIMO
from sionna.sys.utils import spread_across_subcarriers

from bayesian_optimizer import optimize_uav_positions

from Utilities import *

config.precision = 'single'  # 'single' or 'double'

# Set random seed for reproducibility
config.seed = 48


class SimulationEnvironment:
    def __init__(self, seed=42):
        # Default scene is San Francisco for now.
        self.scene = load_scene(sn.rt.scene.san_francisco)

        self.noise_temp_k = 290.0
        self.carrier_hz = 2.0e9  
        self.bler_target = 0.1
        self.mcs_table_index = 1
        self.subcarrier_spacing = 30e3
        self.num_subcarriers = 51*12 # 51 PRBs = 20MHz, 133 = 50MHz
        self.num_ofdm_symbols = 12
        self.no = BOLTZMANN_CONSTANT * self.noise_temp_k * self.subcarrier_spacing
        self.frequencies = subcarrier_frequencies(num_subcarriers=self.num_subcarriers,
                                                 subcarrier_spacing=self.subcarrier_spacing)

        self.scene.frequency = self.carrier_hz
        self.scene.bandwidth = self.subcarrier_spacing * self.num_subcarriers
        self.scene.temperature = self.noise_temp_k
        self.bs_power_dbm = 23

        self.scene.tx_array = PlanarArray(
            num_rows=4, num_cols=4,
            pattern="dipole", polarization="V"
        )

        self.ue_array = PlanarArray( 
            num_rows=1, num_cols=1,
            pattern="iso",            
            polarization="V"
        )
        self.scene.rx_array = self.ue_array

        self.ues = []
        self.ues_pert = []            # positions used by the digital twin
        self.txs = []                 # all transmitters (BS + UAV)
        self.uavs = []                # UAV transmitters only
        self.base_stations = []       # gNodeB transmitters
        self.bs_positions = []        # original BS locations for toggling
        self.bs_active = []

        self.seed = seed
        self.rm_samples_per_tx = 10**6
        self.rm_eval_samples_per_tx = 10**6
        self.fidelity_eval = False

    def add_tx(self, name, location, tx_power=30):
        """Add a UAV transmitter to the scene."""
        loc = location
        tx = Transmitter(name=name,
                  position=loc,
                  display_radius=20)

        loc[2] = 0
        tx.look_at(loc)
        self.bs_power_dbm = tx_power
        tx.power_dbm = self.bs_power_dbm

        self.txs.append(tx)
        self.uavs.append(tx)
        self.scene.add(tx)

    def get_ground_height(self, x, y, z_drop=2000.0, max_hits=32):
        """Return terrain height at ``(x, y)`` using downward ray casting.

        The ray may first intersect building roofs. In that case we keep
        marching downward until a hit on the ``Terrain`` shape is found.
        Returns ``None`` when the ray exits the mesh without a terrain hit.
        """
        def _shape_id(shape):
            try:
                return str(shape.id())
            except Exception:
                return str(shape)

        z_cursor = float(z_drop)
        for _ in range(max_hits):
            ray = mi.Ray3f(
                o=mi.Point3f(np.array([x, y, z_cursor], dtype=np.float32)),
                d=mi.Vector3f(0, 0, -1),
            )
            si = self.scene.mi_scene.ray_intersect(ray)
            if not si.is_valid():
                return None

            p = np.squeeze(si.p.numpy())
            hit_z = float(p[2])
            sid = _shape_id(si.shape).lower()
            if "terrain" in sid:
                return hit_z

            z_cursor = hit_z - 0.5

        return None

    def add_base_station(self, name, position, orientation=[0, 0, 0],
                         tx_power=30, color=(0, 0, 1)):
        """Add a fixed gNodeB transmitter."""
        tx = Transmitter(name=name,
                          position=position,
                          orientation=orientation,
                          display_radius=20,
                          color=color)
        tx.power_dbm = tx_power
        self.txs.append(tx)
        self.base_stations.append(tx)
        self.bs_positions.append(np.array(position))
        self.bs_active.append(True)
        self.scene.add(tx)

    def set_base_station_active(self, index: int, active: bool) -> None:
        """Enable or disable a gNodeB by moving it below ground."""
        tx = self.base_stations[index]
        self.bs_active[index] = active
        if active:
            tx.position = np.array(self.bs_positions[index])
            tx.power_dbm = 23
        else:
            off_pos = np.array(self.bs_positions[index])
            off_pos[2] = -500
            tx.position = off_pos
            tx.power_dbm = -200

    def deploy_ues(self, num_ue=20, location_error=0.0):
        # If we sample full random, some UEs might be in the buildings.
        # So we give some lower bound to sampling after initial RM.

        meas = self.scene.objects["Terrain"].clone(as_mesh=True)
        transform_mesh(meas, translation=np.array([0, 0, 1.5]))

        rm = self.rm_solver(
            self.scene,
            measurement_surface=meas,
            samples_per_tx=self.rm_samples_per_tx, # To give less fidelity to positions
            max_depth=self.rm_max_depth,
            diffuse_reflection=False,#self.rm_diffuse,
            refraction=self.rm_refraction,
        )

        pos, cell_ids = rm.sample_positions(num_pos=num_ue//len(self.txs), 
                                            metric="path_gain",
                                            min_val_db=-120, # -100 for failure
                                            max_val_db=50,
                                            min_dist=0,
                                            max_dist=5000,
                                            tx_association=False,
                                            center_pos=False,
                                            seed=self.seed)
        pos = pos.numpy()
        pos = pos.reshape(-1, 3)             # flatten the TX dimension

        # --- parameters for the local perturbation -------------------------
        err_r   = location_error    # [m] maximum horizontal offset, average err is location_error/2
        max_trials = 60       # attempts per UE to find a valid perturbed point
        z_drop     = 1000.0   # start 1km above

        perturbed_list = []                          # collect here

        # TODO: Probably we can make this parallel.
        sigma = err_r / 1.41421
        for p in pos:
            found = False
            for _ in range(max_trials):
                # 1) random planar offset
                phi = np.random.uniform(0, 2*np.pi)
                cand_xy = np.array(p[:2]) + np.random.normal(0, sigma, 2)

                # 2) drop a ray straight down
                ray = mi.Ray3f(
                    o = mi.Point3f(np.array([cand_xy[0], cand_xy[1], z_drop])),
                    d = mi.Vector3f(0, 0, -1)
                )
                si = self.scene.mi_scene.ray_intersect(ray)
                if not si.is_valid():
                    continue                             # off the mesh – try again

                prim_idx = int(si.prim_index.numpy().item())
                pg_lin   = rm.path_gain[0, prim_idx].numpy().item()
                if pg_lin == 0:                          # no coverage in that cell
                    continue

                pg_db = 10.0*np.log10(pg_lin)
                if pg_db < -120.0:                       # your original threshold
                    continue

                # 3) accept this candidate
                p_new = si.p.numpy()                     # (3,) float64
                p_new = np.squeeze(p_new)
                perturbed_list.append(p_new.astype(np.float32))
                found = True
                break

            if not found:
                # fall back to the original UE position
                perturbed_list.append(p.astype(np.float32))

        # final NumPy array, guaranteed shape (num_ue, 3)
        perturbed_pos = np.stack(perturbed_list, axis=0)
        self.ues_pert = perturbed_pos.copy()


        for k, p in enumerate(pos):
            # original UE (magenta)
            rx = Receiver(name=f"ue_{k}",
                          position=p,
                          display_radius=10,
                          color=(0.50, 0.0, 0.0))
            rx.array = self.ue_array
            self.scene.add(rx)
            self.ues.append(rx)
        
        # To show perturbations
        show_perturbations = False
        if show_perturbations:
            for k, p in enumerate(perturbed_pos):
                rx = Receiver(name=f"pert_ue_{k}",
                              position=p,
                              display_radius=10,
                              color=(0.0, 0.0, 0.0))
                rx.array = self.ue_array
                self.scene.add(rx)
                #self.ues.append(rx)


    def init_solvers(
        self,
        samples_per_tx=10**6,
        max_depth=8,
        refraction=False,
        diffuse=False,
        eval_samples_per_tx=None,
        fidelity_eval=False,
    ):
        self.rm_solver = RadioMapSolver()
        self.path_solver = PathSolver()

        self.rm_samples_per_tx = samples_per_tx
        if eval_samples_per_tx is None:
            eval_samples_per_tx = samples_per_tx
        self.rm_eval_samples_per_tx = eval_samples_per_tx
        self.rm_max_depth = max_depth
        self.rm_refraction = refraction
        self.rm_diffuse = diffuse
        self.fidelity_eval = fidelity_eval
        
    def call_rm_solver(self):
        meas = self.scene.objects["Terrain"].clone(as_mesh=True)
        transform_mesh(meas, translation=np.array([0, 0, 1.5]))

        self.rm = self.rm_solver(
            self.scene,
            measurement_surface=meas,
            samples_per_tx=self.rm_samples_per_tx,
            max_depth=self.rm_max_depth,
            diffuse_reflection=self.rm_diffuse,
            refraction=self.rm_refraction,
        )
    
    def call_path_solver(self, use_perturbed=False):
        """Run the ray-tracing path solver.

        Parameters
        ----------
        use_perturbed : bool
            If ``True``, temporarily replace UE positions with the perturbed
            ones stored in :attr:`ues_pert` when computing the channel. This
            enables modelling location errors in the digital twin.
        """

        orig_pos = None
        if use_perturbed:
            orig_pos = [np.array(rx.position) for rx in self.ues]
            for rx, p in zip(self.ues, self.ues_pert):
                rx.position = p

        paths = self.path_solver(scene=self.scene,
                                 max_depth=self.rm_max_depth,
                                 samples_per_src=self.rm_samples_per_tx,
                                 los=True,
                                 specular_reflection=True,
                                 diffuse_reflection=self.rm_diffuse,
                                 refraction=self.rm_refraction,
                                 synthetic_array=True,
                                 seed=self.seed)

        if orig_pos is not None:
            # Restore original UE positions
            for rx, p in zip(self.ues, orig_pos):
                rx.position = p

        self.paths = paths

    def render_scene(self, metric="sinr"):
        cam = Camera(position=[0, 500, 3000])
        cam.look_at(np.array([0, 0, 0]))

        self.scene.render(
                camera=cam,
                radio_map=self.rm,
                rm_metric=metric,
                rm_vmin=-40,
                rm_vmax=40,
                rm_show_color_bar=True,
                fov=30,
                resolution=(1200,1000)
            )

    def render_paths(self):
        cam = Camera(position=[0, 500, 3000])
        cam.look_at(np.array([0, 0, 0]))

        self.scene.render(camera=cam,
                          paths=self.paths,
                          fov=30,
                          resolution=(1200,1000))

    def check_sinr(self, tx_locs, use_ue_offset=False):
        """Return UE association and SINR.

        Parameters
        ----------
        tx_locs : list
            List of transmitter positions. The order must match
            ``self.txs`` so that base stations remain fixed while UAV
            coordinates are updated.
        use_ue_offset : bool
            If ``True``, the perturbed UE positions generated during deployment
            are used to emulate location errors.
        """

        for tx, loc in zip(self.txs, tx_locs):
            tx.position = loc
            #tx.look_at(np.array([loc[0], loc[1], 0.0]))  # point towards ground

        # Compute paths with optional UE location error
        self.call_path_solver(use_perturbed=use_ue_offset)

        sinr_cutoff_db = -5# TODO make this tied to sim -6
        # 3) Channel frequency responses  (shape: [ut, ut_ant, bs, bs_ant,
        #                                       num_sym, num_sc])
        h_f = self.paths.cfr(frequencies=self.frequencies,
                             out_type="tf")

        g_lin = tf.reduce_mean(tf.abs(h_f)**2, axis=[1, 3, 4, 5])   # [ue,bs]
        p_rx  = dbm_to_watt(self.bs_power_dbm) * g_lin              # [ue,bs]

        serving = tf.argmax(p_rx, axis=1, output_type=tf.int32)      # [ue]
        gather  = tf.stack([tf.range(len(self.ues), dtype=tf.int32),
                            serving], axis=1)
        p_sig = tf.gather_nd(p_rx, gather)                           # [ue]
        p_int = tf.reduce_sum(p_rx, axis=1) - p_sig
        n0    = tf.fill(tf.shape(p_sig), tf.cast(self.no, tf.float32))

        sinr_lin = p_sig / (p_int + n0)
        sinr_db  = lin_to_db(sinr_lin)

        # --- apply cut-off --------------------------------------------------
        below = sinr_db < sinr_cutoff_db
        serving_np = serving.numpy()
        serving_np[below.numpy()] = -1

        cmap = plt.get_cmap("brg", len(self.txs))

        colors = [(1,0,0), (0.4, 0.0, 0.5)]
        for ue_id, rx in enumerate(self.ues):
            if serving_np[ue_id] == -1:
                rx.color = (0.0, 0.0, 0.0)          # grey = unserved
            else:
                rx.color = tuple(cmap(serving_np[ue_id])[:3])
                #rx.color = colors[serving_np[ue_id]]
        for tx_id, tx in enumerate(self.txs):
            tx.color = tuple(cmap(tx_id)[:3])
            #tx.color = colors[tx_id]

        #print(serving_np, sinr_db)

        return serving_np, sinr_lin
        

    def estimate_achievable_rate(self, channel_gain):
        no = self.no
        # [num_ut, num_ut_ant, num_bs, num_bs_ant, num_ofdm_symbols, num_subcarriers]
        rate_achievable_est = log2(tf.cast(1, tf.float32) + channel_gain / self.no)
        # [num_ut, num_bs, num_ofdm_symbols, num_subcarriers]
        rate_achievable_est = tf.reduce_mean(rate_achievable_est, axis=[-3, -5])
        # [num_bs, num_ofdm_symbols, num_subcarriers, num_ut]
        rate_achievable_est = tf.transpose(rate_achievable_est, [1, 2, 3, 0])
        return rate_achievable_est

    def build_rx_tx_association(self, serving):
        """
        serving : tf.Tensor, shape [num_rx]  (-1 kapsama dışı)
        """
        num_tx = len(self.txs)
        valid   = serving >= 0
        clipped = tf.where(valid, serving, tf.zeros_like(serving))
        assoc   = tf.one_hot(clipped, depth=num_tx, dtype=tf.int32)
        assoc   = assoc * tf.expand_dims(tf.cast(valid, tf.int32), -1)

        # GPU’da kalır; StreamManagement hemen öncesinde CPU’ya kopyalanır
        self.rx_tx_association = assoc
        return assoc

    def get_served_sets(self, serving_tf):
        sets = []
        for b in range(len(self.txs)):
            idx = tf.where(serving_tf == b)[:,0]
            sets.append(idx.numpy())

        return sets

    @tf.function(jit_compile=True)
    def step(self, h, harq_feedback, sinr_eff_feedback, num_decoded_bits):
        """ Perform system-level operations at a single slot:
        - Scheduling
        - Power control
        - SINR computation
        - Link adaptation
        - PHY abstraction
        Perfect channel knowledge at both transmitter and receivers is assumed.
        """
        # Compute channel gain for scheduling using the possibly mismatched channel
        # [num_ut, num_ut_ant, num_bs, num_bs_ant, num_ofdm_symbols, num_subcarriers]
        channel_gain = tf.cast(tf.abs(h)**2, tf.float32)

        # Estimate achievable rate via Shannon formula
        # [num_bs, num_ofdm_symbols, num_subcarriers, num_ut]
        rate_achievable_est = self.estimate_achievable_rate(channel_gain)

        # --------- #
        # Scheduler #
        # --------- #
        # Determine which stream of which user is scheduled on which RE
        # [num_bs, num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]
        # num_decoded_bits = rate_last_slot
        # TODO: I don't like this solution. I add +1000 to calm down scheduling giving
        # Everything to associated UTs with terrible sinr.
        rate_last_slot = tf.where(harq_feedback == 1,
                                  num_decoded_bits,
                                  tf.zeros_like(num_decoded_bits))

        is_scheduled = self.scheduler(rate_last_slot, rate_achievable_est)

        # Determine the index of the scheduled user for every RE
        # [num_bs, num_ofdm_symbols, num_subcarriers]
        ut_scheduled = tf.argmax(tf.reduce_sum(
            tf.cast(is_scheduled, tf.int32), axis=-1), axis=-1)

        # Compute the number of allocated REs per user
        # [num_bs, num_ut]
        num_allocated_re = tf.reduce_sum(
            tf.cast(is_scheduled, tf.int32), axis=[-4, -3, -1])

        # Compute the average pathloss per user
        # [num_ut, num_bs]
        pathloss_per_ut = tf.reduce_mean(1 / channel_gain,
                                         axis=[1, 3, 4, 5])
        # [num_bs, num_ut]
        pathloss_per_ut = tf.transpose(pathloss_per_ut, [1, 0])

        # ------------- #
        # Power control #
        # ------------- #
        # Allocate power to each user in fair manner
        # [num_bs, num_ut]
        tx_power_per_ut, _ = downlink_fair_power_control(
            pathloss_per_ut,
            self.no,
            num_allocated_re,
            bs_max_power_dbm=self.bs_power_dbm,
            # in [0;1]; Minimum ratio of power for each user
            guaranteed_power_ratio=.2,
            fairness=0)  # Fairness parameter>=0. If 0, sum rate across users is maximized


        # Spread power uniformly across allocated subcarriers and streams
        # [num_bs, num_tx_per_bs, num_streams_per_tx, num_ofdm_sym, num_subcarriers]
        tx_power = spread_across_subcarriers(
            tf.expand_dims(tx_power_per_ut, axis=-2),
            is_scheduled,
            num_tx=1)#self.scene.tx_array.num_ant)

        zero_ut = tf.squeeze(tx_power_per_ut == 0.0, axis=0)        # [num_ut]  (tek BS varsayıyoruz)

        # ------------- #
        # Schenenigans  #
        # ------------- #
        # H’nin UE ekseni 0.  True olan satırlara komple 0 bas.
        # This essentially creates a block diagonal matrix of scheduled users only.
        # h: [num_ut, ut_ant, num_bs, bs_ant, sym, sc]
        h_masked = tf.where(zero_ut[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis],
                     tf.zeros_like(h),
                     h)

        # ---------------- #
        # SINR computation #
        # ---------------- #
        # Effective channel upon regularized zero-forcing precoding
        precoded_channel = RZFPrecodedChannel(resource_grid=self.resource_grid,
                                              stream_management=self.stream_management)
        # NOTE: Ideally the precoder weights should be computed using ``h`` and
        # then applied to a different evaluation channel. Sionna's API does not
        # expose the weights directly, therefore we use the same channel for
        # both weight computation and effective channel evaluation.
        h_eff = precoded_channel(h_masked[tf.newaxis, ...],
                                 h_hat=h_masked[tf.newaxis, ...],
                                 tx_power=tx_power,
                                 alpha=self.no)
        #h_eff = precoded_channel(h[tf.newaxis, ...], tx_power=tx_power, alpha=self.no)

        # LMMSE post-equalization SINR computation
        lmmse_posteq_sinr = LMMSEPostEqualizationSINR(resource_grid=self.resource_grid,
                                                      stream_management=self.stream_management)
        # [batch_size, num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]
        sinr = lmmse_posteq_sinr(h_eff, no=self.no, interference_whitening=True)

        # --------------- #
        # Link adaptation #
        # --------------- #
        # [num_bs, num_ut]
        mcs_index = self.olla(num_allocated_re=num_allocated_re,
                         sinr_eff=sinr_eff_feedback,
                         mcs_table_index=self.mcs_table_index,
                         mcs_category=1,  # downlink
                         harq_feedback=harq_feedback)

        # --------------- #
        # PHY abstraction #
        # --------------- #
        # [num_bs, num_ut]
        num_decoded_bits, harq_feedback, sinr_eff_true, *_ = self.phy_abs(
            mcs_index,
            sinr=sinr,
            mcs_table_index=self.mcs_table_index,
            mcs_category=1)  # downlink)

        sinr_eff_db_true = lin_to_db(sinr_eff_true)

        # SINR feedback for OLLA
        sinr_eff_feedback = tf.where(num_allocated_re > 0, sinr_eff_true, 0)

        # Spectral efficiency
        # [num_bs, num_ut]
        mod_order, coderate = decode_mcs_index(
            mcs_index,
            table_index=self.mcs_table_index,
            is_pusch=False)

        se_la = tf.where(harq_feedback == 1,
                         tf.cast(mod_order, coderate.dtype) * coderate,
                         tf.cast(0, tf.float32))

        # Shannon capacity
        se_shannon = log2(1 + sinr_eff_true)
        return harq_feedback, sinr_eff_feedback, num_decoded_bits, \
            mcs_index, se_la, se_shannon, sinr_eff_db_true, \
            self.scheduler.pf_metric, ut_scheduled

    def evaluate_single_bs(self, bs_id:int, ue_idx:np.ndarray, num_slots:int=200):
        """Evaluate link-level performance for a single BS.

        UEs accumulating five consecutive negative HARQ feedbacks are
        deactivated and excluded from subsequent scheduling.
        """

        if len(ue_idx) == 0:
            print(f"{bs_id}: skipping")
            return None, None

        # ----- Kanalı alt-kümele -----
        h = tf.gather(self.h_freq, ue_idx, axis=0)      # UE ekseni süz
        h = tf.gather(h, [bs_id], axis=2)               # BS ekseni süz

        # şekil = [n_ue, ue_ant, 1, bs_ant, sym, sc]

        # ----- Sistem objelerini bir kere oluştur -----
        assoc  = tf.ones((len(ue_idx), 1), tf.int32)

        self.stream_management = StreamManagement(assoc.numpy(), len(ue_idx))

        num_tx_ant = self.scene.tx_array.num_ant

        self.resource_grid = ResourceGrid(self.num_ofdm_symbols,
                                          self.num_subcarriers,
                                          self.subcarrier_spacing,
                                          num_tx=num_tx_ant,
                                          num_streams_per_tx=1)

        self.phy_abs = PHYAbstraction()
        self.olla = OuterLoopLinkAdaptation(self.phy_abs, num_ut=len(ue_idx),
                                            bler_target=self.bler_target,
                                            batch_size=[1])

        self.scheduler = CoarsePFSchedulerSUMIMO(
            len(ue_idx),
            self.num_subcarriers,
            self.num_ofdm_symbols,
            batch_size=[1],
            num_streams_per_ut=1,
            rb_size=12, # this must divide total subcarrier count
            beta=.99)

        # ----- Tarih dizileri ( tam evaluate_linksz boyutları ) -----
        harq_hist  = np.full([num_slots, 1, len(ue_idx)], np.nan)
        se_la_hist = np.full_like(harq_hist, np.nan)
        se_sh_hist = np.full_like(harq_hist, np.nan)
        sinr_hist  = np.full_like(harq_hist, np.nan)
        sched_hist = np.full([
            num_slots, 1, self.num_ofdm_symbols, self.num_subcarriers], np.nan)
        nbits_hist = np.full([num_slots, 1, len(ue_idx)], np.nan, dtype=np.float32)

        # Consecutive HARQ NACK counters and UE activity mask
        nack_count = tf.zeros(len(ue_idx), tf.int32)
        active_mask = tf.ones(len(ue_idx), tf.bool)


        # ----- Slot döngüsü -----
        harq  = -tf.ones([1, len(ue_idx)], tf.int32)
        sinrf = tf.zeros_like(harq, tf.float32)
        nbits = tf.zeros_like(harq, tf.int32)

        for s in range(num_slots):
            # Mask out deactivated UEs from the channel
            mask = tf.reshape(active_mask, [len(ue_idx), 1, 1, 1, 1, 1])
            h_mask = tf.where(mask, h, tf.zeros_like(h))
            harq, sinrf, nbits, _, se_la, se_sh, sinr_db, _, ut_sched = \
                self.step(h_mask, harq, sinrf, nbits)

            harq_vec = tf.squeeze(harq, axis=0)
            harq_np = harq_vec.numpy()
            harq_hist[s,0,:] = harq_np
            se_la_hist[s,0,:] = se_la.numpy()[0,:]
            se_sh_hist[s,0,:] = se_sh.numpy()[0,:]
            sinr_hist[s,0,:] = sinr_db.numpy()[0,:]
            sched_hist[s,0,:,:] = ue_idx[ut_sched.numpy()[0,:,:]]
            nbits_hist[s,0,:] = nbits.numpy()[0,:]

            # Vectorized update of consecutive HARQ failure counters
            nack_count = tf.where(
                tf.logical_and(tf.equal(harq_vec, 0), active_mask),
                nack_count + 1,
                tf.where(tf.equal(harq_vec, 1), tf.zeros_like(nack_count), nack_count)
            )

            deactivate = tf.logical_and(active_mask, nack_count >= 5)
            if tf.reduce_any(deactivate):
                idx = tf.where(deactivate)
                scatter = tf.concat([tf.zeros_like(idx), idx], axis=1)
                harq = tf.tensor_scatter_nd_update(harq, scatter,
                                                  tf.fill([tf.shape(idx)[0]], -1))
                sinrf = tf.tensor_scatter_nd_update(sinrf, scatter,
                                                   tf.zeros([tf.shape(idx)[0]], sinrf.dtype))
                nbits = tf.tensor_scatter_nd_update(nbits, scatter,
                                                   tf.zeros([tf.shape(idx)[0]], nbits.dtype))
            active_mask = tf.where(deactivate, False, active_mask)


        # ----- Maskeleme & BLER -----
        not_sched = harq_hist == -1
        se_la_hist[not_sched] = np.nan
        se_sh_hist[not_sched] = np.nan
        sinr_hist [not_sched] = np.nan
        harq_hist [not_sched] = np.nan

        df_slot, df_alloc = hist_to_df(bs_id, ue_idx,
                                se_la_hist, se_sh_hist,
                                sinr_hist, harq_hist,
                                nbits_hist,
                                sched_hist)

        return df_slot, df_alloc


    def evaluate_links(self, num_steps=200):
        """Evaluate link-level performance.

        Parameters
        ----------
        num_steps : int
            Number of time slots to simulate.
        """

        df_all, alloc_all = [], []
        original_samples_per_tx = self.rm_samples_per_tx
        if self.fidelity_eval and self.rm_samples_per_tx != self.rm_eval_samples_per_tx:
            self.init_solvers(
                samples_per_tx=self.rm_eval_samples_per_tx,
                max_depth=self.rm_max_depth,
                refraction=self.rm_refraction,
                diffuse=self.rm_diffuse,
                eval_samples_per_tx=self.rm_eval_samples_per_tx,
                fidelity_eval=self.fidelity_eval,
            )

        try:
            tx_pos = [np.array(tx.position).flatten().tolist() for tx in self.txs]

            # Association and channel estimation with true UE locations
            serving_tf, _ = self.check_sinr(tx_pos)
            self.call_path_solver()
            self.h_freq = self.paths.cfr(frequencies=self.frequencies,
                                         sampling_frequency=1/self.subcarrier_spacing,
                                         num_time_steps=self.num_ofdm_symbols,
                                         out_type="tf")

            for bs in range(len(self.txs)):
                ue_set = tf.where(serving_tf == bs)[:,0].numpy()
                df_s, alloc_s = self.evaluate_single_bs(bs, ue_set,
                                                       num_slots=num_steps)
                if df_s is not None:
                    df_all.append(df_s)
                if alloc_s is not None:
                    alloc_all.append(alloc_s)
        finally:
            if self.rm_samples_per_tx != original_samples_per_tx:
                self.init_solvers(
                    samples_per_tx=original_samples_per_tx,
                    max_depth=self.rm_max_depth,
                    refraction=self.rm_refraction,
                    diffuse=self.rm_diffuse,
                    eval_samples_per_tx=self.rm_eval_samples_per_tx,
                    fidelity_eval=self.fidelity_eval,
                )

        df_slot = pd.concat(df_all, ignore_index=True) if df_all else pd.DataFrame()
        df_alloc = pd.concat(alloc_all, ignore_index=True) if alloc_all else pd.DataFrame()

        return df_slot, df_alloc


if __name__ == "__main__":
    env = SimulationEnvironment(seed=44)

    # Sample Valid User Positions
    env.add_tx("dummy", [0, 0, 1000], 23)
    env.init_solvers()
    env.deploy_ues(150, location_error=20.0)
    env.scene.remove("dummy")
    env.txs.clear()
    env.uavs.clear()
    env.base_stations.clear()
    env.bs_positions.clear()
    env.bs_active.clear()

    toggle = False
    if toggle:
        power = 43 + 40 - 5 # 
        angle = 10
        z = 600000
    else:
        power = 23
        angle = 15
        z = 150
    # Add UAV
    env.add_tx("uav_1", [0, 0, 1000], power)

    env.call_rm_solver()
    env.call_path_solver()
    env.render_scene()
    env.render_paths()
    plt.show()
    exit()

    env.uavs[0].position = np.array([0, np.tan(angle)*z, z])
    env.uavs[0].look_at([0,0,0])
    env.add_tx("uav_2", [-0, -0, 300])
    #env.uavs[1].position[2] = -400
    #env.uavs[1].power_dbm = -200

    env.add_tx("uav_3", [450, 300, 300])

    serving, sinr = env.check_sinr([np.array(tx.position).flatten().tolist() for tx in env.txs],
                                   use_ue_offset=False)

    covered = serving != -1
    coverage_ratio = np.sum(covered) / len(serving)
    print(sinr[covered])
    mean_sinr_lin = np.mean(10*np.log10(sinr[covered])) if np.any(covered) else 0.0

    print(f"Coverage ratio: {coverage_ratio:.2f}")
    print(f"Mean SINR (linear, covered only): {mean_sinr_lin:.4f}")


    if not toggle:
        best_xyz, best_obj = optimize_uav_positions(env,
                                                  n_trials=10,
                                                  xy_extent=500,
                                                  z_min=10, z_max=1000,
                                                  use_ue_offset=False)

        print("Best UAV positions (x,y,z):\n", best_xyz)
        #best_xyz[1,2] = -400

        bs_xyz = [np.array(tx.position).flatten().tolist() for tx in env.base_stations]
        serving, sinr = env.check_sinr(bs_xyz + best_xyz.tolist(), use_ue_offset=False)   # colours UEs and UAVs

        covered = serving != -1
        coverage_ratio = np.sum(covered) / len(serving)
        mean_sinr_lin = np.mean(10*np.log10((sinr[covered]))) if np.any(covered) else 0.0

    env.build_rx_tx_association(serving)

    df_slot, df_alloc = env.evaluate_links(num_steps=100)

    for bs_id, df_bs in df_alloc.groupby('bs'):
        plot_allocation_from_df(df_bs,
                                num_symbols=env.num_ofdm_symbols,
                                num_subcarriers=env.num_subcarriers,
                                title=f'OFDMA allocation – BS {bs_id}')

    for bs_id, df_bs in df_slot.groupby('bs'):
        plot_timeseries_from_df(df_bs, title="Time-series metrics")

    print(f"Coverage ratio: {coverage_ratio:.2f}")
    print(f"Mean SINR (linear, covered only): {mean_sinr_lin:.4f}")

    env.call_rm_solver()
    env.render_scene()
    env.render_paths()
    plt.show()
    #print(env.evaluate_links())
