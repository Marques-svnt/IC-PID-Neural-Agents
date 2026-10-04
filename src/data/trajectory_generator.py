"""Synthetic trajectory generator for the nonlinear CSTR thermal process.

This module simulates multiple dynamic scenarios including setpoint changes,
feed temperature disturbances, reactant concentration variations, and measurement
noise to create rich datasets for training adaptive neural controllers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.systems.cstr import CSTRParameters, CSTRSystem

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryConfig:
    """Configuration for synthetic trajectory generation.

    Attributes:
        num_trajectories: Number of distinct trajectories to generate.
        duration: Total duration of each trajectory [min].
        dt: Sampling time interval [min].
        setpoint_range: Range (min, max) of temperature setpoint targets [K].
        feed_temp_range: Range (min, max) of feed temperature disturbances [K].
        feed_conc_range: Range (min, max) of feed concentration disturbances [mol/L].
        measurement_noise_std: Standard deviation of additive Gaussian sensor noise [K].
        random_seed: Random seed for deterministic reproducibility.
    """

    num_trajectories: int = 15
    duration: float = 25.0
    dt: float = 0.05
    setpoint_range: Tuple[float, float] = (390.0, 405.0)
    feed_temp_range: Tuple[float, float] = (345.0, 360.0)
    feed_conc_range: Tuple[float, float] = (0.9, 1.1)
    measurement_noise_std: float = 0.08
    random_seed: int = 42


@dataclass
class ProcessTrajectory:
    """Container holding time-series signals from a single trajectory simulation.

    Attributes:
        time: Time vector [min].
        setpoint: Reference temperature trajectory [K].
        temperature: Measured reactor temperature [K].
        error: Tracking error e(t) = T(t) - T_sp(t) [K] (reverse-acting convention).
        d_error: Error derivative de/dt [K/min].
        integral_error: Accumulated error integral [K*min].
        control_action: Applied coolant flow rate q_j [L/min].
        concentration: Reactant concentration C_A [mol/L].
        jacket_temperature: Jacket temperature T_j [K].
        kp_applied: Baseline proportional gain applied during simulation.
        ki_applied: Baseline integral gain applied during simulation.
        kd_applied: Baseline derivative gain applied during simulation.
    """

    time: NDArray[np.float64]
    setpoint: NDArray[np.float64]
    temperature: NDArray[np.float64]
    error: NDArray[np.float64]
    d_error: NDArray[np.float64]
    integral_error: NDArray[np.float64]
    control_action: NDArray[np.float64]
    concentration: NDArray[np.float64]
    jacket_temperature: NDArray[np.float64]
    kp_applied: NDArray[np.float64]
    ki_applied: NDArray[np.float64]
    kd_applied: NDArray[np.float64]


class TrajectoryGenerator:
    """Generates synthetic closed-loop and open-loop trajectories on the CSTR.

    Uses the phenomenological CSTRSystem model with anti-windup PID control
    across varying operating points to construct supervision datasets.
    """

    def __init__(
        self,
        config: Optional[TrajectoryConfig] = None,
        system_params: Optional[CSTRParameters] = None,
    ) -> None:
        """Initializes the trajectory generator.

        Args:
            config: TrajectoryConfig dataclass specifying simulation bounds.
            system_params: Physical parameters for the CSTR.
        """
        self.config = config if config is not None else TrajectoryConfig()
        self.system = CSTRSystem(params=system_params, dt=self.config.dt)
        self.rng = np.random.default_rng(self.config.random_seed)

        logger.info(
            "TrajectoryGenerator initialized: %d trajectories, duration=%.1f min, dt=%.3f min",
            self.config.num_trajectories,
            self.config.duration,
            self.config.dt,
        )

    def generate_single_trajectory(
        self,
        target_setpoint: float,
        disturb_t_f: float,
        disturb_c_af: float,
        baseline_kp: float = 25.0,
        baseline_ki: float = 18.0,
        baseline_kd: float = 3.5,
    ) -> ProcessTrajectory:
        """Simulates a single closed-loop trajectory under specific disturbances.

        Args:
            target_setpoint: Final desired temperature setpoint [K].
            disturb_t_f: Reactant feed inlet temperature disturbance [K].
            disturb_c_af: Reactant feed concentration disturbance [mol/L].
            baseline_kp: Proportional gain for baseline tracking.
            baseline_ki: Integral gain for baseline tracking.
            baseline_kd: Derivative gain for baseline tracking.

        Returns:
            ProcessTrajectory object containing all logged time series.
        """
        cfg = self.config
        num_steps = int(cfg.duration / cfg.dt)
        time_vec = np.linspace(0.0, cfg.duration, num_steps + 1)

        # Preallocate signal arrays
        t_sp_arr = np.zeros(num_steps + 1, dtype=np.float64)
        t_arr = np.zeros(num_steps + 1, dtype=np.float64)
        error_arr = np.zeros(num_steps + 1, dtype=np.float64)
        d_error_arr = np.zeros(num_steps + 1, dtype=np.float64)
        integral_arr = np.zeros(num_steps + 1, dtype=np.float64)
        u_arr = np.zeros(num_steps + 1, dtype=np.float64)
        ca_arr = np.zeros(num_steps + 1, dtype=np.float64)
        tj_arr = np.zeros(num_steps + 1, dtype=np.float64)
        kp_arr = np.full(num_steps + 1, baseline_kp, dtype=np.float64)
        ki_arr = np.full(num_steps + 1, baseline_ki, dtype=np.float64)
        kd_arr = np.full(num_steps + 1, baseline_kd, dtype=np.float64)

        # Initial state at nominal operating point
        state = self.system.get_nominal_steady_state()
        current_u = self.system.cstr_params.q_j_ss

        ca_arr[0] = state[0]
        t_arr[0] = state[1]
        tj_arr[0] = state[2]
        u_arr[0] = current_u

        # Step time schedule
        step_time_1 = cfg.duration * 0.15
        step_time_2 = cfg.duration * 0.55
        disturb_time = cfg.duration * 0.70

        # Baseline controller with anti-windup clamping
        integral_val = 0.0
        prev_error = 0.0

        for k in range(num_steps):
            t_curr = k * cfg.dt

            # Scheduled setpoint: multi-stage profile
            if t_curr < step_time_1:
                sp = self.system.cstr_params.t_ss
            elif t_curr < step_time_2:
                sp = target_setpoint
            else:
                sp = (
                    target_setpoint
                    + (self.system.cstr_params.t_ss - target_setpoint) * 0.5
                )
            t_sp_arr[k] = sp

            # Disturbance injections
            t_f_in = (
                disturb_t_f if t_curr >= disturb_time else self.system.cstr_params.t_f0
            )
            c_af_in = (
                disturb_c_af if t_curr >= disturb_time else self.system.cstr_params.c_af
            )

            # Measured temperature with sensor noise
            noise = float(self.rng.normal(0.0, cfg.measurement_noise_std))
            t_meas = float(state[1] + noise)

            # Reverse acting error: e = T_meas - T_sp (higher temp -> higher cooling u)
            err = t_meas - sp
            error_arr[k] = err

            d_err = (err - prev_error) / cfg.dt if cfg.dt > 0 else 0.0
            d_error_arr[k] = d_err

            # Anti-windup conditional integration
            is_saturated = (current_u >= self.system.cstr_params.u_max and err > 0) or (
                current_u <= self.system.cstr_params.u_min and err < 0
            )
            if not is_saturated:
                integral_val += err * cfg.dt
            integral_arr[k] = integral_val

            # Baseline PID calculation
            u_raw = (
                self.system.cstr_params.q_j_ss
                + baseline_kp * err
                + baseline_ki * integral_val
                + baseline_kd * d_err
            )

            # Integrate plant state forward one dt
            state, current_u = self.system.integrate_step(
                state=state,
                u=u_raw,
                prev_u=current_u,
                dt=cfg.dt,
                t_f=t_f_in,
                c_af=c_af_in,
            )

            ca_arr[k + 1] = state[0]
            t_arr[k + 1] = state[1]
            tj_arr[k + 1] = state[2]
            u_arr[k + 1] = current_u
            prev_error = err

        # Final step padding for consistency
        t_sp_arr[-1] = t_sp_arr[-2]
        err_final = t_arr[-1] - t_sp_arr[-1]
        error_arr[-1] = err_final
        d_error_arr[-1] = d_error_arr[-2]
        integral_arr[-1] = integral_val

        return ProcessTrajectory(
            time=time_vec,
            setpoint=t_sp_arr,
            temperature=t_arr,
            error=error_arr,
            d_error=d_error_arr,
            integral_error=integral_arr,
            control_action=u_arr,
            concentration=ca_arr,
            jacket_temperature=tj_arr,
            kp_applied=kp_arr,
            ki_applied=ki_arr,
            kd_applied=kd_arr,
        )

    def generate_dataset(self) -> List[ProcessTrajectory]:
        """Generates a complete batch of trajectories across parameter ranges.

        Returns:
            List of ProcessTrajectory objects.
        """
        cfg = self.config
        trajectories: List[ProcessTrajectory] = []

        logger.info(
            "Starting generation of %d synthetic trajectories...", cfg.num_trajectories
        )

        for i in range(cfg.num_trajectories):
            sp_target = float(
                self.rng.uniform(cfg.setpoint_range[0], cfg.setpoint_range[1])
            )
            t_f_dist = float(
                self.rng.uniform(cfg.feed_temp_range[0], cfg.feed_temp_range[1])
            )
            c_af_dist = float(
                self.rng.uniform(cfg.feed_conc_range[0], cfg.feed_conc_range[1])
            )

            # Gain perturbation for diverse training feedback
            kp_var = float(self.rng.uniform(18.0, 35.0))
            ki_var = float(self.rng.uniform(12.0, 24.0))
            kd_var = float(self.rng.uniform(2.0, 5.5))

            traj = self.generate_single_trajectory(
                target_setpoint=sp_target,
                disturb_t_f=t_f_dist,
                disturb_c_af=c_af_dist,
                baseline_kp=kp_var,
                baseline_ki=ki_var,
                baseline_kd=kd_var,
            )
            trajectories.append(traj)

        logger.info(
            "Trajectory dataset generation complete (%d trajectories).",
            len(trajectories),
        )
        return trajectories

    def save_to_numpy(
        self,
        trajectories: List[ProcessTrajectory],
        output_path: Path,
    ) -> None:
        """Saves a batch of trajectories as a compressed NumPy .npz archive.

        Args:
            trajectories: List of ProcessTrajectory objects.
            output_path: Path to target file (e.g., data/trajectories/cstr_trajectories.npz).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        features_list = []
        targets_list = []
        gains_list = []

        for t in trajectories:
            # Features: [error, d_error, integral_error, temperature]
            feats = np.column_stack(
                [t.error, t.d_error, t.integral_error, t.temperature]
            )
            # Targets: control_action
            targets = t.control_action.reshape(-1, 1)
            # Gains: [Kp, Ki, Kd]
            gains = np.column_stack([t.kp_applied, t.ki_applied, t.kd_applied])

            features_list.append(feats)
            targets_list.append(targets)
            gains_list.append(gains)

        all_features = np.concatenate(features_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)
        all_gains = np.concatenate(gains_list, axis=0)

        np.savez_compressed(
            output_path,
            features=all_features,
            targets=all_targets,
            gains=all_gains,
        )
        logger.info(
            "Saved %d total state samples to %s",
            len(all_features),
            output_path,
        )
