"""Unit tests for the TrajectoryGenerator and CSTRDataset modules."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CSTRDataset, FeatureScaler
from src.data.trajectory_generator import TrajectoryConfig, TrajectoryGenerator


class TestTrajectoryGenerator:
    """Test suite for synthetic CSTR trajectory generation."""

    @pytest.fixture
    def generator(self) -> TrajectoryGenerator:
        """Fixture providing a fast-running generator."""
        cfg = TrajectoryConfig(
            num_trajectories=2,
            duration=4.0,
            dt=0.1,
            random_seed=123,
        )
        return TrajectoryGenerator(config=cfg)

    def test_single_trajectory_signals_integrity(
        self,
        generator: TrajectoryGenerator,
    ) -> None:
        """Verifies that generated trajectories have matching lengths and valid numbers."""
        traj = generator.generate_single_trajectory(
            target_setpoint=398.0,
            disturb_t_f=352.0,
            disturb_c_af=1.05,
        )

        expected_steps = int(generator.config.duration / generator.config.dt) + 1
        assert len(traj.time) == expected_steps
        assert len(traj.temperature) == expected_steps
        assert len(traj.control_action) == expected_steps
        assert len(traj.error) == expected_steps

        # No NaNs or Infs allowed
        assert not np.isnan(traj.temperature).any()
        assert not np.isnan(traj.control_action).any()
        assert not np.isinf(traj.temperature).any()

        # Control action strictly bounded within actuator limits
        assert (traj.control_action >= 0.0).all()
        assert (traj.control_action <= 300.0).all()

    def test_save_and_load_dataset(
        self,
        generator: TrajectoryGenerator,
        tmp_path,
    ) -> None:
        """Verifies saving to .npz and loading into CSTRDataset."""
        trajectories = generator.generate_dataset()
        save_path = tmp_path / "test_trajs.npz"

        generator.save_to_numpy(trajectories, save_path)
        assert save_path.exists()

        dataset = CSTRDataset.from_file(save_path, target_key="targets")
        assert len(dataset) > 0
        feat, target = dataset[0]

        assert isinstance(feat, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert feat.shape == (4,)
        assert target.shape == (1,)


class TestCSTRDataset:
    """Test suite for PyTorch CSTRDataset and FeatureScaler."""

    def test_feature_scaler_fit_transform(self) -> None:
        """Verifies z-score standardization and inverse reconstruction."""
        data = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]], dtype=np.float64)
        scaler = FeatureScaler().fit(data)

        transformed = scaler.transform(data)
        assert np.isclose(np.mean(transformed, axis=0), 0.0, atol=1e-7).all()
        assert np.isclose(np.std(transformed, axis=0), 1.0, atol=1e-7).all()

        reconstructed = scaler.inverse_transform(transformed)
        assert np.allclose(data, reconstructed)

    def test_dataset_splitting_and_dataloader(self) -> None:
        """Verifies dataset partitioning and batch loading through DataLoader."""
        n_samples = 100
        features = np.random.randn(n_samples, 4)
        targets = np.random.randn(n_samples, 1)

        dataset = CSTRDataset(features=features, targets=targets)
        train_ds, val_ds, test_ds = dataset.split(val_ratio=0.2, test_ratio=0.1)

        assert len(train_ds) == 70
        assert len(val_ds) == 20
        assert len(test_ds) == 10

        loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        batch_feat, batch_target = next(iter(loader))
        assert batch_feat.shape == (16, 4)
        assert batch_target.shape == (16, 1)
