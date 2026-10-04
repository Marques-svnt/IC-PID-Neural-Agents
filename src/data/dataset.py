"""PyTorch Dataset and feature scaling abstractions for CSTR control trajectories.

Provides dataset management, feature normalization, and train/val/test splitting
for training supervised neural networks to predict adaptive gains and actions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FeatureScaler:
    """Z-score feature normalizer with epsilon protection against division by zero."""

    def __init__(self, eps: float = 1e-8) -> None:
        """Initializes the feature scaler.

        Args:
            eps: Small constant to avoid zero variance division.
        """
        self.eps = eps
        self.mean: Optional[NDArray[np.float64]] = None
        self.std: Optional[NDArray[np.float64]] = None

    def fit(self, data: NDArray[np.float64]) -> FeatureScaler:
        """Computes empirical mean and standard deviation along features (axis 0).

        Args:
            data: Array of shape (num_samples, num_features).

        Returns:
            Self instance for chained calls.
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Avoid division by zero for zero-variance features
        self.std[self.std < self.eps] = 1.0
        return self

    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalizes data using fitted parameters.

        Args:
            data: Raw feature array.

        Returns:
            Normalized feature array.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fit() before calling transform().")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstructs original physical scale from normalized inputs.

        Args:
            data: Normalized feature array.

        Returns:
            Reconstructed array in original physical units.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError(
                "Scaler must be fit() before calling inverse_transform()."
            )
        return (data * self.std) + self.mean


class CSTRDataset(Dataset):
    """PyTorch Dataset holding CSTR state trajectories for neural training.

    Features vector x: [error, d_error, integral_error, temperature] (dim=4).
    Target vector y: [target_u] or [Kp, Ki, Kd] gains.
    """

    def __init__(
        self,
        features: Union[NDArray[np.float64], torch.Tensor],
        targets: Union[NDArray[np.float64], torch.Tensor],
        scaler: Optional[FeatureScaler] = None,
        fit_scaler: bool = True,
    ) -> None:
        """Initializes CSTRDataset.

        Args:
            features: Input state feature matrix (N, num_features).
            targets: Target matrix (N, target_dim).
            scaler: Optional pre-fitted FeatureScaler instance.
            fit_scaler: If True and scaler is None, fits a new scaler on features.
        """
        feats_np = (
            features.cpu().numpy()
            if isinstance(features, torch.Tensor)
            else np.asarray(features, dtype=np.float64)
        )
        targets_np = (
            targets.cpu().numpy()
            if isinstance(targets, torch.Tensor)
            else np.asarray(targets, dtype=np.float32)
        )

        if fit_scaler and scaler is None:
            self.scaler = FeatureScaler().fit(feats_np)
            normalized_feats = self.scaler.transform(feats_np)
        elif scaler is not None:
            self.scaler = scaler
            normalized_feats = self.scaler.transform(feats_np)
        else:
            self.scaler = None
            normalized_feats = feats_np

        self.features = torch.tensor(normalized_feats, dtype=torch.float32)
        self.targets = torch.tensor(targets_np, dtype=torch.float32)

        logger.info(
            "CSTRDataset loaded with %d samples | Feature dim: %d, Target dim: %d",
            len(self.features),
            self.features.shape[1],
            self.targets.shape[1],
        )

    def __len__(self) -> int:
        """Returns total number of samples in dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves single sample tuple (features, target).

        Args:
            idx: Sample index.

        Returns:
            Tuple of (feature_tensor, target_tensor).
        """
        return self.features[idx], self.targets[idx]

    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
        target_key: str = "targets",
        scaler: Optional[FeatureScaler] = None,
    ) -> CSTRDataset:
        """Loads dataset directly from a saved .npz archive.

        Args:
            filepath: Path to .npz file containing 'features' and target_key.
            target_key: Key name for target data ('targets' or 'gains').
            scaler: Optional pre-fitted FeatureScaler.

        Returns:
            Instantiated CSTRDataset.
        """
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Trajectory file not found: {p}")

        data = np.load(p)
        features = data["features"]
        targets = data[target_key]

        return cls(features=features, targets=targets, scaler=scaler)

    def split(
        self,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> Tuple[CSTRDataset, CSTRDataset, CSTRDataset]:
        """Splits current dataset into training, validation, and test subsets.

        Args:
            val_ratio: Fraction of data allocated to validation.
            test_ratio: Fraction of data allocated to testing.
            random_seed: Deterministic random seed for indexing.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        total = len(self)
        n_val = int(total * val_ratio)
        n_test = int(total * test_ratio)
        n_train = total - n_val - n_test

        indices = np.random.default_rng(random_seed).permutation(total)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        train_ds = CSTRDataset(
            features=self.features[train_idx],
            targets=self.targets[train_idx],
            scaler=None,
            fit_scaler=False,
        )
        val_ds = CSTRDataset(
            features=self.features[val_idx],
            targets=self.targets[val_idx],
            scaler=None,
            fit_scaler=False,
        )
        test_ds = CSTRDataset(
            features=self.features[test_idx],
            targets=self.targets[test_idx],
            scaler=None,
            fit_scaler=False,
        )

        return train_ds, val_ds, test_ds
