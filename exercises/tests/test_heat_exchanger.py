"""Unit tests for the heat exchanger temperature control simulation module."""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from heat_exchanger import HeatExchangerAnalyzer, HeatExchangerParameters


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Provides a sample DataFrame mimicking experimental data."""
    return pd.DataFrame(
        {
            "tempo": [0.0, 5.0, 10.0, 15.0],
            "Tpv": [70.0, 79.775, 79.995, 76.554],
        }
    )


@pytest.fixture
def analyzer() -> HeatExchangerAnalyzer:
    """Provides an instance of HeatExchangerAnalyzer with standard parameters."""
    return HeatExchangerAnalyzer()


def test_control_error_and_proportional_action(
    analyzer: HeatExchangerAnalyzer, sample_data: pd.DataFrame
) -> None:
    """Validates control error and proportional action calculations."""
    df = analyzer.compute_simulation(sample_data)

    # For Tpv = 70.0, T_sp = 80.0 => e = 10.0, u = K_p * 10 = 20.0
    assert np.isclose(df.loc[0, "e"], 10.0)
    assert np.isclose(df.loc[0, "u"], 20.0)

    # For Tpv = 79.995 => e = 0.005, u = 0.01
    assert np.isclose(df.loc[2, "e"], 0.005)
    assert np.isclose(df.loc[2, "u"], 0.010)


def test_valve_opening_and_saturation(analyzer: HeatExchangerAnalyzer) -> None:
    """Verifies that valve opening respects physical saturation bounds [0, 100]%."""
    extreme_data = pd.DataFrame(
        {
            "tempo": [0.0, 1.0],
            "Tpv": [20.0, 150.0],  # 20.0 gives large positive error, 150 gives negative error
        }
    )
    df = analyzer.compute_simulation(extreme_data)

    # For Tpv = 20: e = 60, u = 120, uv = 33.3 + 120 = 153.3 -> saturated to 100.0
    assert df.loc[0, "uv"] == 100.0

    # For Tpv = 150: e = -70, u = -140, uv = 33.3 - 140 = -106.7 -> saturated to 0.0
    assert df.loc[1, "uv"] == 0.0


def test_process_stream_disturbance_step(
    analyzer: HeatExchangerAnalyzer, sample_data: pd.DataFrame
) -> None:
    """Validates that process flow rate steps from 3.0 to 4.0 kg/min at t = 10 min."""
    df = analyzer.compute_simulation(sample_data)

    # t < 10
    assert df.loc[0, "m_in"] == 3.0
    assert df.loc[1, "m_in"] == 3.0

    # t >= 10
    assert df.loc[2, "m_in"] == 4.0
    assert df.loc[3, "m_in"] == 4.0


def test_thermal_energy_conservation(
    analyzer: HeatExchangerAnalyzer, sample_data: pd.DataFrame
) -> None:
    """Ensures dynamic thermal energy conservation: Q_liq = Q_v - Q_processo - Q_perdas."""
    df = analyzer.compute_simulation(sample_data)

    expected_q_liq = df["Q_v"] - df["Q_processo"] - df["Q_perdas"]
    assert np.allclose(df["Q_liq"], expected_q_liq)


def test_temperature_rate_scaling(
    analyzer: HeatExchangerAnalyzer, sample_data: pd.DataFrame
) -> None:
    """Verifies temperature derivative scaling between seconds and minutes."""
    df = analyzer.compute_simulation(sample_data)

    expected_dt_dt_s = df["Q_liq"] / analyzer.params.c_heat_capacity
    assert np.allclose(df["dT_dt_s"], expected_dt_dt_s)
    assert np.allclose(df["dT_dt_min"], 60.0 * df["dT_dt_s"])


def test_load_data_validation(analyzer: HeatExchangerAnalyzer) -> None:
    """Ensures proper exceptions are raised on invalid data paths or malformed schemas."""
    with pytest.raises(FileNotFoundError):
        analyzer.load_data(Path("non_existent_file.csv"))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp.write("col_a,col_b\n1,2\n")
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(ValueError):
            analyzer.load_data(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
