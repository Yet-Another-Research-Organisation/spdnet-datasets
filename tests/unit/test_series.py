"""Tests for the synthetic covariance-series generator (spdnet_datasets.synthetic.series)."""

import math
from pathlib import Path

import pytest
import torch

from spdnet_datasets.synthetic.series import (
    SeriesConfig,
    generate_series,
    sample_wishart,
    validate_series,
)
from yetanotherspdnet.functions.spd_geometries import affine_invariant as ai


@pytest.fixture(scope="module")
def series():
    return generate_series(SeriesConfig(), n_series_per_class=200)


def test_shapes_and_spd(series):
    assert series.data.shape == (600, 8, 3, 3)
    assert series.labels.shape == (600,)
    assert series.trajectory.shape == (8, 3, 3)
    assert series.centres.shape == (3, 8, 3, 3)
    assert torch.linalg.eigvalsh(series.data).min() > 0
    assert torch.linalg.eigvalsh(series.centres).min() > 0


def test_reproducible():
    a = generate_series(SeriesConfig(seed=3), 5)
    b = generate_series(SeriesConfig(seed=3), 5)
    assert torch.equal(a.data, b.data)


def test_trajectory_is_geodesic(series):
    config = SeriesConfig()
    steps = ai.affine_invariant_distance(series.trajectory[:-1], series.trajectory[1:])
    torch.testing.assert_close(
        steps, torch.full_like(steps, config.trajectory_norm / (config.n_steps - 1))
    )
    total = ai.affine_invariant_distance(series.trajectory[0], series.trajectory[-1])
    torch.testing.assert_close(total, torch.tensor(config.trajectory_norm, dtype=total.dtype))


@pytest.mark.parametrize("noise", [0.0, 0.1])
def test_centre_of_centres_is_trajectory(noise):
    """Mean-removed class vectors make G(t) the exact Karcher mean of the G_i(t)."""
    series = generate_series(SeriesConfig(n_classes=4, tangent_noise_std=noise), 1)
    log_coords = ai.affine_invariant_log_coordinates(series.centres, series.trajectory)
    torch.testing.assert_close(
        log_coords.sum(dim=0), torch.zeros_like(log_coords[0]), atol=1e-10, rtol=0
    )


def test_class_configuration(series):
    """Default angles 360/N and radius r: equilateral configuration, rigid in time."""
    config = SeriesConfig()
    radius = ai.affine_invariant_distance(series.trajectory[None], series.centres)
    torch.testing.assert_close(radius, torch.full_like(radius, config.class_radius))
    # in coordinates at G(t), centres form an equilateral triangle of side r*sqrt(3)
    coords = ai.affine_invariant_log_coordinates(series.centres, series.trajectory)
    sides = torch.stack(
        [(coords[i] - coords[j]).norm(dim=-1) for i, j in [(0, 1), (1, 2), (0, 2)]]
    )
    torch.testing.assert_close(sides, torch.full_like(sides, config.class_radius * math.sqrt(3)))
    # parallel transport is an isometry: inter-centre distances do not depend on t
    distances = ai.affine_invariant_distance(series.centres[:, None], series.centres[None])
    torch.testing.assert_close(distances, distances[..., :1].expand_as(distances))


def test_class_plane_orthogonal_to_trajectory(series):
    coords_v = ai.affine_invariant_log_coordinates(series.trajectory[-1], series.trajectory[0])
    coords_u = ai.affine_invariant_log_coordinates(series.centres[:, 0], series.trajectory[0])
    torch.testing.assert_close(
        coords_u @ coords_v, torch.zeros(3, dtype=coords_u.dtype), atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("inverse", [False, True])
def test_samples_centred_in_arithmetic_mean(inverse):
    generator = torch.Generator().manual_seed(0)
    centre = torch.tensor([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 0.5]], dtype=torch.float64)
    samples = sample_wishart(centre, 50, 20000, generator, inverse=inverse)
    assert torch.linalg.eigvalsh(samples).min() > 0
    distance = ai.affine_invariant_distance(samples.mean(dim=0), centre)
    assert distance < 0.03


@pytest.mark.parametrize("inverse", [False, True])
def test_karcher_scale_bias(inverse):
    """
    Samples are centred in arithmetic mean only. Since harmonic <= Karcher <= arithmetic
    (Loewner order), their Karcher mean is shrunk in scale for both distributions;
    for G = I, p = 3, df = 20 the expected log det is about -0.32 (W) and -0.35 (IW).
    """
    generator = torch.Generator().manual_seed(1)
    centre = torch.eye(3, dtype=torch.float64)
    samples = sample_wishart(centre, 20, 5000, generator, inverse=inverse)
    karcher = ai.affine_invariant_mean(samples, 30)
    harmonic = torch.linalg.inv(torch.linalg.inv(samples).mean(dim=0))
    log_dets = [torch.logdet(m).item() for m in (harmonic, karcher, samples.mean(dim=0))]
    assert log_dets[0] < log_dets[1] < log_dets[2]
    assert log_dets[1] < -0.2


def test_dispersion_decreases_with_df():
    dispersions = []
    for df in (30, 100, 300, 1000):
        series = generate_series(SeriesConfig(df=df), 100)
        dispersions.append(max(validate_series(series)["dispersion_q"]))
    assert all(a > b for a, b in zip(dispersions, dispersions[1:]))


def test_df_per_class():
    series = generate_series(SeriesConfig(df=[30, 300, 3000]), 100)
    dispersion = validate_series(series)["dispersion_q"]
    assert dispersion[0] > dispersion[1] > dispersion[2]


def test_validate_default_config(series):
    """The default configuration satisfies the spec's separation conditions."""
    report = validate_series(series)
    assert report["min_eigval"] > 0
    assert report["separation_ratio"] > 2
    assert report["no_temporal_overlap"]


def test_rotating_scenario_tokens_carry_no_class():
    """Rotating scenario: per step, every class vector has norm r and zero mean over phases."""
    config = SeriesConfig(scenario="rotating", df=1000)
    series = generate_series(config, 400)
    assert series.data.shape == (1200, config.n_steps, 3, 3)
    assert series.centres.shape == (1200, config.n_steps, 3, 3)
    assert (torch.linalg.eigvalsh(series.data) > 0).all()
    coords = ai.affine_invariant_log_coordinates(series.centres, series.trajectory)  # (n, T, d)
    torch.testing.assert_close(
        coords.norm(dim=-1), torch.full(coords.shape[:2], config.class_radius, dtype=coords.dtype)
    )
    for label in range(config.n_classes):
        mean = coords[series.labels == label].mean(dim=0)  # (T, d)
        assert mean.norm(dim=-1).max() < 0.1 * config.class_radius


def test_rotating_scenario_speeds():
    """Consecutive class vectors (both read at G(0)) turn by c_i pi / (T - 1)."""
    config = SeriesConfig(scenario="rotating", class_speeds=[1, -1, 2])
    series = generate_series(config, 5)
    g0 = series.trajectory[0]
    logs = ai.affine_invariant_parallel_transport(
        ai.affine_invariant_log(series.trajectory, series.centres), series.trajectory, g0
    )
    coords = ai.affine_invariant_log_coordinates(ai.affine_invariant_exp(g0, logs), g0)
    cos = (coords[:, 1:] * coords[:, :-1]).sum(-1) / config.class_radius**2
    for label, speed in enumerate(config.class_speeds):
        expected = math.cos(speed * math.pi / (config.n_steps - 1))
        observed = cos[series.labels == label]
        torch.testing.assert_close(observed, torch.full_like(observed, expected))


def test_match_log_det_removes_the_df_shift():
    """Same centres (r = 0), different df: with match_log_det, log det no longer depends on the class."""
    config = SeriesConfig(class_radius=0.0, df=[10, 20, 80], match_log_det=True, n_steps=2)
    series = generate_series(config, 3000)
    log_det = torch.logdet(series.data) - torch.logdet(series.trajectory)  # (n, T)
    means = torch.stack([log_det[series.labels == i].mean() for i in range(3)])
    assert means.abs().max() < 0.03
    raw = generate_series(SeriesConfig(class_radius=0.0, df=[10, 20, 80], n_steps=2), 3000)
    raw_log_det = torch.logdet(raw.data) - torch.logdet(raw.trajectory)
    assert raw_log_det[raw.labels == 0].mean() < raw_log_det[raw.labels == 2].mean() - 0.3


def test_otsu_threshold_separates_two_modes():
    import numpy as np

    from spdnet_datasets.real.series import otsu_threshold

    values = np.concatenate([np.full(300, 0.1), np.full(700, 0.8)]) + np.random.default_rng(0).normal(0, 0.02, 1000)
    threshold = otsu_threshold(values)
    assert values[:300].max() <= threshold < values[300:].min()


@pytest.mark.skipif(not Path("/DATA/DeepHS-Fruit").is_dir(), reason="DeepHS-Fruit not available")
def test_deephs_series_shapes_and_groups():
    import numpy as np

    from spdnet_datasets.real.series import deephs_series

    data, labels, n_pixels, groups = deephs_series(max_images=4)
    assert data.shape == (4, 27, 16, 16) and np.all(np.linalg.eigvalsh(data) > 0)
    assert np.all(n_pixels >= 2 * 16) and set(labels) <= {0, 1, 2}
    assert groups[0] == groups[1]  # front and back of avocado_day_01_01


def test_jet_to_scalar_inverts_the_colour_map():
    import numpy as np
    from matplotlib import colormaps

    from spdnet_datasets.real.series import jet_to_scalar

    values = np.linspace(0, 1, 50)[None, :]
    rgb = (colormaps["jet"](values)[..., :3] * 255).round().astype(np.uint8)
    assert np.abs(jet_to_scalar(rgb) - values).max() < 0.01


@pytest.mark.skipif(not Path("/DATA/CI4R-MULTI3").is_dir(), reason="CI4R-MULTI3 not available")
def test_ci4r_series_shapes():
    import numpy as np

    from spdnet_datasets.real.series import ci4r_series

    data, labels, n_columns, groups = ci4r_series(max_images=3)
    assert data.shape == (3, 13, 16, 16) and np.all(np.linalg.eigvalsh(data) > 0)
    assert n_columns >= 2 * 16 and set(labels) == {0} and len(set(groups)) == 1


def test_uav_series_blocks_do_not_share_pixels(tmp_path):
    import numpy as np

    from spdnet_datasets.real.series import uav_series

    rng = np.random.default_rng(0)
    n, w = 120, 15
    np.save(tmp_path / "uav_windows_data.npy", rng.random((n, w, w, 40)).astype(np.float32))
    np.save(tmp_path / "uav_windows_labels.npy", rng.choice([1.0, 2.0, 5.0], n, p=[0.5, 0.45, 0.05]))
    centres = rng.integers(0, 512, (n, 2))
    np.save(tmp_path / "uav_windows_positions.npy", centres)
    data, labels, n_pixels, groups = uav_series(str(tmp_path), min_windows=10, max_per_class=30)
    assert data.shape[1:] == (4, 16, 16) and np.all(np.linalg.eigvalsh(data) > 0)
    assert set(labels) <= {0, 1} and np.all(n_pixels == w * w)  # the rare crop 5 is dropped
    # one group = one (block, crop): two windows of different groups never overlap
    kept = np.array([c for c in centres if ((c % 128 >= w // 2) & (c % 128 <= 128 - w)).all()])
    assert len(data) <= len(kept)
    for g in np.unique(groups):
        assert len(np.unique(labels[groups == g])) == 1


def test_kaggle_wheat_series_drops_empty_patches(tmp_path):
    import numpy as np
    import tifffile

    from spdnet_datasets.real.series import kaggle_wheat_series

    rng = np.random.default_rng(0)
    tifffile.imwrite(tmp_path / "Health_hyper_1.tif", rng.integers(100, 4000, (32, 32, 126), dtype=np.uint16))
    tifffile.imwrite(tmp_path / "Rust_hyper_1.tif", rng.integers(100, 4000, (32, 32, 125), dtype=np.uint16))
    tifffile.imwrite(tmp_path / "Other_hyper_1.tif", np.zeros((32, 32, 125), dtype=np.uint16))
    data, labels, n_pixels = kaggle_wheat_series(str(tmp_path))
    assert data.shape == (2, 14, 16, 16) and np.all(np.linalg.eigvalsh(data) > 0)
    assert list(labels) == [0, 2] and np.all(n_pixels == 1024)


def test_per_position_scenario_changes_the_class_plane():
    """Per-position scenario: G(t) is still the centre of the centres, the class radius is r at every
    step, but the class configuration of one step is not the transported configuration of the next."""
    static = generate_series(SeriesConfig(n_features=6), 10)
    config = SeriesConfig(n_features=6, scenario="per_position")
    series = generate_series(config, 10)
    assert (torch.linalg.eigvalsh(series.data) > 0).all()
    coords = ai.affine_invariant_log_coordinates(series.centres, series.trajectory)  # (N, T, d) at G(t)
    torch.testing.assert_close(coords.mean(dim=0), torch.zeros_like(coords[0]), atol=1e-10, rtol=0)
    torch.testing.assert_close(coords.norm(dim=-1).amax(dim=0), torch.full((config.n_steps,), config.class_radius,
                                                                        dtype=coords.dtype))
    static_coords = ai.affine_invariant_log_coordinates(static.centres, static.trajectory)

    def step_agreement(c):  # mean |cosine| between u_i(t) and u_i(t + 1), in coordinates at each G(t)
        return torch.nn.functional.cosine_similarity(c[:, :-1], c[:, 1:], dim=-1).abs().mean()

    assert step_agreement(static_coords) > 0.95 and step_agreement(coords) < 0.6


@pytest.mark.parametrize("scenario", ["static", "per_position"])
def test_continuous_target_moves_the_centre_along_one_direction(scenario):
    """Continuous target: SPD samples, y in [-1, 1], centre coordinates at G(t) of norm r |y| and of the
    same direction (up to sign) for every series at a given step."""
    config = SeriesConfig(n_features=4, scenario=scenario, target="continuous")
    series = generate_series(config, 20)
    assert series.labels.shape == (60,) and series.labels.is_floating_point()
    assert series.labels.abs().max() <= 1 and (torch.linalg.eigvalsh(series.data) > 0).all()
    coords = ai.affine_invariant_log_coordinates(series.centres, series.trajectory)  # (n, T, d)
    torch.testing.assert_close(coords.norm(dim=-1), config.class_radius * series.labels.abs()[:, None].expand(-1, config.n_steps))
    unit = coords / coords.norm(dim=-1, keepdim=True)
    assert ((unit * unit[:1]).sum(-1).abs() > 1 - 1e-8).all()
