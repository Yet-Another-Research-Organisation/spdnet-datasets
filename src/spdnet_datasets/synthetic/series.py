"""
Synthetic series of SPD matrices (covariance series) for sequence models.

Generation, step by step:

1. Trajectory. ``G(0) = random_SPD(p, cond)``; a random tangent vector of norm
   ``trajectory_norm`` (in orthonormal coordinates at ``G(0)``) gives
   ``G(T-1) = Exp_{G(0)}(v)``; the centres ``G(t)`` are sampled along the
   affine-invariant geodesic ``G(0) -> G(T-1)``.
2. Class centres. ``N`` tangent vectors ``u_i`` of norm ``class_radius``, at angles
   ``360°/N`` (by default) in a random plane orthogonal to ``v``, with their mean
   removed. They are parallel-transported along the trajectory, so the class
   configuration is the same at every ``t``: ``G_i(t) = Exp_{G(t)}(u_i(t))``.
   Optional tangent noise (centred over the classes, so that ``G(t)`` stays the
   centre of the centres) adds variability over time.
3. Samples. A sequence of class ``i`` is ``(Sigma_0, ..., Sigma_{T-1})`` with
   ``Sigma_t`` drawn independently around ``G_i(t)`` from a Wishart (or inverse
   Wishart) distribution whose arithmetic mean is ``G_i(t)``, with ``df`` per class.

Scenarios (``SeriesConfig.scenario``):

- ``"static"`` (default, the steps above): the class is readable at every step.
- ``"rotating"``: only the temporal order carries the class. Each series ``n`` of
  class ``i`` has its own class vector ``u_{i,n}(t) = r (cos a e1 + sin a e2)`` with
  ``a = phi_n + c_i omega t``: a random phase ``phi_n`` (uniform) and a class-specific
  angular speed ``c_i`` (``class_speeds``, default +1, -1, +2, ...), ``omega = pi / (T - 1)``.
  At any fixed step, the class vector is uniform on the circle of radius ``r`` for every
  class: a single token says nothing about the class, its direction of rotation does.
  The centre of the centres is ``G(t)`` in expectation over the phase.
- ``"per_position"``: as ``"static"``, but the plane of the class vectors is drawn
  independently at each step (orthogonal to ``v``): ``u_i(t) = r (cos a_i e1(t) + sin a_i e2(t))``.
  The class is readable at every step, but its signature differs from one position to the
  next, as for spectral windows or calendar dates: each position has its own identity.

Continuous target (``SeriesConfig.target = "continuous"``, regression), with the plane of
``"static"`` or ``"per_position"``: each series ``n`` draws ``y_n ~ U[-1, 1]`` and its centres are
``Exp_{G(t)}(r y_n e1(t))``. The label is ``y_n``; ``G(t)`` is the centre of the centres in
expectation over ``y``. ``n_series_per_class * n_classes`` series are drawn, as for classes.
"""

import math
from dataclasses import dataclass, field

import torch

from yetanotherspdnet.functions.spd_geometries.affine_invariant import (
    affine_invariant_distance,
    affine_invariant_exp,
    affine_invariant_exp_coordinates,
    affine_invariant_parallel_transport,
)
from yetanotherspdnet.functions.spd_linalg import (
    sqrtm_SPD,
    sym_coordinates_to_matrix,
)
from yetanotherspdnet.random.spd import random_SPD


@dataclass
class SeriesConfig:
    """Parameters of the synthetic covariance-series generator."""

    n_features: int = 3
    n_classes: int = 3
    n_steps: int = 8
    trajectory_norm: float = 6.3
    """Norm of the tangent vector v from G(0) to G(T-1): d_AI(G(0), G(T-1))."""
    class_radius: float = 0.3
    """Norm r of the class vectors u_i before centring (exact for default angles)."""
    class_angles: list[float] | None = None
    """Angles (degrees) of the u_i in their plane; default 360°/N spacing."""
    df: int | list[int] = 300
    """Wishart degrees of freedom, one value or one per class."""
    distribution: str = "wishart"
    """'wishart' or 'inverse_wishart'."""
    tangent_noise_std: float = 0.0
    """Std of the per-step tangent noise on each u_i (coordinates at G(t))."""
    g0_cond: float = 10.0
    match_log_det: bool = False
    """Wishart only: scale the samples of class i by exp(-E[log det(W/df_i)] / p), so that
    E[log det Sigma] = log det G_i(t) whatever df_i (removes the df-dependent shift of log det,
    a linear signature of the dispersion; the arithmetic mean is then no longer G_i(t))."""
    scenario: str = "static"
    """"static" or "rotating" (see module docstring)."""
    class_speeds: list[float] | None = None
    """"rotating" only: angular speed of each class, in units of pi / (T - 1)."""
    target: str = "class"
    """"class" (labels are class indices) or "continuous" (regression, see module docstring)."""
    seed: int = 0

    def df_per_class(self) -> list[int]:
        if isinstance(self.df, int):
            return [self.df] * self.n_classes
        assert len(self.df) == self.n_classes, "one df per class"
        return list(self.df)


@dataclass
class SeriesData:
    data: torch.Tensor
    """(n_series, T, p, p) SPD samples."""
    labels: torch.Tensor
    """(n_series,) class indices (float targets for a continuous target)."""
    trajectory: torch.Tensor
    """(T, p, p) centres of the class centres G(t)."""
    centres: torch.Tensor
    """(N, T, p, p) class centres G_i(t); "rotating": (n_series, T, p, p), one per series."""
    info: dict = field(default_factory=dict)


def _orthonormal_frame(
    first: torch.Tensor, n_vectors: int, generator: torch.Generator
) -> torch.Tensor:
    """Orthonormal vectors (n_vectors, d) whose first one is along `first`."""
    d = first.shape[-1]
    assert n_vectors <= d
    random = torch.randn(d, n_vectors - 1, dtype=first.dtype, generator=generator)
    basis, _ = torch.linalg.qr(torch.cat([first[:, None], random], dim=1))
    basis = basis * torch.sign(basis[:, :1].T @ first)  # keep the orientation of `first`
    return basis.T


def sample_wishart(
    centres: torch.Tensor,
    df: int,
    n_samples: int,
    generator: torch.Generator,
    inverse: bool = False,
) -> torch.Tensor:
    """
    SPD samples whose arithmetic mean is ``centres``.

    Wishart: ``Sigma = G^{1/2} (X X^T / df) G^{1/2}``, ``X`` of shape (p, df), N(0, 1).
    Inverse Wishart: ``Sigma = (df - p - 1) G^{1/2} (X X^T)^{-1} G^{1/2}``.

    centres : (..., p, p) -> samples : (n_samples, ..., p, p)
    """
    p = centres.shape[-1]
    shape = (n_samples, *centres.shape[:-2], p, df)
    x = torch.randn(shape, dtype=centres.dtype, generator=generator)
    scatter = x @ x.mT
    if inverse:
        assert df > p + 1, "inverse Wishart mean needs df > p + 1"
        white = (df - p - 1) * torch.linalg.inv(scatter)
    else:
        assert df >= p, "Wishart samples are SPD only for df >= p"
        white = scatter / df
    centres_sqrtm = sqrtm_SPD(centres)[0]
    return centres_sqrtm @ white @ centres_sqrtm


def wishart_log_det_bias(p: int, df: int) -> float:
    """E[log det(X X^T / df)], X of shape (p, df) with N(0, 1) entries (Wishart W(I, df) / df)."""
    i = torch.arange(1, p + 1, dtype=torch.float64)
    return (torch.special.digamma((df - i + 1) / 2).sum() + p * math.log(2) - p * math.log(df)).item()


def generate_series(config: SeriesConfig, n_series_per_class: int) -> SeriesData:
    """Generate labelled covariance series following ``config`` (see module docstring)."""
    p, n_classes, n_steps = config.n_features, config.n_classes, config.n_steps
    dim = p * (p + 1) // 2
    dtype = torch.float64
    generator = torch.Generator().manual_seed(config.seed)

    # 1. trajectory G(t) along the geodesic G(0) -> Exp_{G(0)}(v)
    g0 = random_SPD(p, cond=config.g0_cond, dtype=dtype, generator=generator)
    direction = torch.randn(dim, dtype=dtype, generator=generator)
    direction = direction / direction.norm()
    steps = torch.linspace(0, 1, n_steps, dtype=dtype)[:, None]
    trajectory = affine_invariant_exp_coordinates(
        steps * config.trajectory_norm * direction, g0
    )

    # 2. class vectors at G(0), in a random plane orthogonal to v, mean removed
    _, e1, e2 = _orthonormal_frame(direction, 3, generator)
    g0_sqrtm = sqrtm_SPD(g0)[0]
    if config.scenario == "rotating":
        return _rotating_series(config, n_series_per_class, generator, g0, g0_sqrtm, trajectory, e1, e2)
    assert config.scenario in ("static", "per_position"), f"unknown scenario {config.scenario}"
    angles = config.class_angles
    if angles is None:
        angles = [360 * i / n_classes for i in range(n_classes)]
    angles = torch.tensor(angles, dtype=dtype) * math.pi / 180
    if config.scenario == "per_position":  # one plane per step: (T, 2, d)
        planes = torch.stack([_orthonormal_frame(direction, 3, generator)[1:] for _ in range(n_steps)])
    else:
        planes = torch.stack([e1, e2])[None]  # the same plane at every step
    if config.target == "continuous":
        return _continuous_series(config, n_series_per_class * n_classes, generator, g0, g0_sqrtm, trajectory, planes[:, 0])
    assert config.target == "class", f"unknown target {config.target}"
    class_coords = config.class_radius * (
        torch.cos(angles)[:, None, None] * planes[None, :, 0] + torch.sin(angles)[:, None, None] * planes[None, :, 1]
    )  # (N, 1 or T, d)
    class_coords = class_coords - class_coords.mean(dim=0)

    # tangent vectors at G(0), parallel-transported to every G(t)
    tangent_g0 = g0_sqrtm @ sym_coordinates_to_matrix(class_coords, p) @ g0_sqrtm
    tangent = affine_invariant_parallel_transport(
        tangent_g0, g0, trajectory
    )  # (N, T, p, p)

    # optional tangent noise, centred over the classes (coordinates at G(t))
    if config.tangent_noise_std > 0:
        noise = config.tangent_noise_std * torch.randn(
            n_classes, n_steps, dim, dtype=dtype, generator=generator
        )
        noise = noise - noise.mean(dim=0)
        traj_sqrtm = sqrtm_SPD(trajectory)[0]
        tangent = tangent + traj_sqrtm @ sym_coordinates_to_matrix(noise, p) @ traj_sqrtm

    # G_i(t) = Exp_{G(t)}(u_i(t)): SPD for any tangent vector, noise included
    centres = affine_invariant_exp(trajectory, tangent)

    # 3. Wishart (or inverse Wishart) samples around each G_i(t)
    data, labels = [], []
    for i, df in enumerate(config.df_per_class()):
        samples = sample_wishart(
            centres[i],
            df,
            n_series_per_class,
            generator,
            inverse=config.distribution == "inverse_wishart",
        )
        if config.match_log_det:
            assert config.distribution == "wishart", "match_log_det is defined for Wishart samples"
            samples = samples * math.exp(-wishart_log_det_bias(p, df) / p)
        data.append(samples)
        labels.append(torch.full((n_series_per_class,), i))
    info = {
        "step_distance": config.trajectory_norm / (n_steps - 1),
        "class_radius": class_coords.norm(dim=-1).amax(dim=-1),
    }
    return SeriesData(torch.cat(data), torch.cat(labels), trajectory, centres, info)


def _rotating_series(
    config: SeriesConfig,
    n_series_per_class: int,
    generator: torch.Generator,
    g0: torch.Tensor,
    g0_sqrtm: torch.Tensor,
    trajectory: torch.Tensor,
    e1: torch.Tensor,
    e2: torch.Tensor,
) -> SeriesData:
    """Scenario "rotating": one centre series per sample, the class is the rotation speed."""
    p, n_classes, n_steps = config.n_features, config.n_classes, config.n_steps
    assert config.tangent_noise_std == 0, "no tangent noise in the rotating scenario"
    speeds = config.class_speeds
    if speeds is None:
        speeds = [(1 + i // 2) * (-1) ** i for i in range(n_classes)]  # +1, -1, +2, -2, ...
    assert len(speeds) == n_classes, "one speed per class"
    omega = math.pi / (n_steps - 1)
    steps = torch.arange(n_steps, dtype=g0.dtype)
    data, labels, centres = [], [], []
    for i, df in enumerate(config.df_per_class()):
        phases = 2 * math.pi * torch.rand(n_series_per_class, 1, dtype=g0.dtype, generator=generator)
        angles = phases + speeds[i] * omega * steps  # (n, T)
        coords = config.class_radius * (
            torch.cos(angles)[..., None] * e1 + torch.sin(angles)[..., None] * e2
        )  # (n, T, d) at G(0)
        tangent_g0 = g0_sqrtm @ sym_coordinates_to_matrix(coords, p) @ g0_sqrtm
        tangent = affine_invariant_parallel_transport(tangent_g0, g0, trajectory)
        series_centres = affine_invariant_exp(trajectory, tangent)  # (n, T, p, p)
        samples = sample_wishart(
            series_centres, df, 1, generator, inverse=config.distribution == "inverse_wishart"
        )[0]
        data.append(samples)
        centres.append(series_centres)
        labels.append(torch.full((n_series_per_class,), i))
    info = {
        "step_distance": config.trajectory_norm / (n_steps - 1),
        "class_radius": torch.full((n_classes,), config.class_radius),
        "class_speeds": speeds,
    }
    return SeriesData(torch.cat(data), torch.cat(labels), trajectory, torch.cat(centres), info)


def _continuous_series(
    config: SeriesConfig,
    n_series: int,
    generator: torch.Generator,
    g0: torch.Tensor,
    g0_sqrtm: torch.Tensor,
    trajectory: torch.Tensor,
    directions: torch.Tensor,
) -> SeriesData:
    """Continuous target: y ~ U[-1, 1] per series, centres Exp_{G(t)}(r y e1(t)); directions (1 or T, d)."""
    p = config.n_features
    assert config.tangent_noise_std == 0 and not config.match_log_det, "not defined for a continuous target"
    assert isinstance(config.df, int), "one df for a continuous target"
    targets = 2 * torch.rand(n_series, dtype=g0.dtype, generator=generator) - 1
    coords = config.class_radius * targets[:, None, None] * directions  # (n, 1 or T, d) at G(0)
    tangent_g0 = g0_sqrtm @ sym_coordinates_to_matrix(coords, p) @ g0_sqrtm
    tangent = affine_invariant_parallel_transport(tangent_g0, g0, trajectory)
    centres = affine_invariant_exp(trajectory, tangent)  # (n, T, p, p)
    data = sample_wishart(centres, config.df, 1, generator, inverse=config.distribution == "inverse_wishart")[0]
    info = {"step_distance": config.trajectory_norm / (config.n_steps - 1), "class_radius": config.class_radius}
    return SeriesData(data, targets, trajectory, centres, info)


def validate_series(series: SeriesData, quantile: float = 0.95) -> dict:
    """
    Diagnostics of a generated series dataset (all distances are affine-invariant):

    - ``min_eigval``: smallest eigenvalue over all samples (SPD check)
    - ``max_cond_trajectory``: largest condition number of the G(t)
    - ``step_distance``: d(G(t), G(t+1)), constant along the geodesic
    - ``class_radius``: max_i d(G(t), G_i(t))
    - ``separation_ratio``: step_distance / class_radius (the spec asks for > 2)
    - ``dispersion_q``: per class, ``quantile`` of d(Sigma, G_i(t))
    - ``no_temporal_overlap``: step_distance > 2 * max dispersion_q, i.e. the
      samples of G_i(t) and G_i(t+1) do not overlap
    """
    data, labels, centres = series.data, series.labels, series.centres
    trajectory = series.trajectory
    assert "class_speeds" not in series.info, "validate_series checks the static scenario"
    n_classes = centres.shape[0]
    steps = affine_invariant_distance(trajectory[:-1], trajectory[1:])
    radius = affine_invariant_distance(trajectory[None], centres)
    dispersion = torch.stack(
        [
            torch.quantile(
                affine_invariant_distance(data[labels == i], centres[i]).flatten(),
                quantile,
            )
            for i in range(n_classes)
        ]
    )
    step_distance = steps.mean().item()
    return {
        "min_eigval": torch.linalg.eigvalsh(data).min().item(),
        "max_cond_trajectory": torch.linalg.cond(trajectory).max().item(),
        "step_distance": step_distance,
        "step_distance_spread": (steps.max() - steps.min()).item(),
        "class_radius": radius.max().item(),
        "separation_ratio": step_distance / radius.max().item(),
        "dispersion_q": dispersion.tolist(),
        "no_temporal_overlap": step_distance > 2 * dispersion.max().item(),
    }
