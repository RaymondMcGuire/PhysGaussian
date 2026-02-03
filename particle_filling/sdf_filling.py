import numpy as np
import torch
import taichi as ti

from .filling import densify_grids

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - optional fallback
    ndimage = None


def _binary_fill_holes_3d(mask: np.ndarray) -> np.ndarray:
    if ndimage is not None:
        return ndimage.binary_fill_holes(mask)

    # Fallback: flood fill outside on inverted mask, then invert.
    filled = mask.copy()
    visited = np.zeros_like(filled, dtype=bool)
    stack = []
    nx, ny, nz = filled.shape
    for i in range(nx):
        for j in range(ny):
            for k in (0, nz - 1):
                if not filled[i, j, k]:
                    stack.append((i, j, k))
    for i in range(nx):
        for k in range(nz):
            for j in (0, ny - 1):
                if not filled[i, j, k]:
                    stack.append((i, j, k))
    for j in range(ny):
        for k in range(nz):
            for i in (0, nx - 1):
                if not filled[i, j, k]:
                    stack.append((i, j, k))

    while stack:
        x, y, z = stack.pop()
        if visited[x, y, z]:
            continue
        visited[x, y, z] = True
        for dx, dy, dz in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ):
            nx2, ny2, nz2 = x + dx, y + dy, z + dz
            if (
                0 <= nx2 < nx
                and 0 <= ny2 < ny
                and 0 <= nz2 < nz
                and not filled[nx2, ny2, nz2]
                and not visited[nx2, ny2, nz2]
            ):
                stack.append((nx2, ny2, nz2))

    outside = visited
    return np.logical_not(outside)


def fill_particles_sdf(
    pos: torch.Tensor,
    opacity: torch.Tensor,
    cov: torch.Tensor,
    grid_n: int,
    grid_dx: float,
    density_thres: float,
    max_samples: int,
    boundary: list = None,
    smooth: bool = False,
    fill_per_cell: int = 1,
    jitter: float = 0.0,
    include_original: bool = True,
):
    pos_clone = pos.clone()
    if boundary is not None:
        assert len(boundary) == 6
        mask = torch.ones(pos_clone.shape[0], dtype=torch.bool).cuda()
        max_diff = 0.0
        for i in range(3):
            mask = torch.logical_and(mask, pos_clone[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, pos_clone[:, i] < boundary[2 * i + 1])
            max_diff = max(max_diff, boundary[2 * i + 1] - boundary[2 * i])

        pos = pos[mask]
        opacity = opacity[mask]
        cov = cov[mask]

        if grid_dx is None or grid_dx <= 0:
            grid_dx = max_diff / grid_n
        new_origin = torch.tensor([boundary[0], boundary[2], boundary[4]]).cuda()
        pos = pos - new_origin
    else:
        new_origin = None

    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_opacity = ti.field(dtype=float, shape=opacity.shape[0])
    ti_cov = ti.Vector.field(n=6, dtype=float, shape=cov.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))
    ti_opacity.from_torch(opacity.reshape(-1))
    ti_cov.from_torch(cov.reshape(-1, 6))

    grid = ti.field(dtype=int, shape=(grid_n, grid_n, grid_n))
    grid_density = ti.field(dtype=float, shape=(grid_n, grid_n, grid_n))

    densify_grids(ti_pos, ti_opacity, ti_cov, grid, grid_density, grid_dx)
    df = grid_density.to_numpy()
    if smooth:
        import mcubes

        df = mcubes.smooth(df, method="constrained", max_iters=500).astype(np.float32)

    occ = df > float(density_thres)
    filled = _binary_fill_holes_3d(occ)
    inside = np.logical_and(filled, np.logical_not(occ))

    coords = np.argwhere(inside)
    if coords.shape[0] == 0:
        particles_tensor = pos_clone if include_original else pos.clone()
        return particles_tensor

    if max_samples is not None and coords.shape[0] * fill_per_cell > max_samples:
        max_cells = max_samples // max(fill_per_cell, 1)
        coords = coords[:max_cells]

    samples = []
    for c in coords:
        base = (c.astype(np.float32) + 0.5) * grid_dx
        for _ in range(max(fill_per_cell, 1)):
            if jitter > 0.0:
                offset = (np.random.rand(3).astype(np.float32) - 0.5) * jitter * grid_dx
                samples.append(base + offset)
            else:
                samples.append(base.copy())

    particles = torch.tensor(np.stack(samples, axis=0), device=pos.device)
    if new_origin is not None:
        particles = particles + new_origin

    if include_original:
        return torch.cat([pos_clone, particles], dim=0)
    return particles
