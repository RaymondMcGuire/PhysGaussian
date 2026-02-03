import sys

sys.path.append("gaussian-splatting")

import argparse
import os
import torch
import taichi as ti

from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration

from particle_filling.filling import fill_particles
from utils.decode_param import decode_param_json
from utils.transformation_utils import (
    apply_cov_rotations,
    apply_rotations,
    generate_rotation_matrices,
    shift2center111,
    transform2origin,
)
from utils.render_utils import load_params_from_gs
from mpm_solver_warp.engine_utils import particle_position_tensor_to_ply


class PipelineParamsNoparse:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_ply", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise AssertionError("Model path does not exist!")
    if not os.path.exists(args.config):
        raise AssertionError("Scene config does not exist!")

    if args.output_ply is None:
        config_base = os.path.splitext(os.path.basename(args.config))[0]
        args.output_ply = os.path.join("./log", f"{config_base}_filled_particles.ply")

    output_dir = os.path.dirname(args.output_ply)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ti.init(arch=ti.cuda, device_memory_GB=8.0)

    print("Loading scene config...")
    (
        material_params,
        _bc_params,
        _time_params,
        preprocessing_params,
        _camera_params,
    ) = decode_param_json(args.config)

    print("Loading gaussians...")
    gaussians = load_checkpoint(args.model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_opacity = params["opacity"]

    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]

    if args.debug:
        particle_position_tensor_to_ply(init_pos, "./log/init_particles.ply")

    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    if args.debug:
        particle_position_tensor_to_ply(rotated_pos, "./log/rotated_particles.ply")

    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]

    if preprocessing_params.get("disable_transform", False):
        transformed_pos = rotated_pos
        init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    else:
        transformed_pos, scale_origin, original_mean_pos = transform2origin(
            rotated_pos, preprocessing_params["scale"]
        )
        transformed_pos = shift2center111(transformed_pos)

        init_cov = apply_cov_rotations(init_cov, rotation_matrices)
        init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(
            transformed_pos, "./log/transformed_particles.ply"
        )

    filling_params = preprocessing_params["particle_filling"]
    if filling_params is None:
        raise AssertionError("particle_filling is disabled in config.")

    print("Filling internal particles...")
    device = "cuda:0"
    mpm_init_pos = fill_particles(
        pos=transformed_pos,
        opacity=init_opacity,
        cov=init_cov,
        grid_n=filling_params["n_grid"],
        max_samples=filling_params["max_particles_num"],
        grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
        density_thres=filling_params["density_threshold"],
        search_thres=filling_params["search_threshold"],
        max_particles_per_cell=filling_params["max_partciels_per_cell"],
        search_exclude_dir=filling_params["search_exclude_direction"],
        ray_cast_dir=filling_params["ray_cast_direction"],
        boundary=filling_params["boundary"],
        smooth=filling_params["smooth"],
    ).to(device=device)

    particle_position_tensor_to_ply(mpm_init_pos, args.output_ply)
    print(f"Saved filled particles to: {args.output_ply}")


if __name__ == "__main__":
    main()
