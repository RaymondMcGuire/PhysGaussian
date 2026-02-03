import sys

sys.path.append("gaussian-splatting")

import argparse
import json
import os

import torch
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration


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
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--padding", type=float, default=0.0)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise AssertionError("Model path does not exist!")
    if not os.path.exists(args.config):
        raise AssertionError("Config path does not exist!")

    gaussians = load_checkpoint(args.model_path, iteration=args.iteration)
    pos = gaussians.get_xyz
    if pos.is_cuda:
        pos = pos.detach().cpu()

    min_xyz = torch.min(pos, dim=0).values
    max_xyz = torch.max(pos, dim=0).values
    pad = args.padding

    boundary = [
        float(min_xyz[0] - pad),
        float(max_xyz[0] + pad),
        float(min_xyz[1] - pad),
        float(max_xyz[1] + pad),
        float(min_xyz[2] - pad),
        float(max_xyz[2] + pad),
    ]

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "particle_filling" in cfg and cfg["particle_filling"] is not None:
        cfg["particle_filling"]["boundary"] = boundary
    elif "particle_filling_sdf" in cfg and cfg["particle_filling_sdf"] is not None:
        cfg["particle_filling_sdf"]["boundary"] = boundary
    else:
        raise AssertionError("particle_filling or particle_filling_sdf is missing in config.")

    with open(args.config, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")

    print("Updated boundary:")
    print(boundary)


if __name__ == "__main__":
    main()
