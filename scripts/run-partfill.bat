cd ..


@REM uv run python gs_fill_particles.py --model_path "F:/dataset/PhysGaussian-dataset/wolf_whitebg-trained/wolf_whitebg-trained" --config "config/wolf_config.json"

@REM uv run python gs_fill_particles.py --model_path "F:/dataset/PhysGaussian-dataset/ficus_whitebg-trained/ficus_whitebg-trained" --config "config/ficus_config.json"

@REM uv run python gs_fill_particles.py --model_path "F:/dataset/PhysGaussian-dataset/plane-trained/plane-trained" --config "config/plane_config.json"

@REM uv run python gs_fill_particles.py --model_path "F:/dataset/PhysGaussian-dataset/vasedeck_whitebg-trained/vasedeck_whitebg-trained" --config "config/vasedeck_config.json"

@REM uv run python gs_fill_particles.py --model_path "F:/dataset/PhysGaussian-dataset/bread-trained/bread-trained" --config "config/tear_bread_config.json"

uv run python gs_fill_particles.py --model_path "D:/GS-project/MultiLayer-3DGS/Multi-Layer-Anatomy-GS-Training/eval/example-man" --config "config/no_transform_config.json"

pause

