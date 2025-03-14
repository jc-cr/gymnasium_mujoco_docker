# gymnasium_mujoco_docker

# Gymnasium MuJoCo Docker

A Docker-based environment for training and evaluating the Inverted Pendulum MuJoCo environment from Gymnasium. This project supports both CPU and GPU training using NVIDIA CUDA.

## Project Structure

```
.
├── .docker
│   ├── docker-compose.yaml
│   ├── Dockerfile
│   └── req.txt
├── eval.py
├── LICENSE
├── README.md
└── train.py
```

## Requirements

- Docker
- Docker Compose
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) (for GPU support, optional)

## Quick Start

### Building the Container

```bash
cd .docker
docker-compose build
```

### Training the Agent

To train the agent using default parameters:

```bash
cd .docker
docker-compose run mujoco python3 train.py
```

With custom parameters:

```bash
cd .docker
docker-compose run mujoco python3 train.py --timesteps 2000000 --num-envs 4
```

### Evaluating the Trained Agent

After training, you can evaluate the agent:

```bash
cd .docker
docker-compose run mujoco python3 eval.py --model-path ./models/best/best_model.zip --num-episodes 10
```


## Training Options

The `train.py` script accepts the following command-line arguments:

- `--timesteps`: Number of timesteps to train for (default: 1000000)
- `--seed`: Random seed (default: 42)
- `--num-envs`: Number of parallel environments (default: 8)
- `--save-dir`: Directory to save models (default: ./models)
- `--log-dir`: Directory to save logs (default: ./logs)

## Evaluation Options

The `eval.py` script accepts the following command-line arguments:

- `--model-path`: Path to the trained model (required)
- `--num-episodes`: Number of episodes to evaluate (default: 10)
- `--render`: Render the environment during evaluation
- `--save-video`: Save a video of the evaluation

## Using GPU Acceleration

This Docker setup is configured to work on machines without GPU by default. If you have an NVIDIA GPU and want to use it for training:

1. Install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
2. In `.docker/docker-compose.yaml`, uncomment the `deploy` section
3. In `.docker/Dockerfile`, replace `FROM ubuntu:22.04` with:
   ```dockerfile
   FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
   ```
4. Change the `MUJOCO_GL` environment variable from `osmesa` to `egl` in both Dockerfile and docker-compose.yaml
5. Rebuild the container:
   ```bash
   cd .docker
   docker-compose build
   ```

## Environment Details

This Docker environment uses:
- Python 3.12 (from the official python:3.12-slim-bookworm image)
- Gymnasium with MuJoCo support
- Stable Baselines3 for reinforcement learning algorithms
- Software rendering (osmesa) for environments without GPU

## Customization

To use different MuJoCo environments, simply modify the `env_id` variable in the training and evaluation scripts.

## License

This project is licensed under the MIT License - see the LICENSE file for details.