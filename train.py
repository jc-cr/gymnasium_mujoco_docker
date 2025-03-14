#!/usr/bin/env python3
"""
Training script for Inverted Pendulum using Stable Baselines3
"""
import os
import time
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    
    set_random_seed(seed)
    return _init


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train an agent for InvertedPendulum-v5')
    parser.add_argument('--timesteps', type=int, default=1000000, 
                        help='Number of timesteps to train for')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--num-envs', type=int, default=8, 
                        help='Number of parallel environments')
    parser.add_argument('--save-dir', type=str, default='./models', 
                        help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='./logs', 
                        help='Directory to save logs')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create environments
    env_id = "InvertedPendulum-v5"
    
    # Create vectorized environment
    if args.num_envs == 1:
        env = DummyVecEnv([make_env(env_id, 0, args.seed)])
    else:
        env = SubprocVecEnv([make_env(env_id, i, args.seed) for i in range(args.num_envs)])
    
    # Create separate environment for evaluation
    eval_env = DummyVecEnv([make_env(env_id, 0, args.seed + 1000)])
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=args.save_dir,
        name_prefix='ppo_inverted_pendulum'
    )
    
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=f"{args.save_dir}/best",
        log_path=args.log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Set up model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=args.log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=args.seed
    )
    
    # Train the model
    print(f"Starting training for {args.timesteps} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="ppo_inverted_pendulum"
    )
    
    # Save the final model
    final_model_path = os.path.join(args.save_dir, f"final_model_{int(time.time())}")
    model.save(final_model_path)
    print(f"Training completed in {time.time() - start_time:.2f}s")
    print(f"Final model saved to {final_model_path}")
    
    # Close environments
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()