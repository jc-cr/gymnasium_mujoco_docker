#!/usr/bin/env python3
"""
Evaluation script for Inverted Pendulum using Stable Baselines3
"""
import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import imageio


def evaluate_policy(model, env, num_episodes=10, render=False, save_video=False):
    """
    Evaluate a trained model on given environment for multiple episodes
    
    :param model: Trained model
    :param env: Environment to evaluate on
    :param num_episodes: Number of episodes to evaluate
    :param render: Whether to render the environment
    :param save_video: Whether to save video of the evaluation
    :return: Mean reward and standard deviation
    """
    episode_rewards = []
    episode_lengths = []
    
    # Set up video recording if required
    video_frames = []
    
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_step = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_step += 1
            
            if render:
                frame = env.render()
                if save_video and frame is not None:
                    video_frames.append(frame)
                    
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step)
        print(f"Episode {i+1}: Reward = {episode_reward}, Length = {episode_step}")
    
    # Save video if required
    if save_video and video_frames:
        video_path = 'evaluation_video.mp4'
        imageio.mimsave(video_path, video_frames, fps=30)
        print(f"Evaluation video saved to {video_path}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean episode length: {mean_length:.2f}")
    
    return mean_reward, std_reward


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate a trained agent for InvertedPendulum-v5')
    parser.add_argument('--model-path', type=str, required=True, 
                        help='Path to the trained model')
    parser.add_argument('--num-episodes', type=int, default=10, 
                        help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true', 
                        help='Render the environment during evaluation')
    parser.add_argument('--save-video', action='store_true', 
                        help='Save a video of the evaluation')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}")
    
    # Create environment
    env = gym.make("InvertedPendulum-v5", render_mode="rgb_array" if args.render or args.save_video else None)
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = PPO.load(args.model_path)
    
    # Evaluate the model
    evaluate_policy(model, env, args.num_episodes, args.render, args.save_video)
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()