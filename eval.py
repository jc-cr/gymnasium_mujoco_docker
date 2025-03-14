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
        # Get the environment to manually set state
        if hasattr(env, 'unwrapped'):
            unwrapped_env = env.unwrapped
        else:
            unwrapped_env = env
            
        # Create extremely challenging initial states
        if i % 4 == 0:
            # Pole nearly falling to the right
            angle = np.random.uniform(0.18, 0.19)  # Nearly at the failure threshold of 0.2
            cart_pos = np.random.uniform(-0.5, 0.5)
            print(f"Starting episode {i+1} with pole NEARLY FALLING right ({angle:.3f} radians)")
        elif i % 4 == 1:
            # Pole nearly falling to the left
            angle = np.random.uniform(-0.19, -0.18)  # Nearly at the failure threshold of -0.2
            cart_pos = np.random.uniform(-0.5, 0.5)
            print(f"Starting episode {i+1} with pole NEARLY FALLING left ({angle:.3f} radians)")
        elif i % 4 == 2:
            # Pole at medium angle with high velocity
            angle = np.random.uniform(-0.15, 0.15)
            cart_pos = np.random.uniform(-0.5, 0.5)
            print(f"Starting episode {i+1} with medium tilt and HIGH VELOCITY")
        else:
            # Rapid state changes - start, then after 5 steps, apply a large impulse
            angle = 0.0
            cart_pos = 0.0
            print(f"Starting episode {i+1} normally, but will apply DISTURBANCE after 5 steps")
            
        # Set the state if possible
        try:
            # Reset first to ensure environment is in valid state
            obs, _ = env.reset()
            
            # qpos = [cart_position, pole_angle]
            # qvel = [cart_velocity, pole_angular_velocity]
            qpos = np.array([cart_pos, angle])
            
            if i % 4 == 2:
                # High angular velocity
                pole_vel = np.random.uniform(0.5, 1.0) * (1 if np.random.random() > 0.5 else -1)
                cart_vel = np.random.uniform(0.5, 1.0) * (1 if np.random.random() > 0.5 else -1)
                qvel = np.array([cart_vel, pole_vel])
            else:
                qvel = np.array([0.0, 0.0])
                
            unwrapped_env.set_state(qpos, qvel)
            obs = unwrapped_env._get_obs()
        except (AttributeError, NotImplementedError) as e:
            print(f"Could not set custom state: {e}. Using default reset.")
            obs, _ = env.reset()
        
        done = False
        truncated = False
        episode_reward = 0
        episode_step = 0
        disturbance_applied = False
        
        while not (done or truncated):
            # Apply a disturbance after 5 steps for the 4th case
            if i % 4 == 3 and episode_step == 5 and not disturbance_applied:
                try:
                    # Get current state
                    qpos = unwrapped_env.data.qpos.copy()
                    qvel = unwrapped_env.data.qvel.copy()
                    
                    # Apply an extreme angular velocity to the pole
                    qvel[1] = 1.5 * (1 if np.random.random() > 0.5 else -1)  # Angular velocity impulse
                    qvel[0] = 1.0 * (1 if np.random.random() > 0.5 else -1)  # Cart velocity impulse
                    
                    # Set the disturbed state
                    unwrapped_env.set_state(qpos, qvel)
                    obs = unwrapped_env._get_obs()
                    disturbance_applied = True
                    print("  DISTURBANCE APPLIED!")
                except (AttributeError, NotImplementedError) as e:
                    print(f"  Could not apply disturbance: {e}")
            
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
        try:
            # Create directory for videos if it doesn't exist
            import os
            os.makedirs("./videos", exist_ok=True)
            
            video_path = './videos/extreme_recovery_video.mp4'
            print(f"Saving video to {video_path}...")
            import imageio
            imageio.mimsave(video_path, video_frames, fps=30)
            print(f"Video saved successfully!")
        except Exception as e:
            print(f"Error saving video: {e}")
    
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
    parser.add_argument('--num-episodes', type=int, default=5, 
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