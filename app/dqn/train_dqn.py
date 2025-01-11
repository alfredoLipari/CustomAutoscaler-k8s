import torch
import matplotlib.pyplot as plt
from agent import DQNAgent
from test_environmnet import TestMicroserviceEnvironment
from datetime import datetime
import numpy as np
import logging
import json
import os
from collections import deque
from typing import List, Tuple

def moving_average(data, window):
    """Calculate moving average for smoothing"""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_metrics(
    time_steps: List[int], 
    request_rates: List[float], 
    predicted_rates: List[float], 
    latencies: List[float], 
    replicas: List[int], 
    loss_values: List[float], 
    is_training: bool
):
    """Plot metrics showing request rates, predicted rates, and replicas with dual y-axes."""
    time_in_seconds = np.array(time_steps) * 15
    pattern_duration = 2500  # 20 minutes
    steps_per_pattern = pattern_duration // 15
    total_patterns = len(time_steps) // steps_per_pattern

    # Select the last pattern for both subplots
    pattern_idx = total_patterns - 1
    start_idx = pattern_idx * steps_per_pattern
    end_idx = min((pattern_idx + 1) * steps_per_pattern, len(time_steps))

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(24, 20))
    axes = axes.flatten()  # Flatten for easy iteration

    # Subplot 1: Request and Predicted Rates
    ax1 = axes[0]
    ax1.plot(
        time_in_seconds[start_idx:end_idx], 
        request_rates[start_idx:end_idx], 
        label='Request Rate', color='blue', alpha=0.7
    )
    ax1.plot(
        time_in_seconds[start_idx:end_idx], 
        predicted_rates[start_idx:end_idx], 
        label='Predicted Rate', color='red', alpha=0.7
    )

    # Title and labels
    ax1.set_title('Request and Predicted Rates')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Rate (req/min)')

    # Legends
    ax1.legend(loc='upper right')

    # Grid
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Number of Replicas and Response Time
    ax2 = axes[1]
    ax2_2 = ax2.twinx()  # Create secondary y-axis for response time

    # Plot replicas on primary y-axis
    ax2.plot(
        time_in_seconds[start_idx:end_idx], 
        replicas[start_idx:end_idx], 
        label='Replicas', color='purple', linestyle='--', alpha=0.7
    )

    # Plot response time on secondary y-axis
    ax2_2.plot(
        time_in_seconds[start_idx:end_idx], 
        latencies[start_idx:end_idx], 
        label='Response Time', color='green', alpha=0.7
    )

    # Title and labels
    ax2.set_title('Replicas and Response Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Number of Replicas')
    ax2_2.set_ylabel('Response Time (s)')

    # Legends for both axes
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    # Grid
    ax2.grid(True, alpha=0.3)

    # Add summary statistics below the plots
    stats_text = (
        f"Overall Performance:\n"
        f"Avg Response Time: {np.mean(latencies):.3f}s\n"
        f"Avg Replicas: {np.mean(replicas):.2f}\n"
        f"Prediction RMSE: {np.sqrt(np.mean((np.array(predicted_rates) - np.array(request_rates))**2)):.2f}\n"
        f"Total SLO Violations: {sum(np.array(latencies) > 0.7)/len(latencies)*100:.1f}%\n"
        f"Avg Resource Efficiency: {np.mean(replicas)/10:.1f}%"
    )
    fig.text(
        0.5, 0.02, stats_text, 
        fontsize=12, 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'), 
        ha='center', 
        va='bottom'
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the summary text
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'{"training" if is_training else "evaluation"}_plot_{timestamp}.pdf')
    plt.close()


def setup_logging(name: str) -> Tuple[logging.Logger, str]:
    """Configure detailed logging with state space tracking"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler for detailed logging
    fh = logging.FileHandler(f'logs/{name}_{timestamp}.log')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    # Console handler for basic progress
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(ch)
    
    return logger, timestamp

def train_dqn_agent():
    # Setup logging
    logger, timestamp = setup_logging('Training')
    logger.info("Starting continuous training...")
    
    env = TestMicroserviceEnvironment(logger=logger)
    state = env.reset()
    print(f"State shape: {state.shape}") 
    state_size = state.shape[0]
    action_size = len(env.action_space)
    
    steps_per_pattern = 2500 // 15 # 166 steps per pattern due to longer duration, around 40 minutes
    total_patterns = 500 # Train for 500 patterns
    total_steps = steps_per_pattern * total_patterns
    # Since a step includes 15 substeps (15s) the total of steps would be 999,000 but the autosclaer will pass only 66,000 times
    agent = DQNAgent(state_size=state_size, action_size=action_size, total_episodes=total_steps, logger=logger)
    
    # Metrics storage
    time_steps = list(range(total_steps))
    request_rates = []
    predicted_rates = []
    latencies = []
    replicas = []
    rewards = []
    
    logger.info(f"Training Configuration:")
    logger.info(f"- Steps per pattern: {steps_per_pattern}")
    logger.info(f"- Total patterns: {total_patterns}")
    logger.info(f"- Total steps: {total_steps}")
    logger.info(f"- Simulated time: {total_steps * 15 / 3600:.1f} hours")

    patterns_completed = 0
    
    for step in range(total_steps):

        with torch.no_grad():
            q_values = agent.q_network(torch.FloatTensor(state).to(agent.device))
        action_index = agent.act(state)
        action = env.action_space[action_index]
        
        # Take action
        next_state, reward, _, response_time, arrival_rate = env.step(action)

        # Store metrics
        request_rates.append(arrival_rate)
        predicted_rates.append(env.current_prediction)
        latencies.append(response_time)
        replicas.append(env.current_replicas)
        rewards.append(reward)
        
        # Log detailed state information every 100 steps
        if step % steps_per_pattern == 0 and step > 0:
            patterns_completed += 1
            
            # Log pattern completion metrics
            logger.info(f"\nPattern {patterns_completed} Summary:")
            logger.info(f"- Current State: CPU={state[0]:.3f}, Rate={state[1]:.3f}, RT={state[2]:.3f}, Replicas={state[3]:.3f}")
            logger.info(f"- Action Taken: {action} (Q-values: {q_values.cpu().numpy()})")
            logger.info(f"- Pattern Metrics:")
            logger.info(f"  * Avg Response Time: {np.mean(latencies[-steps_per_pattern:]):.3f}s")
            logger.info(f"  * Avg Replicas: {np.mean(replicas[-steps_per_pattern:]):.2f}")     
        
        # Agent learns from experience
        agent.step(state, action_index, reward, next_state, False)
        
        state = next_state
    
    # Log final statistics
    logger.info("\nTraining Completed!")
    logger.info(f"Final Statistics:")
    logger.info(f"Average Response Time: {np.mean(latencies):.3f}s")
    logger.info(f"Average Replicas: {np.mean(replicas):.2f}")
    logger.info(f"Average Reward: {np.mean(rewards):.2f}")
    
    # Save models with same timestamp
    model_path = f'dqn_model_{timestamp}.pth'
    torch.save({
        'dqn_state_dict': agent.q_network.state_dict(),
        'dqn_target_state_dict': agent.target_network.state_dict(),
        'dqn_optimizer_state_dict': agent.optimizer.state_dict(),
        'training_metrics': {
            'request_rates': request_rates,
            'predicted_rates': predicted_rates,
            'latencies': latencies,
            'replicas': replicas,
            'rewards': rewards
        }
    }, model_path)
    logger.info(f"DQN model saved to {model_path}")
    
    lstm_path = f'lstm_model_{timestamp}.pth'
    torch.save({
        'lstm_state_dict': env.lstm_predictor.state_dict(),
        'lstm_optimizer_state_dict': env.optimizer.state_dict(),
        'prediction_window': env.prediction_window,
        'rate_history': list(env.lstm_utils.rate_history),
        'step_counter': env.step_counter
    }, lstm_path)
    logger.info(f"LSTM model saved to {lstm_path}")

    loss_values = env.lstm_utils.loss_values
    
    # Plot and save results
    plot_metrics(time_steps, request_rates, predicted_rates, latencies, replicas,  loss_values, is_training=True)

    rate_history = list(env.lstm_utils.rate_history)
    training_data = {
        'timestamp': timestamp,
        'rate_history': rate_history
    }

    with open(f'training_history_{timestamp}.json', 'w') as f:
        json.dump(training_data, f)

    
    return timestamp  # Return timestamp for evaluation

def evaluate_agent(timestamp: str = None):
    
    # Load latest models if timestamp not provided
    if timestamp is None:
        model_files = [f for f in os.listdir('.') if f.startswith('dqn_model_')]
        latest_model = max(model_files)
        latest_lstm = f'models/lstm_model_{latest_model.split("dqn_model_")[1]}'
    else:
        latest_model = f'models/dqn_model_{timestamp}.pth'
        latest_lstm = f'models/lstm_model_{timestamp}.pth'

    # Setup logging
    logger, eval_timestamp = setup_logging('Evaluation')
    logger.info("Starting evaluation...")

    lstm_checkpoint = torch.load(latest_lstm)

    # Load and setup environment
    env = TestMicroserviceEnvironment(logger=logger)
    env.lstm_predictor.load_state_dict(lstm_checkpoint['lstm_state_dict'])
    env.optimizer.load_state_dict(lstm_checkpoint['lstm_optimizer_state_dict'])
    
    # Restore LSTM history
    # Restore critical LSTM state
    if 'rate_history' in lstm_checkpoint:
        rate_history_list = lstm_checkpoint['rate_history']
        # Make sure we have data
        if len(rate_history_list) > 0:
            env.lstm_utils.rate_history = deque(rate_history_list, maxlen=env.history_length)
            env.rate_history = deque(rate_history_list[-env.history_length:], maxlen=env.history_length)
        else:
            # If no history, initialize with a reasonable starting value
            initial_rate = env.base_rate  # Use base rate as starting point
            env.rate_history.append(initial_rate)
            env.lstm_utils.rate_history.append(initial_rate)

    logger.info(f"Restored rate history with {len(env.lstm_utils.rate_history)} entries")
    
    # Evaluation parameters
    steps_per_pattern = 2500 // 15
    eval_patterns = 1000
    eval_steps = steps_per_pattern * eval_patterns

    # Calculate first metrics - with safety checks
    if len(env.rate_history) > 0:
        logger.info(f"rate history is populated")
        instant_rate = env.rate_history[-1]
    else:
        logger.info(f"fallback rate history is not populated")
        instant_rate = env.base_rate  # Fallback to base rate if history is empty
        env.rate_history.append(instant_rate)  # Initialize history

    # calculate first cpu utilization and response time
    current_replicas = env.min_replicas
    cpu_utilization = env.get_cpu_utilization(instant_rate, current_replicas)
    response_time = env.get_response_time(instant_rate, current_replicas)
    env.cpu_util_history.append(cpu_utilization)
    env.response_time_history.append(response_time)
    
    # Initialize and load DQN agent
    state = env.reset()  # Here do not reset the environment
    state_size = state.shape[0]
    action_size = len(env.action_space)
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, total_episodes=eval_steps, logger=logger)
    dqn_checkpoint = torch.load(latest_model)
    agent.q_network.load_state_dict(dqn_checkpoint['dqn_state_dict'])
    agent.target_network.load_state_dict(dqn_checkpoint['dqn_target_state_dict'])
    agent.epsilon = 0.0  # No exploration during evaluation
    
    # Metrics storage
    request_rates = []
    predicted_rates = []
    latencies = []
    replicas = []
    rewards = []
    slo_violations = 0
    pattern_violations = 0
    
    logger.info(f"Starting evaluation for {eval_steps} steps")
    
    # Evaluation loop
    for step in range(eval_steps):
        # Get prediction and take action
        action_index = agent.act(state)
        action = env.action_space[action_index]
        
        # Take action and observe result
        next_state, reward, done, response_time, arrival_rate = env.step(action)
        
        # Store metrics
        request_rates.append(arrival_rate)
        predicted_rates.append(env.current_prediction)
        latencies.append(response_time)
        replicas.append(env.current_replicas)
        rewards.append(reward)
        
        # Track SLO violations
        if response_time > env.Tmax:
            slo_violations += 1
            pattern_violations += 1

        # Continue updating LSTM during evaluation for online learning
        loss_values = env.lstm_utils.loss_values
        
        state = next_state
        
        # Log pattern summary
        if (step + 1) % steps_per_pattern == 0:
            pattern_metrics = {
                'avg_rt': np.mean(latencies[-steps_per_pattern:]),
                'avg_replicas': np.mean(replicas[-steps_per_pattern:]),
                'prediction_error': np.mean(np.abs(np.array(predicted_rates[-steps_per_pattern:]) - 
                                                 np.array(request_rates[-steps_per_pattern:])))
            }
            logger.info(f"\nPattern Summary:")
            logger.info(f"Avg Response Time: {pattern_metrics['avg_rt']:.3f}s")
            logger.info(f"Avg Replicas: {pattern_metrics['avg_replicas']:.2f}")
            logger.info(f"Avg Prediction Error: {pattern_metrics['prediction_error']:.2f}")
    
    # Plot evaluation results
    time_steps = list(range(eval_steps))
    plot_metrics(time_steps, request_rates, predicted_rates, latencies, replicas, loss_values, is_training=False)

    # Log final statistics
    logger.info("\nEvaluation Completed!")
    logger.info(f"Final Statistics:")
    logger.info(f"Average Response Time: {np.mean(latencies):.3f}s")
    logger.info(f"Average Replicas: {np.mean(replicas):.2f}")
    logger.info(f"Total SLO Violations: {slo_violations}")
    logger.info(f"SLO Rate: {(slo_violations/eval_steps)*100:.1f}%")
    logger.info(f"Average Reward: {np.mean(rewards):.2f}")
    logger.info(f"Total Pattern Violations: {pattern_violations}")
    
    return eval_timestamp

if __name__ == "__main__":
    # refresh cuda cache
    torch.cuda.empty_cache()
    print("Starting training at:", datetime.now())
    #timestamp = train_dqn_agent()
    print("\nStarting evaluation...")
    evaluate_agent("no_predictor")