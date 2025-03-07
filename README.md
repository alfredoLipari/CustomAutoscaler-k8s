# CustomAutoscaler-k8s

A Reinforcement Learning-based autoscaler for Kubernetes that combines Double Deep Q-Networks (DQN) with Long Short-Term Memory (LSTM) networks for proactive workload prediction and intelligent scaling decisions.

## Overview

This project implements the autoscaler described in the paper "Reinforcement Learning-Based Autoscaler with LSTM Workload Prediction", which outperforms standard Kubernetes Horizontal Pod Autoscaler (HPA) by:

1. Proactively predicting future workloads using an LSTM network
2. Making intelligent scaling decisions using a DQN agent
3. Optimizing for both response time (SLO) and resource efficiency

The implementation consists of two main phases:
- A simulation environment for training and evaluating the autoscaler
- A deployment system for running the autoscaler in real Kubernetes clusters

## Architecture

### Core Components

1. **DQN Agent** (`agent.py`, `network.py`, `prioritized_replay_buffer.py`)
   - Implements a Double Deep Q-Network with prioritized experience replay
   - Makes scaling decisions based on current state and predicted workload
   - Trained to balance performance objectives and resource efficiency

2. **LSTM Predictor** (`predictor_network.py`, `predictor_utils.py`)
   - Forecasts future request rates based on historical patterns
   - Enables proactive scaling before workload changes occur
   - Combines model predictions with trend analysis for better accuracy

3. **Simulation Environment** (`test_environment.py`)
   - Generates realistic workload patterns for training
   - Simulates service metrics (CPU usage, response time) based on queuing theory
   - Provides reward signals to guide agent learning

4. **Kubernetes Integration** (`environment.py`, `autoscaler.py`, `prom_query.py`)
   - Collects metrics from Prometheus and Kubernetes API
   - Applies scaling decisions to real Kubernetes deployments
   - Handles production deployment considerations

5. **Workload Generator** (`workload_simulator.py`)
   - Creates realistic request patterns for testing in real environments
   - Supports customizable traffic patterns with sinusoidal waves

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- Kubernetes cluster (for deployment phase)
- Prometheus setup for metrics collection

```bash
# Install dependencies
pip install -r requirements.txt
```

## Training the Autoscaler

The training process consists of two steps:

1. **Training the DQN Agent and LSTM Predictor in Simulation**

```bash
# Start the training process
python train_dqn.py
```

This script:
- Initializes the simulation environment
- Creates and trains both the DQN agent and LSTM predictor
- Saves model checkpoints with timestamps
- Logs training progress and metrics
- Generates performance visualization plots

The training settings in `train_dqn.py` include:
- 500 unique workload patterns
- Each pattern has 166 steps (15 seconds per step, ~40 minutes of simulated time)
- DQN agent uses a 5-dimensional state vector (CPU, request rate, replicas, predicted rate, rate change)

### Training Output

After training completes, you'll find:
- DQN model: `dqn_model_TIMESTAMP.pth`
- LSTM model: `lstm_model_TIMESTAMP.pth`
- Training logs: `Training_TIMESTAMP.log`
- Performance plots: `training_plot_TIMESTAMP.pdf`

## Evaluating the Autoscaler

```bash
# Evaluate the trained models
python train_dqn.py  # Set evaluate_agent() instead of train_dqn_agent()
```

The evaluation:
- Tests the models on 1,000 new workload patterns
- Measures response times, SLO violations, and resource usage
- Generates evaluation plots and metrics

## Deploying to Kubernetes

### 1. Prepare Your Environment

- Ensure you have a running Kubernetes cluster
- Set up Prometheus for metrics collection
- Deploy a sample application (like the CPU-intensive microservice)

### 2. Configure and Run the Autoscaler

```bash
# Edit the autoscaler parameters in autoscaler.py
# then run:
python autoscaler.py
```

Configuration parameters in `autoscaler.py`:
- `deployment_name`: Name of the deployment to scale
- `namespace`: Kubernetes namespace
- `service_url`: URL of the service for latency measurement
- `dqn_model_path`: Path to the trained DQN model
- `lstm_model_path`: Path to the trained LSTM model

### 3. Generate Test Workload

```bash
# Run the workload generator to create test traffic
python workload_simulator.py
```

## Understanding the Code

### State Representation

The DQN agent uses a 5-dimensional state vector:
1. Normalized CPU Utilization (0-1)
2. Normalized Request Rate (0-1)
3. Normalized Number of Replicas (0-1)
4. Normalized Predicted Request Rate (0-1)
5. Rate Change (-1 to 1)

### Reward Function

The reward function is designed to balance three objectives:
1. **Performance Cost**: Penalizes high response times, with a sharp increase for SLO violations
2. **Resource Cost**: Combines linear and exponential components to discourage over-provisioning
3. **Action Alignment**: Encourages stable operation by penalizing unnecessary changes

### LSTM Prediction

The LSTM predictor:
- Takes sequences of historical request rates
- Predicts the next step request rate
- Blends model predictions with trend analysis (70% model, 30% trend)
- Updates continuously during operation for online learning

## Customization

### Training Parameters

Modify `train_dqn.py` to adjust:
- Number of training patterns
- Steps per pattern
- DQN hyperparameters (learning rate, discount factor, etc.)
- LSTM architecture and training settings

### Simulation Environment

Customize `test_environment.py` to change:
- SLO thresholds (Tmax, Tmin)
- Scaling parameters (min/max replicas)
- Service rate per replica
- Workload generation patterns

### Production Deployment

Adjust `autoscaler.py` for:
- Metrics collection interval
- Scaling cooldown period
- Kubernetes API integration settings

## Monitoring

The autoscaler logs detailed information about:
- Current metrics (CPU, request rate, response time)
- State representations
- Q-values for each action
- Scaling decisions and their effects
- Prediction accuracy

Monitor logs with:

```bash
tail -f autoscaler_TIMESTAMP.log
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure PyTorch versions match between training and deployment
   - Check that model architecture hasn't changed

2. **Kubernetes API Issues**
   - Verify RBAC permissions for the autoscaler
   - Check cluster connectivity and authentication

3. **Metrics Collection Failures**
   - Ensure Prometheus is properly configured
   - Verify service endpoints are accessible

## Advanced Configuration

### Fine-tuning the Models

1. **DQN Agent Tuning**
   - Adjust reward function parameters in `test_environment.py`
   - Modify neural network architecture in `network.py`
   - Change prioritized replay buffer parameters

2. **LSTM Predictor Tuning**
   - Adjust sequence length in `predictor_utils.py`
   - Modify LSTM architecture in `predictor_network.py`
   - Change the prediction horizon and blending ratio

### Custom Metrics

The system can be extended to use additional metrics:
1. Add new metrics to the state representation
2. Update the reward function to consider these metrics
3. Integrate with your metrics collection system

## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite the original paper:

```
@article{rl_autoscaler_2025,
  title={Reinforcement Learning-Based Autoscaler with LSTM Workload Prediction},
  author={Lipari, Alfredo},
  journal={IEEE/ACM International Conference on Utility and Cloud Computing},
  year={2025}
}
```

## Acknowledgments

This project builds upon research in reinforcement learning for cloud resource management and extends previous work on autoscaling for containerized applications.
