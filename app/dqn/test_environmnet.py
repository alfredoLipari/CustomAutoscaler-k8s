# test_environment.py
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math
import torch
from predictor_network import LSTMPredictor
from collections import deque
from predictor_utils import LSTMUtils
import random

# Set fixed random seeds
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

@dataclass
class CycleParameters:
    amplitude_mod: float
    phase_shift: float
    secondary_waves: List[Dict[str, float]]

class TestMicroserviceEnvironment:
    def __init__(
        self,
        Tmax: float = 0.7,
        N_max: int = 13,
        mu: float = 0.8,  # Each replica can handle 0.8 req/sec
        history_length: int = 100,  # Store 100 intervals (25 minutes of data)
        collection_interval: int = 15,  # 15 seconds between measurements
        logger = None,  # Add logger parameter
        initial_history = None,  # Add initial history parameter
    ):
        # Service Level Objectives
        self.Tmax = Tmax
        self.Tmin = 0.2
        
        # Scaling parameters
        self.N_max = N_max
        self.min_replicas = 1
        self.mu = mu
        self.action_space = [ -1, 0, 1]

        # Logger
        self.logger = logger
        
        # Time parameters
        self.collection_interval = collection_interval
        self.trend_duration = 1200  # seconds, matching JMeter that is 20 minutes per pattern
        self.step_counter = 0  # Counts individual seconds
        self.scale_cooldown = 60  # 1 minutes cooldown
        self.last_scale_time = 0
        
        # State tracking
        self.current_replicas = self.min_replicas
        self.arrival_rate = 0.0
        
        # History tracking (stores 15s aggregated values)
        self.history_length = history_length
        self.rate_history = deque(maxlen=history_length)
        self.response_time_history = deque(maxlen=history_length)
        self.cpu_util_history = deque(maxlen=history_length)
        
        # Workload generation parameters
        self._init_workload_params()
        
        # LSTM components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm_predictor = LSTMPredictor().to(self.device)
        self.optimizer = torch.optim.Adam(self.lstm_predictor.parameters(), lr=0.001)
        self.prediction_window = 10  # Now represents 150s of data (10 intervals)
        self.lstm_utils = LSTMUtils(self)

        self.is_hpa_simulation = False

        self.prediction_buffer = deque() # uffer to hold predictions along with their target times.
        self.current_prediction = self.arrival_rate  # Store the prediction being currently used
        self.current_rate_change = 0.0  # Store the current rate change for reward calculation
        
    def _init_workload_params(self):
        """Initialize workload generation parameters"""
        self.base_rate = 80.0    # 80 req/min
        self.min_rate = 3.0      # 6 req/min
        self.max_rate = 260.0    # 300 req/min
        self.amplitude = 163.0    # Amplitude in req/min
        
        self.smoothing_factor = 0.95
        self.random_walk = 0.0
        self.previous_rate = None
        
        # Cycle parameters
        self.cycle_variations = {}

    def _generate_cycle_parameters(self) -> CycleParameters:
        return CycleParameters(
            amplitude_mod=np.random.uniform(1.0, 1.5),  
            phase_shift=np.random.uniform(-0.3, 0.3),   
            secondary_waves=[
                {
                    'amplitude': (0.05 + np.random.random() * 0.25) * 60.0,  
                    'frequency': 0.1 + np.random.random() * 0.2,  # Lower frequencies
                    'phase': np.random.uniform(0, 2 * np.pi)
                }
                for _ in range(2)  # Reduced to 2 secondary waves
            ]
        )

    def _calculate_arrival_rate(self, position: float, cycle_params: CycleParameters) -> float:
        """Calculate arrival rate - simpler version"""
        # Main wave
        main_wave = math.sin(2 * math.pi * position + cycle_params.phase_shift)
        base_rate = self.base_rate + (main_wave * self.amplitude * cycle_params.amplitude_mod)
        
        # Add secondary waves
        for wave in cycle_params.secondary_waves:
            base_rate += (wave['amplitude'] * 
                        math.sin(wave['frequency'] * 2 * math.pi * position + wave['phase']))
        
        return base_rate
    

    def _calculate_instant_metrics(self):
        """Calculate metrics based on current cycle and position"""
        current_cycle = self.step_counter // self.trend_duration
        position = (self.step_counter % self.trend_duration) / self.trend_duration
        
        # Generate new cycle parameters if needed
        if current_cycle not in self.cycle_variations:
            self.cycle_variations[current_cycle] = self._generate_cycle_parameters()
            
            # Clean up old cycles to prevent memory growth
            old_cycles = [cycle for cycle in self.cycle_variations.keys() 
                        if cycle < current_cycle - 1]
            for cycle in old_cycles:
                del self.cycle_variations[cycle]
        
        base_rate = self._calculate_arrival_rate(position, self.cycle_variations[current_cycle])
        return base_rate

    def _simulate_interval(self, update_history=True) -> Dict[str, float]:
        """Simulate one 15-second interval"""
        # Simulate the passing of 15 seconds, but only use final values
        for i in range(self.collection_interval - 1):
            # Just advance time
            self.step_counter += 1
            
        # Get the final instant metrics
        base_rate = self._calculate_instant_metrics()
        
        # Apply random walk
        self.random_walk = np.clip(
            self.random_walk + np.random.normal(0, 0.5),
            -0.2, 0.2
        ) * 60.0

        instant_rate = base_rate + self.random_walk
        if self.previous_rate is not None:
            instant_rate = (self.smoothing_factor * self.previous_rate +
                        (1 - self.smoothing_factor) * instant_rate)

        instant_rate = np.clip(instant_rate, self.min_rate, self.max_rate)
        self.previous_rate = instant_rate
        self.arrival_rate = instant_rate

        # Calculate other metrics
        instant_cpu = self.get_cpu_utilization(instant_rate, self.current_replicas)
        instant_resp = self.get_response_time(instant_rate, self.current_replicas)

        self.step_counter += 1

        # Store values in history
        self.rate_history.append(instant_rate)
        self.response_time_history.append(instant_resp)
        self.cpu_util_history.append(instant_cpu)

        self.lstm_utils.update_lstm_predictor()

        return {
            'arrival_rate': instant_rate,
            'response_time': instant_resp,
            'cpu_utilization': instant_cpu
        }

    def reset(self, reset_step_counter=True) -> np.ndarray:
        """Reset environment state"""
        if reset_step_counter:
            self.step_counter = 0
        self.random_walk = 0.0
        self.current_replicas = self.min_replicas
        self.previous_rate = None
        
        # Clear histories
        self.rate_history.clear()
        self.response_time_history.clear()
        self.cpu_util_history.clear()
        
        return np.array([
                0.0,  # CPU utilization
                0.0,  # Request rate
                #self.Tmin / self.Tmax,  # Response time
                self.min_replicas / self.N_max,  # Replicas
                #0.0,  # Predicted rate
                0.0 # request rate trend
            ], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, float, float]:
        """Execute one step (15-second interval)"""

        # Simulate 15-second interval
        metrics = self._simulate_interval(update_history=False)

        # Log detailed step results
        self.logger.info(f"\nStep {self.step_counter} Results:")

        # Apply action to replicas
        if(self.is_hpa_simulation):
            desired_cpu = 60
            current_cpu = self.cpu_util_history[-1] if self.cpu_util_history else 0
            new_replicas = np.clip(
                math.ceil(self.current_replicas * (current_cpu / desired_cpu)),
                self.min_replicas,
                self.N_max
            )
        else:
            new_replicas = np.clip(
                self.current_replicas + action,
                self.min_replicas,
                self.N_max
            )

        current_time = self.step_counter * self.collection_interval
        if current_time - self.last_scale_time < self.scale_cooldown:
            action = 0  # Force no action during cooldown
            self.logger.info(f"Cooldown!")
        elif action != 0:
            self.last_scale_time = current_time
        
        self.logger.info(f"Action: {action} ({self.current_replicas} -> {new_replicas} replicas)")
        
        # Update replicas based on action
        self.current_replicas = new_replicas
        
        # Calculate reward based on averaged metrics
        reward = self._calculate_reward(
            metrics['response_time'],
            self.current_replicas,
            action
        )

        new_pred = None
        while self.prediction_buffer and self.prediction_buffer[0][1] <= self.step_counter:
            pred, time = self.prediction_buffer.popleft()
            new_pred = pred  # Keep track of most recent valid prediction

        if new_pred is not None:
            self.current_prediction = new_pred
        elif self.current_prediction is None:
            self.current_prediction = self.arrival_rate

        # Get prediction for future time
        new_prediction, pred_time = self.lstm_utils.predict_future_load()
        self.prediction_buffer.append((new_prediction, pred_time))
        self.logger.info(f"Made new prediction {new_prediction:.2f} for time {pred_time} with current time: {self.step_counter}")
        
        self.logger.info(f"Metrics:")
        self.logger.info(f"- Response Time: {metrics['response_time']:.3f}s (Tmax: {self.Tmax}s)")
        self.logger.info(f"- CPU Utilization: {metrics['cpu_utilization']:.2f}%")
        self.logger.info(f"- Arrival Rate: {metrics['arrival_rate']:.2f} req/min")
        self.logger.info(f"- Current prediction in use: {self.current_prediction:.2f} req/min")
        self.logger.info(f"- New prediction made: {new_prediction:.2f} req/min for time {pred_time}")

        state = self.get_state(new_prediction)
        self.logger.info(f"-state: {state}")
        
        return (
            state,
            reward,
            False,
            metrics['response_time'],
            metrics['arrival_rate']
        )


    def get_state(self, new_prediction = None) -> np.ndarray:
        """Get current state with normalized values"""

        # Get latest metrics from history
        cpu_util = self.cpu_util_history[-1]
        req_rate = self.rate_history[-1]

        resp_time = self.response_time_history[-1]
        p95_latency = np.percentile(list(self.response_time_history), 95) if self.response_time_history else self.Tmin

        self.logger.info(f"p95_latency: {p95_latency}")
        
         # Use the current prediction (which is for the current time)
        #if new_prediction is None:
          #  predicted_rate_normalized = 0.0
          #  rate_change = 0.0
       # else:
        #    rate_change = (new_prediction - req_rate) / self.max_rate
         #   if len(self.response_time_history) >= 3:
         #       rate_change = (new_prediction - self.rate_history[-2]) / self.max_rate
        #    predicted_rate_normalized = new_prediction / self.max_rate
        #    self.logger.info(f"Comparing prediction {self.current_prediction:.2f} req/min with actual request rate: {req_rate}")

        # Calculate rate of change for key metrics
        if len(self.response_time_history) >= 3:
            rate_change = (self.rate_history[-1] - self.rate_history[-3]) / self.max_rate
        else:
            rate_change = 0

        # Create state vector
        state = np.array([
            np.clip(cpu_util / 100.0, 0, 1),         # CPU utilization [0,1]
            np.clip(req_rate / self.max_rate, 0, 1), # Request rate [0,1]
            #np.clip(p95_latency / self.Tmax, 0, 1),    # Response time [0,1]
            np.clip(self.current_replicas / self.N_max, 0, 1), # Replicas [0,1]
            #np.clip(predicted_rate_normalized, 0, 1), # Predicted rate [0,1]
            rate_change                              # Rate change [-1,1]
        ], dtype=np.float32)

        return state

    def get_cpu_utilization(self, arrival_rate: float, replicas: int) -> float:
        """Calculate CPU utilization"""
        if replicas == 0:
            return 0
        arrival_rate_per_sec = arrival_rate / 60.0
        return min(100 * arrival_rate_per_sec / (replicas * self.mu), 150)

    def get_response_time(self, arrival_rate: float, replicas: int) -> float: 
        if replicas == 0:
            return self.Tmax
        
        # System capacity
        total_service_rate = replicas * self.mu

        arrival_rate_per_sec = arrival_rate / 60.0

        if arrival_rate_per_sec <= total_service_rate:
            # System is under or at capacity, so there is no queuing
            waiting_time = 0.0
        else:
            # Over capacity, approximate # of waiting stages
            backlog = arrival_rate_per_sec - total_service_rate

            # How many requests are in line and +1 ensures at least one stage
            stages = max(int(backlog / total_service_rate) + 1, 1)

            # Rate parameter for the Erlang
            waiting_time = np.random.gamma(shape=stages, scale= 1 / total_service_rate)
        
        self.logger.info(f"waiting_time: {waiting_time:.2f} seconds")

        load_factor = arrival_rate_per_sec / (replicas * self.mu)

        service_time = self.Tmin + load_factor * (1.0 - self.Tmin)
        
        self.logger.info(f"service_time: {service_time:.2f} seconds")

        return min(waiting_time + service_time, 1.0)

    def _calculate_reward(self, response_time: float, replicas: int, action: int) -> float:
        """Calculate reward based on performance and resource costs"""
        # Calculate load factor for additional logging
        arrival_rate_per_sec = self.arrival_rate / 60.0
        load_factor = arrival_rate_per_sec / (replicas * self.mu)

        c_perf = self._calculate_performance_cost(response_time, replicas)
        c_res = self._calculate_resource_cost(replicas)
        action_alignment = -0.4 if action != 0 else 0.2

        forecast_arr_rate = self.current_prediction  # next-step load
        required_replicas = math.ceil(forecast_arr_rate / 60.0 / self.mu)

        immediate_reward = 0.0
        # If the agent scaled to at least 'required_replicas', it's in alignment
        if self.current_replicas >= required_replicas:
            immediate_reward += 0.1  # small positive alignment bonus
        else:
            immediate_reward -= 0.1  # or a small penalty if it's ignoring forecast

        self.logger.info(f"Reward Calculation Details:")
        self.logger.info(f"- Performance Cost: {c_perf:.3f}")
        self.logger.info(f"- Resource Cost: {c_res:.3f}")
        self.logger.info(f"- action_alignment: {action_alignment:.3f}")
        self.logger.info(f"- Action: {action:.3f}")
        self.logger.info(f"- Total Reward: {-(c_perf + c_res ) }")

        return -(c_perf + c_res) + action_alignment #+ immediate_reward

    def _calculate_performance_cost(self, response_time: float, replicas: int) -> float:
        """Calculate performance cost based on response time"""
        if response_time <= self.Tmax:
            return ((response_time - self.Tmin) / (self.Tmax - self.Tmin))
                
        # For RT > Tmax, use a more reasonable scale
        base_penalty = 2.3 + ((response_time - self.Tmin) / (self.Tmax - self.Tmin))  # Start from 1.0 and add overage

        return base_penalty

    def _calculate_resource_cost(self, replicas: int) -> float:
        """Calculate resource cost based on number of replicas with stronger incentive for minimal replicas"""
        # Base cost increases quadratically with number of replicas
        base_cost = 2.5 * (replicas - self.min_replicas) / (self.N_max - self.min_replicas)
        
        # Add exponential penalty for higher replica counts
        exponential_factor = 1.3  # Adjust this to control the strength of the penalty
        additional_penalty = (exponential_factor ** (replicas - 1)) - 1
        
        # Give bonus for using minimum replicas when load is low
        if replicas == 1 and self.arrival_rate < self.base_rate * 0.2:  # 20% of base rate
            return -0.4  # Stronger reward for using minimum replicas under low load
            
        return base_cost + (additional_penalty * 0.1)