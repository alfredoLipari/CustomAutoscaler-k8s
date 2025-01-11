import numpy as np
import torch
from predictor_network import LSTMPredictor
from predictor_utils import LSTMUtils
from collections import deque


class KubernetesEnvironment:
    def __init__(
        self,
        Tmax: float = 0.8,
        N_max: int = 13,
        min_replicas: int = 1,
        history_length: int = 100,
        max_requests_per_minute: float = 260.0,  # 10 req/sec * 60
        metrics_collector = None,  # Add metrics_collector parameter
        logger = None  # Add logger parameter
    ):
        
        self.logger = logger

        # Service Level Objectives
        self.Tmax = Tmax
        self.Tmin = 0.4
        
        # Scaling parameters
        self.N_max = N_max
        self.min_replicas = min_replicas
        self.action_space = [-1, 0, 1]
        self.max_rate = max_requests_per_minute  # Store the parameter
        
        # State tracking
        self.current_replicas = self.min_replicas
        self.request_rate_history = deque(maxlen=history_length)
        
        # Prediction components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm_predictor = LSTMPredictor().to(self.device)
        self.optimizer = torch.optim.Adam(self.lstm_predictor.parameters(), lr=0.001)

        # Store metrics collector reference
        self.metrics_collector = metrics_collector
        
        self.current_prediction = None  # Store the prediction being currently used
    
    def update_request_rate(self, req_rate: float):
        """Update request rate history"""
        self.request_rate_history.append(req_rate)

    def get_state(self, cpu_util: float, replicas: int, req_rate: float, 
                  resp_time: float, percentile_95: float) -> np.ndarray:
        """
        Get current state. All rates are in req/min to match test environment.
        
        Args:
            cpu_util: CPU utilization (0-100)
            replicas: Current number of replicas
            req_rate: Request rate in req/min
            resp_time: Response time in seconds
        """

        # Update current replicas
        self.current_replicas = replicas
        self.update_request_rate(req_rate)

        # Compare current request rate with previous prediction if available
        if self.current_prediction is not None:
            prediction_diff = self.current_prediction - req_rate
            if self.logger:
                self.logger.info("\n┌─── Prediction vs Reality ───┐")
                self.logger.info(f"│ Previous Prediction: {self.current_prediction:.1f} req/min")
                self.logger.info(f"│ Actual Rate: {req_rate:.1f} req/min")
                self.logger.info(f"│ Difference: {prediction_diff:.1f} req/min")
                self.logger.info("└────────────────────────────────┘")

        # Get prediction (already in req/min)
        predicted_rate = self.metrics_collector.predict_next_rate(self.lstm_predictor)
        self.current_prediction = predicted_rate

        if resp_time == 0.0: # Timeout occured
            resp_time = 1.0

        if len(self.request_rate_history) >= 3:
            #rate_change = (self.request_rate_history[-1] - self.request_rate_history[-3]) / self.max_rate
            rate_change = (predicted_rate - self.request_rate_history[-2]) / self.max_rate
        else:
            rate_change = 0

        self.logger.info("\n┌─── Raw Metrics ───┐")
        self.logger.info(f"│ CPU: {cpu_util:.1f}%")
        self.logger.info(f"│ Replicas: {replicas}")
        self.logger.info(f"│ Request Rate: {req_rate:.1f} req/min")
        self.logger.info(f"│ Response Time: {resp_time:.3f}s")
        self.logger.info(f"| Predicted Rate at next step: {predicted_rate} req/min")
        self.logger.info(f"| Change Rate: {(predicted_rate - req_rate)} req/min")
        self.logger.info("└──────────────────────────┘")

        # Create state vector
        state = np.array([
            np.clip(cpu_util / 100.0, 0, 1),         # CPU utilization [0,1]
            np.clip(req_rate / self.max_rate, 0, 1), # Request rate [0,1]
            #np.clip(percentile_95 / self.Tmax, 0, 1),    # Response time [0,1]
            #np.clip(0.9507416, 0, 1),    # Response time [0,1]
            np.clip(replicas / self.N_max, 0, 1), # Replicas [0,1]
            np.clip(predicted_rate / self.max_rate, 0, 1), # Predicted rate [0,1]
            rate_change                              # Rate change [-1,1]
        ], dtype=np.float32)


        print(f"State")
        print(state)
            

        return state