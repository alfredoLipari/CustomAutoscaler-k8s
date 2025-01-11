# predictor_utils.py
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import logging
from datetime import datetime
from collections import deque
import traceback
from typing import List, Dict, Tuple, Optional

class LSTMUtils:
    def __init__(self, env):
        self.env = env
        self.rate_history = deque(maxlen=1000)
        self.sequence_length = 16
        self.prediction_horizon = 1
        self.training_buffer = deque(maxlen=32)  # Small buffer for recent samples
        self.batch_size = 8  # Small batch size
        self.loss_values = []
        self.logger = self.setup_logging()

        # Add timestamps to rate history
        self.timestamped_history = deque(maxlen=1000)  # Store (timestamp, rate) pairs
    
    def predict_trend(self, sequence: torch.Tensor) -> float:
        """Calculate the trend direction and magnitude"""
        # Get the last few points to determine trend
        last_points = sequence[0, -4:, 0]  # Last 4 points
        slope = (last_points[-1] - last_points[0]) / 3
        return slope

    def update_lstm_predictor(self) -> Optional[float]:
        try:
            current_time = self.env.step_counter
            self.timestamped_history.append((current_time, self.env.arrival_rate))
            self.rate_history.append(self.env.arrival_rate)
            
            if len(self.rate_history) < self.sequence_length + self.prediction_horizon:
                return None
            
            self.logger.info(f"update called")
            self.logger.info(f"current_time: {current_time}")

            current_sequence = list(self.rate_history)[-self.sequence_length-1:-1]
            future_target = self.rate_history[-1]

            self.training_buffer.append((current_sequence, future_target))

            if len(self.training_buffer) >= self.batch_size:
                sequences, targets = zip(*self.training_buffer)
                
                sequences = torch.FloatTensor(sequences).to(self.env.device)
                sequences = sequences.unsqueeze(-1)
                targets = torch.FloatTensor(targets).to(self.env.device)
                targets = targets.unsqueeze(-1)

                self.env.lstm_predictor.train()
                self.env.optimizer.zero_grad()
                
                predictions = self.env.lstm_predictor(sequences)
                loss = nn.MSELoss()(predictions, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.env.lstm_predictor.parameters(), 1.0)
                self.env.optimizer.step()

                self.loss_values.append(loss.item())
                self.training_buffer.clear()
                
                return loss.item()

            return None

        except Exception as e:
            self.logger.error(f"Update error: {str(e)}\n{traceback.format_exc()}")
            return None

    def predict_future_load(self) -> Tuple[float, int]:
        try:
            if len(self.rate_history) < self.sequence_length:
                return self.env.arrival_rate, self.env.step_counter + self.prediction_horizon
            
            # Get latest sequence
            sequence = list(self.rate_history)[-self.sequence_length:]
            sequence = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1).to(self.env.device)
            self.logger.info(f"predict - sequence before normalize: {sequence}")

            # Calculate trend (still useful for bounds)
            last_points = sequence[0, -4:, 0].cpu()
            trend = (last_points[-1] - last_points[0]) / 3
            self.logger.info(f"predict - trend: {trend}")
            
            # Don't need normalization anymore since we're predicting differences
            self.env.lstm_predictor.eval()
            with torch.no_grad():
                # Model now directly predicts next value
                model_prediction = float(self.env.lstm_predictor(sequence).cpu().item())

                last_value = float(sequence[0, -1, 0].cpu().item())
                trend_prediction = last_value + trend
                
                # Get the predicted change from model
                predicted_change = model_prediction - last_value
                
                # Blend changes instead of absolute values
                blended_change = 0.70 * predicted_change + 0.30 * trend
                prediction = last_value + blended_change
                
                # Apply bounds based on trend
                max_change = abs(trend) * 1.5
                if prediction > last_value:
                    prediction = min(prediction, last_value + max_change)
                else:
                    prediction = max(prediction, last_value - max_change)
                
                # Final bounds check
                prediction = min(max(prediction, float(self.env.min_rate)), float(self.env.max_rate))
                
                self.logger.info(f"prediction - predicted_change: {predicted_change}")
                self.logger.info(f"prediction - trend_prediction: {trend_prediction}")
                self.logger.info(f"prediction - model_prediction: {model_prediction}")
                self.logger.info(f"prediction - blended prediction: {0.70 * model_prediction + 0.30 * trend_prediction}")
                self.logger.info(f"prediction - final prediction: {prediction}")


            return prediction, self.env.step_counter + self.prediction_horizon

        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
            return self.env.arrival_rate, self.env.step_counter + self.prediction_horizon
        
    def setup_logging(self):
        logger = logging.getLogger('LSTM_Predictor')
        if not logger.handlers:
            fh = logging.FileHandler(f'lstm_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            fh.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(fh)
            logger.setLevel(logging.INFO)
        return logger