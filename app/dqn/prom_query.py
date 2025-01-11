import re
from datetime import datetime
from kubernetes import client, config
from collections import deque
from typing import Optional, Dict, Deque
import torch
import numpy as np

class MetricsCollector:
    def __init__(self, deployment_name: str, namespace: str, window_size: int = 60, max_requests_per_minute: int = 300):
        self.deployment_name = deployment_name
        self.namespace = namespace
        self.window_size = window_size  # Number of intervals to keep per pod
        
        # For rate calculations
        self.sequence_length = 16  # Should match training window
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.request_rates = deque(maxlen=window_size)
        
        self.max_requests_per_minute = max_requests_per_minute
        self.min_rate = 3.0  # Match training environment
        
        # Initialize per-pod data storage
        self.pod_metrics = {}  # Stores per-pod counts and timestamps
        
        config.load_kube_config()
        self.core_v1 = client.CoreV1Api()
        
        # Smoothing factor for EMA
        self.alpha = 0.5

    def query_metrics(self) -> Optional[Dict]:
        """Query metrics from pods and compute per-pod rates using linear regression."""
        try:
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"app={self.deployment_name}"
            )
            
            if not pods.items:
                raise Exception("No pods found for the deployment")
            
            # Sort pods by creation timestamp and get the oldest one
            oldest_pod = sorted(
                pods.items, 
                key=lambda x: x.metadata.creation_timestamp
            )[0]

            oldest_pod_name = oldest_pod.metadata.name
            current_time = datetime.now()
            total_rate = 0.0
            oldest_pod_percentile_95 = 0.0  # Initialize the 95th percentile value
            
            # Get current pod names and remove data for pods that no longer exist
            current_pod_names = set([pod.metadata.name for pod in pods.items])
            previous_pod_names = set(self.pod_metrics.keys())
            removed_pods = previous_pod_names - current_pod_names
            
            for pod_name in removed_pods:
                print(f"Pod {pod_name} has been removed, deleting stored data.")
                del self.pod_metrics[pod_name]
            
            for pod in pods.items:
                pod_name = pod.metadata.name
                try:
                    metrics_text = self.core_v1.connect_get_namespaced_pod_proxy_with_path(
                        name=pod_name,
                        namespace=self.namespace,
                        path="metrics"
                    )
                    
                    pod_requests = self.extract_metric(metrics_text, 'example_requests_total') or 0.0
                    timestamp = current_time.timestamp()
                    
                    # Initialize metrics storage for new pods
                    if pod_name not in self.pod_metrics:
                        self.pod_metrics[pod_name] = deque(maxlen=self.window_size)
                    
                    # Append current count and timestamp
                    self.pod_metrics[pod_name].append((timestamp, pod_requests))
                    
                    # Compute rate using linear regression if we have enough data points
                    pod_data = self.pod_metrics[pod_name]
                    if len(pod_data) >= 2:
                        rate = self.compute_rate(pod_data)
                        total_rate += rate
                        print(f"Pod {pod_name}: Computed rate: {rate:.2f} req/min")
                    else:
                        print(f"Pod {pod_name}: Not enough data points to compute rate.")

                    # Extract 95th percentile latency only from the oldest pod
                    if pod_name == oldest_pod_name:
                        oldest_pod_percentile_95 = self.extract_metric(metrics_text, 'percentile_95_latency_seconds') or 0.0
                        print(f"Oldest Pod {pod_name}: 95th percentile latency: {oldest_pod_percentile_95:.2f} seconds")
                    
                except Exception as e:
                    print(f"Error querying metrics from pod {pod_name}: {e}")
                    continue

            print(f"Total request rate from pods: {total_rate:.2f} req/min")
            print(f"95th percentile latency from oldest pod: {oldest_pod_percentile_95:.2f} seconds")
            
            return {
                'request_rate': total_rate,
                '95th_percentile_latency': oldest_pod_percentile_95
            }
        
        except Exception as e:
            print(f"Error querying metrics: {e}")
            return None

    def extract_metric(self, metrics_text: str, metric_name: str) -> Optional[float]:
        """Extract metric value from text."""
        match = re.search(rf'^{metric_name}\s+([\d.eE+-]+)', metrics_text, re.MULTILINE)
        return float(match.group(1)) if match else None

    def compute_rate(self, pod_data: Deque) -> float:
        """Compute rate using shorter time windows for better accuracy."""
        try:
            if len(pod_data) < 2:
                return 0.0

            # Use last 60 seconds for rate calculation
            current_time = pod_data[-1][0]
            window_start_time = current_time - 45
            
            # Filter data points within the last 60 seconds
            recent_data = [(t, c) for t, c in pod_data if t >= window_start_time]
            
            if len(recent_data) < 2:
                return 0.0
                
            time_start, count_start = recent_data[0]
            time_end, count_end = recent_data[-1]
            
            if count_end < count_start:  # Handle counter reset
                count_start = 0
                
            time_diff = time_end - time_start
            count_diff = count_end - count_start
            
            if time_diff <= 0:
                return 0.0
                
            rate = (count_diff / time_diff) * 60
            return max(3.0, min(rate, self.max_requests_per_minute))
                    
        except Exception as e:
            print(f"Error computing rate: {e}")
            return 0.0

    def collect_metrics(self) -> Optional[Dict]:
        """Collect request rate metrics with exponential moving average smoothing."""
        metrics = self.query_metrics()
        if not metrics:
            return {
                'request_rate': self.request_rates[-1] if self.request_rates else self.min_rate,
                '95th_percentile_latency': 0.0
            }
        
        current_rate = metrics['request_rate']
        
        # Apply exponential moving average smoothing
        if self.request_rates:
            previous_smoothed_rate = self.request_rates[-1]
            smoothed_rate = self.alpha * current_rate + (1 - self.alpha) * previous_smoothed_rate
        else:
            smoothed_rate = current_rate
        
        # Ensure rate is within bounds
        smoothed_rate = max(self.min_rate, min(smoothed_rate, self.max_requests_per_minute))
        
        self.request_rates.append(smoothed_rate)
        
        print(f"Calculated Rate: {current_rate:.2f} req/min")
        print(f"Smoothed Rate: {smoothed_rate:.2f} req/min")
        if len(self.request_rates) > 1:
            print(f"Previous Smoothed Rate: {self.request_rates[-2]:.2f} req/min")
        
        return {'request_rate': smoothed_rate, '95th_percentile_latency': metrics['95th_percentile_latency']}

    def predict_next_rate(self, lstm_model: torch.nn.Module) -> float:
        """Predict next request rate using pretrained LSTM model."""
        try:
            if len(self.request_rates) < self.sequence_length:
                return self.request_rates[-1] if self.request_rates else self.min_rate
            
            # Get latest sequence
            sequence = list(self.request_rates)[-self.sequence_length:]
            sequence = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1).to(self.device)
            
            # Calculate trend
            last_points = sequence[0, -4:, 0].cpu()
            trend = (last_points[-1] - last_points[0]) / 3
            
            lstm_model.eval()
            with torch.no_grad():
                # Get model prediction
                model_prediction = float(lstm_model(sequence).cpu().item())
                last_value = float(sequence[0, -1, 0].cpu().item())
                
                # Blend predictions
                predicted_change = model_prediction - last_value
                blended_change = 0.40 * predicted_change + 0.60 * trend
                prediction = last_value + blended_change
                print(f"model_prediction: {model_prediction}")
                print(f"last_value: {last_value}")
                
                # Apply trend-based bounds
                max_change = abs(trend) * 1.5
                if prediction > last_value:
                    prediction = min(prediction, last_value + max_change)
                else:
                    prediction = max(prediction, last_value - max_change)
                
                # Final bounds check
                prediction = min(max(prediction, self.min_rate), self.max_requests_per_minute)
                
                return prediction
                
        except Exception as e:
            print(f"Error in LSTM prediction: {e}")
            return self.request_rates[-1] if self.request_rates else self.min_rate
