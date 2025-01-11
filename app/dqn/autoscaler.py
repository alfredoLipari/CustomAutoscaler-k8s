import time
import numpy as np
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import requests
import logging
from datetime import datetime
from typing import Optional, Tuple
import torch
import kubernetes.stream

from agent import DQNAgent
from environment import KubernetesEnvironment
from prom_query import MetricsCollector

class KubernetesAutoscaler:
    def __init__(
        self, 
        deployment_name: str, 
        namespace: str,
        service_url: str,
        dqn_model_path: Optional[str] = None, # Path to DQN model
        lstm_model_path: Optional[str] = None,  # Path to LSTM model
        scaling_cooldown: int = 45,  # Cooldown period between scaling actions
        metrics_interval: int = 15    # Time between metrics collection
    ):
        # Kubernetes configuration
        config.load_kube_config()
        self.apps_v1 = client.AppsV1Api()
        self.metrics_v1 = client.CustomObjectsApi()
        self.core_v1 = client.CoreV1Api()  # Add this line
        self.deployment_name = deployment_name
        self.namespace = namespace
        self.service_url = service_url


        # Setup logging
        self._setup_logging()
        
        # Initialize environment and metrics collector
        self.metrics_collector = MetricsCollector(deployment_name, namespace)
        self.env = KubernetesEnvironment(max_requests_per_minute=300.0, metrics_collector=self.metrics_collector, logger=self.logger)
        
        # Initialize agent
        self.state_size = 5  # [CPU, Req, Resp, Rep, Pred]

        # Load models
        self.agent = self._initialize_models(lstm_model_path, dqn_model_path)
        
        # Operational parameters
        self.metrics_interval = metrics_interval
        self.scaling_cooldown = scaling_cooldown
        self.last_scale_time = 0

        self.current_replicas = self.env.current_replicas
        self.desired_replicas = None

        self.predicted_rate = None
    
        
    def _setup_logging(self):
        """Configure logging with cross-platform compatible formatting"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Custom formatter with ASCII characters instead of Unicode
        class CustomFormatter(logging.Formatter):
            grey = "\x1b[38;21m"
            blue = "\x1b[34;21m"
            yellow = "\x1b[33;21m"
            red = "\x1b[31;21m"
            bold_red = "\x1b[31;1m"
            reset = "\x1b[0m"

            format_str = "%(asctime)s - %(levelname)s - %(message)s"
            
            FORMATS = {
                logging.DEBUG: grey + format_str + reset,
                logging.INFO: blue + format_str + reset,
                logging.WARNING: yellow + format_str + reset,
                logging.ERROR: red + format_str + reset,
                logging.CRITICAL: bold_red + format_str + reset
            }

            def format(self, record):
                log_fmt = self.FORMATS.get(record.levelno)
                formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
                return formatter.format(record)

        handler = logging.StreamHandler()
        handler.setFormatter(CustomFormatter())
        
        file_handler = logging.FileHandler(f'autoscaler_{timestamp}.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        logging.basicConfig(level=logging.INFO, handlers=[handler, file_handler])
        self.logger = logging.getLogger('KubernetesAutoscaler')

    def _initialize_models(self, lstm_model_path: Optional[str], dqn_model_path: Optional[str]):
        """Initialize and load both LSTM and DQN models"""
        # Load LSTM model
        if lstm_model_path:
            try:
                # Add required numpy types to safe globals
                import numpy as np
                from torch.serialization import add_safe_globals
                add_safe_globals([
                    np.ndarray,
                    np._core.multiarray.scalar,
                    np.dtype,  # Added numpy.dtype to safe globals
                    np.bool_,  # Common numpy types that might be in the checkpoint
                    np.float32,
                    np.float64,
                    np.int64
                ])
                
                try:
                    # Try safe loading first
                    checkpoint = torch.load(lstm_model_path, weights_only=True)
                except Exception as e:
                    # Fallback to unsafe loading for your own models
                    self.logger.warning(f"Safe loading failed, falling back to unsafe loading: {e}")
                    checkpoint = torch.load(lstm_model_path, weights_only=False)
                    
                if 'lstm_state_dict' in checkpoint:
                    self.env.lstm_predictor.load_state_dict(checkpoint['lstm_state_dict'])
                elif 'model_state_dict' in checkpoint:
                    self.env.lstm_predictor.load_state_dict(checkpoint['model_state_dict'])
                else:
                    raise KeyError("No state dict found in LSTM checkpoint")
                        
                self.env.lstm_predictor.eval()
                self.logger.info(f"Loaded LSTM model from {lstm_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load LSTM model: {e}")
                raise

        # Initialize agent parameters
        self.action_space = [-1, 0, 1]
        self.action_size = len(self.action_space)
        
        # Initialize agent and load DQN model
        agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            epsilon_start=0.0,
            logger=self.logger
        )
        
        if dqn_model_path:
            try:
                try:
                    # Try safe loading first
                    checkpoint = torch.load(dqn_model_path, weights_only=True)
                except Exception as e:
                    # Fallback to unsafe loading for your own models
                    self.logger.warning(f"Safe loading failed, falling back to unsafe loading: {e}")
                    checkpoint = torch.load(dqn_model_path, weights_only=False)
                    
                if 'dqn_state_dict' not in checkpoint:
                    raise KeyError("No DQN state dict found in checkpoint")
                    
                agent.q_network.load_state_dict(checkpoint['dqn_state_dict'])
                agent.target_network.load_state_dict(checkpoint['dqn_target_state_dict'])
                agent.q_network.eval()
                agent.target_network.eval()
                self.logger.info(f"Loaded DQN model from {dqn_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load DQN model: {e}")
                raise
                    
        return agent

    def get_cpu_utilization(self) -> float:
        """Get CPU utilization and pod count"""
        try:
            cpu_usage_total = 0
            cpu_requests_total = 0
            pod_count = 0

            pod_metrics = self.metrics_v1.list_namespaced_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="pods"
            )
            
            deployment = self.apps_v1.read_namespaced_deployment(
                self.deployment_name, 
                self.namespace
            )

            for pod in pod_metrics['items']:
                if pod['metadata']['labels'].get("app") == self.deployment_name:
                    pod_count += 1
                    for container in pod['containers']:
                        cpu_usage = container['usage']['cpu']
                        cpu_usage_total += self._convert_cpu_value(cpu_usage)

                    for container_spec in deployment.spec.template.spec.containers:
                        cpu_requests = container_spec.resources.requests['cpu']
                        cpu_requests_total += self._convert_cpu_value(cpu_requests)

            if cpu_requests_total > 0 and pod_count > 0:
                cpu_utilization = (cpu_usage_total / cpu_requests_total) * 100
            else:
                cpu_utilization = 0

            self.logger.info(f'CPU Utilization: {cpu_utilization:.2f}% across {pod_count} pods')
            return cpu_utilization

        except ApiException as e:
            self.logger.error(f"Failed to fetch CPU metrics: {e}")
            return 0.0

    def _convert_cpu_value(self, cpu_value: str) -> float:
        """Convert CPU value to millicores"""
        if cpu_value.endswith('n'):
            return float(cpu_value[:-1]) / 1e6
        elif cpu_value.endswith('u'):
            return float(cpu_value[:-1]) / 1e3
        elif cpu_value.endswith('m'):
            return float(cpu_value[:-1])
        else:
            return float(cpu_value) * 1000

    def get_response_time(self) -> float:
        """Get response time from application endpoint"""
        try:
            if self.predicted_rate is not None:
                response = requests.get(f"{self.service_url}", timeout=5)
            else:
                response = requests.get(f"{self.service_url}/", timeout=5)
            response.raise_for_status()
            response_json = response.json()
            response_time = float(response_json.get("time_taken_seconds", 0))
            #self.update_metric_all_pods(self.predicted_rate)
            self.logger.debug(f"Response time: {response_time:.3f}s")
            return response_time
        except Exception as e:
            self.logger.error(f"Failed to fetch response time: {e}")
            return 0
        
    def get_deployment_state(self) -> Tuple[int, int]:
        """Get current and available replicas from deployment"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                self.deployment_name,
                self.namespace
            )
            return deployment.spec.replicas, (deployment.status.available_replicas or 0)
        except ApiException as e:
            self.logger.error(f"Failed to get deployment state: {e}")
            return self.current_replicas, 0

    def scale_deployment(self, desired_replicas: int) -> bool:
        """Scale the deployment to the specified number of replicas"""
        current_time = time.time()

        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                self.deployment_name,
                self.namespace
            )

            current_replicas = deployment.spec.replicas
            available_replicas = deployment.status.available_replicas or 0

            if desired_replicas == current_replicas:
                self.logger.info("Scaling skipped: no change in replicas")
                return False


            self.current_replicas = current_replicas

            # If this is our first scaling action or we've reached our previous desired state
            if (self.desired_replicas is None or 
                (self.desired_replicas == available_replicas and 
                 current_replicas == available_replicas)):
                
                # Set new desired state and perform scaling
                self.desired_replicas = desired_replicas
                deployment.spec.replicas = int(desired_replicas)
                
                self.apps_v1.patch_namespaced_deployment(
                    self.deployment_name,
                    self.namespace,
                    deployment
                )
                
                self.last_scale_time = current_time
                self.logger.info(f"Scaling deployment from {current_replicas} to {desired_replicas} replicas")
                return True
            
            else:
                # We're still waiting for previous scaling action to complete
                self.logger.info(f"Scaling skipped: waiting for previous scaling to complete")
                self.logger.info(f"Current replicas: {current_replicas}")
                self.logger.info(f"Available replicas: {available_replicas}")
                self.logger.info(f"Target replicas: {self.desired_replicas}")
                return False
            
        except ApiException as e:
            self.logger.error(f"Failed to scale deployment: {e}")
            return False
        

    def update_metric_all_pods(self, predicted_rate: float) -> None:
        if predicted_rate is None:
            self.logger.warning("Skipping metric update: predicted_rate is None")
            return
            
        try:
            # Use the service URL which is already accessible
            url = f"{self.service_url.rstrip('/')}/update_metric"
            response = requests.post(
                url, 
                json={"value": float(predicted_rate)}, 
                timeout=5
            )
            response.raise_for_status()
            self.logger.info(f"Successfully updated metric to {predicted_rate}")
                
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")

    def run(self):
        """Main autoscaling loop"""
        self.logger.info("╔════ Autoscaler Started ════╗")
        self.logger.info(f"║ Deployment: {self.deployment_name}")
        self.logger.info(f"║ Namespace: {self.namespace}")
        self.logger.info("╚══════════════════════════╝")

        # Initialize deployment state
        current_replicas, available_replicas = self.get_deployment_state()
        self.current_replicas = current_replicas
        self.desired_replicas = current_replicas  # Set initial desired replicas only once
        
        while True:
            try:
                # 1. Collect all metrics
                self.logger.info("Step 1: Collecting Metrics...")
                cpu_util = self.get_cpu_utilization()
                response_time = self.get_response_time()
                
                # Get current deployment state
                current_replicas, available_replicas = self.get_deployment_state()
                self.current_replicas = current_replicas  # Update current replicas
                
                metrics = self.metrics_collector.collect_metrics()
                if not metrics:
                    self.logger.warning("⚠ Failed to collect metrics, skipping iteration")
                    time.sleep(self.metrics_interval)
                    continue

                self.logger.info("\nStep 2: Build State")
                self.logger.info(f"metrics: {metrics}")
                # 2. Build current state
                state = self.env.get_state(
                    cpu_util=cpu_util,
                    replicas=current_replicas,
                    req_rate=metrics['request_rate'],
                    percentile_95=metrics['95th_percentile_latency'],
                    resp_time=response_time
                )

                self.predicted_rate = state[-2]

                # 3. Get action from agent
                self.logger.info("\nStep 3: Selecting Action")
                action_index = self.agent.act(state)
                action = self.action_space[action_index]
                
                # Log Q-values for debugging
                with torch.no_grad():
                    q_values = self.agent.q_network(torch.FloatTensor(state).to(self.agent.device))
                    self.logger.info(f"Q-values for actions (-1,0,1): {q_values.cpu().numpy()}")
                    
                # 4. Apply scaling action
                desired_replicas = int(np.clip(  # Convert to int to fix the openapi_types error
                    current_replicas + action,
                    self.env.min_replicas,
                    self.env.N_max
                ))
                
                self.logger.info("\nStep 4: Scaling Decision")
                self.logger.info(f"▶ Action: {action:+d}")
                self.logger.info(f"▶ Current Replicas: {current_replicas}")
                self.logger.info(f"▶ Desired Replicas: {desired_replicas}")
                scaling_applied = self.scale_deployment(desired_replicas)
                
                if not scaling_applied:
                    self.logger.info("⚠ Scaling action skipped (cooldown or no change needed)")

                self.logger.info("\n" + "─" * 50)
                # 6. Wait before next iteration
                time.sleep(self.metrics_interval)

            except Exception as e:
                self.logger.error("╳╳╳ Error in autoscaling loop ╳╳╳")
                self.logger.error(f"Error in autoscaling loop: {e}", exc_info=True)  # Added exc_infoù
                time.sleep(self.metrics_interval)
                

if __name__ == "__main__":
    autoscaler = KubernetesAutoscaler(
        deployment_name="cpu-intensive-app",
        namespace="fastapi-app",
        service_url="http://34.154.36.173/",
        dqn_model_path="dqn_model_20250105_103937.pth",
        lstm_model_path="lstm_model_20250105_103937.pth"
    )
    autoscaler.run()