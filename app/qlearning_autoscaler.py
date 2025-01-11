import time
import numpy as np
import math
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import requests
from prom_query import MetricsCollector

class MicroserviceEnvironment:
    def __init__(self, Tmax=0.7, theta_min=0.5, theta_max=0.9, delta=0.05):
        self.response_time_history = []  # Keep a history of response times
        self.Tmax = Tmax
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.delta = delta
        self.theta_u = (theta_min + theta_max) / 2
        self.action_space = [-delta, 0, delta]
        # Scaling factors for costs
        self.alpha = 10   # Reward scaling for performance
        self.beta = 100   # Penalty scaling for performance
        self.gamma = 2   # Scaling for resource cost
        self.k_max = 10   # Maximum number of replicas
        self.w_perf = 0.5
        self.w_res = 0.5

    def get_state(self, u, k, request_rate, latency_95th):
        u_disc = min(int(u / 10), 10)  # CPU utilization binned every 10%
        k_disc = min(k, 10)  # Number of replicas, capped at 10

        # Adjust binning for request rate
        # Assuming request_rate is in requests per minute
        request_rate_disc = min(int(request_rate / 10), 10)  # Bins of 10 requests per minute

        latency_95th_rounded = round(latency_95th, 1)

        # Print the discretization for debugging
        print(f"Request Rate: {request_rate}, Discretized Request Rate Bin: {request_rate_disc}")

        return (u_disc, k_disc, request_rate_disc, latency_95th_rounded)

    def calculate_moving_average_response_time(self, t, window_size=5):
        # Keep the history within the window size
        self.response_time_history.append(t)
        if len(self.response_time_history) > window_size:
            self.response_time_history.pop(0)
        # Calculate the moving average
        return sum(self.response_time_history) / len(self.response_time_history)

    def step(self, action, u, t, k, request_rate, latency_95th):
        # Update the threshold based on the action
        print(f"Action taken: {action}")
        self.theta_u += action
        self.theta_u = np.clip(self.theta_u, self.theta_min, self.theta_max)

        # Get the moving average response time
        avg_response_time = self.calculate_moving_average_response_time(t)

        # Calculate costs
        c_perf = self.performance_cost(avg_response_time)
        c_res = self.resource_cost(k)
        cost = self.w_perf * c_perf + self.w_res * c_res
        print(f"Performance Cost: {c_perf}")
        print(f"Resource Cost: {c_res}")
        print(f"Total Cost: {cost}")
        reward = -cost

        next_state = self.get_state(u, k, request_rate, latency_95th)
        return next_state, reward, self.theta_u

    def performance_cost(self, t):
        if t <= self.Tmax:
            c_perf = -self.alpha * ((self.Tmax - t) / self.Tmax)
        else:
            c_perf = self.beta * ((t - self.Tmax) / self.Tmax)
        return c_perf

    def resource_cost(self, k):
        c_res = -self.gamma * (1 - (k / self.k_max))
        return c_res


class QLearningAgent:
    def __init__(self, alpha=0.2, gamma=0.99, epsilon=1.0):
        self.Q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_Q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def update_Q(self, state, action, reward, next_state, possible_actions):
        max_Q_next = max([self.get_Q(next_state, a) for a in possible_actions])
        current_Q = self.get_Q(state, action)
        new_Q = (1 - self.alpha) * current_Q + self.alpha * (reward + self.gamma * max_Q_next)
        print(f"Updated Q-value: {new_Q}")
        self.Q[(state, action)] = new_Q

    def select_action(self, state, possible_actions):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(possible_actions)
        else:
            Q_values = [self.get_Q(state, a) for a in possible_actions]
            max_Q = max(Q_values)
            max_actions = [a for a, q in zip(possible_actions, Q_values) if q == max_Q]
            return np.random.choice(max_actions)

    def print_q_table(self):
        print("### Current Q-Table ###")
        sorted_q_table = sorted(self.Q.items(), key=lambda x: x[0])
        for (state, action), q_value in sorted_q_table:
            print(f"State: {state}, Action: {action}, Q-Value: {q_value:.4f}")
        print("######################")


class KubernetesAutoscaler:
    def __init__(self, deployment_name, namespace):
        config.load_kube_config()  # Use kubeconfig locally
        self.apps_v1 = client.AppsV1Api()
        self.metrics_v1 = client.CustomObjectsApi()
        self.deployment_name = deployment_name
        self.namespace = namespace
        self.env = MicroserviceEnvironment()
        self.agent = QLearningAgent()
        self.metrics_collector = MetricsCollector(deployment_name, namespace)

        # Tracking variables
        self.total_replicas = 0
        self.total_cpu_utilization = 0
        self.iteration_count = 0
        self.response_time_exceed_count = 0

        self.time_sleep = 20

    def get_metrics(self):
        cpu_usage_total = 0
        cpu_requests_total = 0
        pod_count = 0

        try:
            pod_metrics = self.metrics_v1.list_namespaced_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="pods"
            )
            deployment = self.apps_v1.read_namespaced_deployment(self.deployment_name, self.namespace)

            for pod in pod_metrics['items']:
                if pod['metadata']['labels'].get("app") == self.deployment_name:
                    pod_count += 1
                    for container in pod['containers']:
                        cpu_usage = container['usage']['cpu']
                        cpu_usage_total += self.convert_cpu_usage(cpu_usage)

                    # Get resource requests from the deployment spec
                    for container_spec in deployment.spec.template.spec.containers:
                        cpu_requests = container_spec.resources.requests['cpu']
                        cpu_requests_total += self.convert_cpu_request(cpu_requests)

        except ApiException as e:
            print(f"Exception while fetching metrics: {e}")

        if cpu_requests_total > 0 and pod_count > 0:
            cpu_utilization_percentage = (cpu_usage_total / cpu_requests_total) * 100
        else:
            cpu_utilization_percentage = 0

        print(f'Average CPU Utilization: {cpu_utilization_percentage:.2f}% across {pod_count} pods')
        return cpu_utilization_percentage, pod_count

    def convert_cpu_usage(self, cpu_usage):
        if cpu_usage.endswith('n'):
            return int(cpu_usage[:-1]) / 1e6
        elif cpu_usage.endswith('u'):
            return int(cpu_usage[:-1]) / 1e3
        elif cpu_usage.endswith('m'):
            return int(cpu_usage[:-1])
        else:
            return float(cpu_usage) * 1000

    def convert_cpu_request(self, cpu_request):
        if cpu_request.endswith('m'):
            return int(cpu_request[:-1])
        else:
            return float(cpu_request) * 1000

    def get_response_time(self):
        """Fetches real-time response time from the application"""
        try:
            response = requests.get("http://34.17.13.81/")
            response_json = response.json()
            print(f"Response JSON: {response_json}")
            response_time = float(response_json.get("time_taken_seconds", 0))
        except Exception as e:
            print(f"Failed to fetch response time: {e}")
            response_time = 0
        return response_time

    def get_current_replicas(self):
        deployment = self.apps_v1.read_namespaced_deployment(self.deployment_name, self.namespace)
        print(f'Current replicas: {deployment.spec.replicas}')
        return deployment.spec.replicas

    def scale_deployment(self, replicas):
        deployment = self.apps_v1.read_namespaced_deployment(self.deployment_name, self.namespace)
        deployment.spec.replicas = replicas
        print(f'Scaling replicas to: {replicas}')
        self.apps_v1.patch_namespaced_deployment(self.deployment_name, self.namespace, deployment)

    def run(self):
        epsilon_decay = 0.995  # Decay epsilon by 0.5% each iteration
        min_epsilon = 0.05  # Minimum exploration rate
        q_table_print_interval = 180  # Print the Q-table every 3 minutes
        last_print_time = time.time()

        while True:
            cpu_utilization_percentage, pod_count = self.get_metrics()
            current_replicas = self.get_current_replicas()

            if current_replicas != pod_count:
                print(f"Skipping iteration. Current Replicas: {current_replicas}, Running Pods: {pod_count}")
                time.sleep(15)
                continue

            # Collect new metrics from the MetricsCollector
            collected_metrics = self.metrics_collector.collect_metrics()
            if collected_metrics:
                request_rate = collected_metrics['request_rate']
                latency_95th = collected_metrics['latency_95th']
            else:
                print("Waiting for metrics to be collected...")
                time.sleep(15)
                continue

            # Fetch real-time response time from the application
            response_time = self.get_response_time()

            current_state = self.env.get_state(cpu_utilization_percentage, current_replicas, request_rate, latency_95th)
            # Unpack the state for printing with labels
            u_disc, k_disc, request_rate_disc, latency_95th_rounded = current_state
            print(f"Current State:")
            print(f"  - CPU Utilization Bin: {u_disc}")
            print(f"  - Number of Replicas: {k_disc}")
            print(f"  - Request Rate Bin: {request_rate_disc}")
            print(f"  - 95th Percentile Latency: {latency_95th_rounded}")

            # Check if response time exceeds Tmax
            if response_time > self.env.Tmax:
                self.response_time_exceed_count += 1

            action = self.agent.select_action(current_state, self.env.action_space)
            next_state, reward, new_threshold = self.env.step(
                action,
                cpu_utilization_percentage,
                response_time,  # Use real-time response time
                current_replicas,
                request_rate,
                latency_95th
            )

            # Calculate desired replicas based on the new threshold
            desired_replicas = math.ceil(current_replicas * (cpu_utilization_percentage / (new_threshold * 100)))
            desired_replicas = max(1, desired_replicas)  # Ensure at least one replica
            desired_replicas = min(desired_replicas, 9)  # Limit to maximum of 9 replicas

            self.scale_deployment(desired_replicas)

            self.agent.update_Q(current_state, action, reward, next_state, self.env.action_space)

            # Decay epsilon
            self.agent.epsilon = max(min_epsilon, self.agent.epsilon * epsilon_decay)

            print(f"CPU Utilization: {cpu_utilization_percentage:.2f}%, Current Replicas: {current_replicas}, "
                  f"New Threshold: {new_threshold:.2f}, Desired Replicas: {desired_replicas}, Reward: {reward:.2f}, Epsilon: {self.agent.epsilon:.4f}")
            print("##########")

            # Update averages
            self.total_replicas += current_replicas
            self.total_cpu_utilization += cpu_utilization_percentage
            self.iteration_count += 1

            # Print Q-table periodically
            current_time = time.time()
            if current_time - last_print_time >= q_table_print_interval:
                self.agent.print_q_table()
                avg_replicas = self.total_replicas / self.iteration_count
                avg_cpu_utilization = self.total_cpu_utilization / self.iteration_count
                print(f"Average Replicas: {avg_replicas:.2f}, Average CPU Utilization: {avg_cpu_utilization:.2f}%, "
                      f"Response Time Exceed Count: {self.response_time_exceed_count}, Iterations: {self.iteration_count}")
                last_print_time = current_time

            time.sleep(self.time_sleep)

            if self.agent.epsilon < 0.07:
                print("Agent finished training, starting optimal policy")
                self.iteration_count = 0
                self.response_time_exceed_count = 0
                self.total_cpu_utilization = 0
                self.time_sleep = 30


if __name__ == "__main__":
    autoscaler = KubernetesAutoscaler("cpu-intensive-app", "fastapi-app")
    autoscaler.run()
