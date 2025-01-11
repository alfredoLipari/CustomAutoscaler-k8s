import time
import math
import requests
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict
from collections import deque
import threading
from queue import Queue, Empty
import random


SEED = 100
random.seed(SEED)
np.random.seed(SEED)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

@dataclass
class CycleParameters:
    amplitude_mod: float
    phase_shift: float
    secondary_waves: List[Dict[str, float]]

class WorkloadGenerator:
    def __init__(self):
        # Parameters matching the test environment
        self.base_rate = 80.0    # Base rate in req/min
        self.min_rate = 3.0      # Minimum rate in req/min
        self.max_rate = 260.0    # Maximum rate in req/min
        self.amplitude = 120.0   # Amplitude in req/min
        self.trend_duration = 1800  # 30 minutes in seconds
        self.smoothing_factor = 0.95

        # Internal state
        self.random_walk = 0.0
        self.previous_rate = None
        self.cycle_variations = {}
        self.request_times = deque()
        self.response_times = []  # To store response times
        self.slo_violations = 0   # Count of SLO violations
        self.total_requests = 0   # Total requests sent
        self.url = self.url = "http://cpu-intensive-app.fastapi-app.svc.cluster.local:80/"  # Replace with your service URL

        # Concurrency settings
        self.num_threads = 50  # Adjust the number of threads as needed
        self.request_queue = Queue()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def _generate_cycle_parameters(self) -> CycleParameters:
        return CycleParameters(
            amplitude_mod=np.random.uniform(1.0, 1.5),  # Increased variation
            phase_shift=np.random.uniform(-0.3, 0.3),   # Increased phase variation
            secondary_waves=[
                {
                    'amplitude': (0.05 + np.random.random() * 0.25) * 60.0,  # Reduced amplitude
                    'frequency': 0.21+ np.random.random() * 0.2,  # Lower frequencies
                    'phase': np.random.uniform(0, 2 * np.pi)
                }
                for _ in range(2)  # Reduced to 2 secondary waves
            ]
        )

    def _calculate_rate(self, position: float, cycle_params: CycleParameters) -> float:
        """Calculate the desired request rate at a given position in the cycle."""
        # Main wave
        main_wave = math.sin(2 * math.pi * position + cycle_params.phase_shift)
        base_rate = self.base_rate + (main_wave * self.amplitude * cycle_params.amplitude_mod)

        # Add secondary waves
        for wave in cycle_params.secondary_waves:
            base_rate += (
                wave['amplitude'] * 
                math.sin(wave['frequency'] * 2 * math.pi * position + wave['phase'])
            )

        return base_rate

    def _get_instant_rate(self, elapsed_seconds: float) -> float:
        """Get the instantaneous desired request rate."""
        current_cycle = int(elapsed_seconds // self.trend_duration)
        position = (elapsed_seconds % self.trend_duration) / self.trend_duration

        # Generate new cycle parameters if needed
        if current_cycle not in self.cycle_variations:
            self.cycle_variations[current_cycle] = self._generate_cycle_parameters()

            # Clean up old cycles
            old_cycles = [
                cycle for cycle in self.cycle_variations.keys() 
                if cycle < current_cycle - 1
            ]
            for cycle in old_cycles:
                del self.cycle_variations[cycle]

        # Calculate base rate
        base_rate = self._calculate_rate(position, self.cycle_variations[current_cycle])

        # Apply random walk
        self.random_walk = np.clip(
            self.random_walk + np.random.normal(0, 0.5),
            -0.2, 0.2
        ) * 60.0

        instant_rate = base_rate + self.random_walk

        # Apply smoothing
        if self.previous_rate is not None:
            instant_rate = (
                self.smoothing_factor * self.previous_rate +
                (1 - self.smoothing_factor) * instant_rate
            )

        # Apply bounds
        instant_rate = np.clip(instant_rate, self.min_rate, self.max_rate)
        self.previous_rate = instant_rate

        return instant_rate

    def worker(self):
        while not self.stop_event.is_set():
            try:
                # Wait for a task from the queue
                delay = self.request_queue.get(timeout=1)
                try:
                    # Sleep for the delay
                    time.sleep(delay)
                    # Make the request
                    res = requests.get(self.url, timeout=5)
                    status_code = res.status_code
                    current_time = time.time()

                    # Parse response
                    try:
                        if res.headers.get("Content-Type") == "application/json":
                            response_json = res.json()
                            response_time = float(response_json.get("time_taken_seconds", 0.0))
                        else:
                            logging.error(f"Non-JSON response received: {res.text}")
                            response_time = 0.0
                    except Exception as e:
                        logging.error(f"Error parsing response: {e}")
                        response_time = 0.0

                    # Update metrics
                    with self.lock:
                        self.total_requests += 1
                        self.response_times.append(response_time)
                        if response_time > 0.7 * 1.1: # SLO violation with 10% margin
                            self.slo_violations += 1

                    # Log status
                    logging.info(
                        f"Response Time: {response_time:.2f}s, Status Code: {status_code}"
                    )

                    logging.info(
                        f"Current Time: {current_time:.2f}, Total Requests: {self.total_requests}, Total SLO Violations: {self.slo_violations}"
                    )

                except Exception as e:
                    logging.error(f"Request failed: {e}")

                finally:
                    self.request_queue.task_done()

            except Empty:
                continue  # No task available, loop again

            except Exception as e:
                logging.error(f"Worker exception: {e}")

    def run(self, duration_minutes=30):
        """Run the workload generator."""
        start_time = time.time()

        # Start worker threads
        threads = []
        for _ in range(self.num_threads):
            t = threading.Thread(target=self.worker)
            t.daemon = True  # Allows the program to exit even if threads are still running
            t.start()
            threads.append(t)

        try:
            while time.time() - start_time < duration_minutes * 60:
                elapsed_seconds = time.time() - start_time

                # Get current desired rate
                current_rate = self._get_instant_rate(elapsed_seconds)
                logging.info(f"Current Rate: {current_rate:.2f} req/min")
                requests_per_second = current_rate / 60.0

                # Schedule requests for the next second
                for _ in range(int(requests_per_second)):
                    self.request_queue.put(0)

                # Handle fractional requests
                if np.random.rand() < (requests_per_second % 1):
                    self.request_queue.put(0)

                # Sleep for 1 second before scheduling the next batch
                time.sleep(1)

        except KeyboardInterrupt:
            logging.info("Workload generator interrupted by user.")

        finally:
            # Signal threads to stop
            self.stop_event.set()
            self.request_queue.join()
            for t in threads:
                t.join()

            # Calculate final metrics
            self.calculate_final_metrics()

    def calculate_final_metrics(self):
        """Calculate and log final metrics."""
        avg_response_time = np.mean(self.response_times) if self.response_times else 0
        logging.info("Workload Completed.")
        logging.info(f"Total Requests: {self.total_requests}")
        logging.info(f"Average Response Time: {avg_response_time:.2f}s")
        logging.info(f"SLO Violations (>0.7s): {self.slo_violations}")


if __name__ == "__main__":
    generator = WorkloadGenerator()
    generator.run(duration_minutes=180)  # Run for the desired duration
