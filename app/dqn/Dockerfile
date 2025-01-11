FROM python:3.10-slim

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your workload generator script:
COPY workload_simulator.py .

# We do NOT expose a port because this is a client script, not a server
# EXPOSE 8000 (not needed for a load tester)

# Run the Python script
CMD ["python", "workload_simulator.py"]
