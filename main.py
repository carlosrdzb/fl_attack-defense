import  subprocess
import  time

def start_server():
    """Starts the Flower federated learning server."""
    print("Starting Federated Learning Server...")
    subprocess.Popen(["python", "server.py"])  # Run server.py in a separate process
    time.sleep(5)  # Wait for the server to start before launching clients

def start_clients(num_clients=2):
    """Starts multiple federated clients."""
    print(f"Starting {num_clients} clients...")
    clients = []
    for i in range(num_clients):
        clients.append(subprocess.Popen(["python", "client.py"]))  # Run each client in a new process
        time.sleep(2)  # Slight delay to avoid connection issues
    return clients

if __name__ == "__main__":
    start_server()
    clients = start_clients(num_clients=2)

    print("Federated Learning Simulation Started. Press CTRL+C to stop.")
