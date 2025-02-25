import os
import sys
import time
import argparse
import subprocess
from typing import List
import signal
import atexit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_process(command: List[str], name: str):
    print(f"Starting {name}...")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    return process


def log_output(processes: List[tuple]):
    while True:
        all_terminated = True
        for name, process in processes:
            if process.poll() is None:
                all_terminated = False
                
                output = process.stdout.readline()
                if output:
                    print(f"[{name}] {output.strip()}")
        
        if all_terminated:
            break
            
        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description="Run a complete NANCY Federated Learning experiment")
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port number for the server"
    )
    
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of local training epochs per client"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/final_model.keras", 
        help="Path to save the final model"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base directory containing client data"
    )
    
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="data/Standard Scaler/standard_scaler_new.joblib",
        help="Path to the StandardScaler joblib file"
    )
    
    args = parser.parse_args()
    
    for i in range(1, 4):
        client_path = os.path.join(args.data_dir, f"Client {i}")
        if not os.path.exists(client_path):
            print(f"Error: Data path '{client_path}' does not exist.")
            sys.exit(1)
    
    if not os.path.exists(args.scaler_path):
        print(f"Error: Scaler path '{args.scaler_path}' does not exist.")
        sys.exit(1)
    
    server_cmd = [
        sys.executable, "scripts/run_server.py",
        "--port", str(args.port),
        "--rounds", str(args.rounds),
        "--output-path", args.output_path
    ]
    
    client_cmds = []
    for i in range(1, 4):
        client_cmd = [
            sys.executable, "scripts/run_client.py",
            "--client-id", str(i),
            "--server-address", f"127.0.0.1:{args.port}",
            "--data-path", os.path.join(args.data_dir, f"Client {i}"),
            "--scaler-path", args.scaler_path,
            "--epochs", str(args.epochs)
        ]
        client_cmds.append(client_cmd)
    
    server_process = run_process(server_cmd, "Server")

    
    print("Waiting for server to initialize...")
    time.sleep(15)  

    
    client_processes = []
    for i, cmd in enumerate(client_cmds, 1):
        client_process = run_process(cmd, f"Client {i}")
        client_processes.append((f"Client {i}", client_process))
        
        time.sleep(3)
    
    all_processes = [("Server", server_process)] + client_processes
    
    def cleanup():
        print("\nCleaning up processes...")
        for name, process in all_processes:
            if process.poll() is None:
                print(f"Terminating {name}...")
                process.terminate()
    
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))
    
    log_output(all_processes)
    
    for name, process in all_processes:
        rc = process.wait()
        print(f"{name} exited with return code {rc}")


if __name__ == "__main__":
    main()