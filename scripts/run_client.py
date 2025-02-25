import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.client.client import start_client


def main():
    parser = argparse.ArgumentParser(description="Start a NANCY Federated Learning client")
    
    parser.add_argument(
        "--client-id",
        type=int,
        required=True,
        help="ID of the client (1, 2, or 3)"
    )
    
    parser.add_argument(
        "--server-address",
        type=str,
        default="127.0.0.1:8080",
        help="Server address in the format host:port"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the client data directory (default: based on client ID)"
    )
    
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="data/Standard Scaler/standard_scaler_new.joblib",
        help="Path to the StandardScaler joblib file"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of local training epochs"
    )
    
    args = parser.parse_args()
    
    if args.data_path is None:
        args.data_path = f"data/Client {args.client_id}"
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data path '{args.data_path}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(args.scaler_path):
        print(f"Error: Scaler path '{args.scaler_path}' does not exist.")
        sys.exit(1)
    
    print(f"Starting client {args.client_id} with server address {args.server_address}")
    print(f"Using data from {args.data_path}")
    
    start_client(
        client_id=args.client_id,
        server_address=args.server_address,
        data_path=args.data_path,
        scaler_path=args.scaler_path,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()

import os
import sys
import argparse


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.client.client import start_client


def main():
    parser = argparse.ArgumentParser(description="Start a NANCY Federated Learning client")
    
    parser.add_argument(
        "--client-id",
        type=int,
        required=True,
        help="ID of the client (1, 2, or 3)"
    )
    
    parser.add_argument(
        "--server-address",
        type=str,
        default="127.0.0.1:8080",
        help="Server address in the format host:port"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the client data directory (default: based on client ID)"
    )
    
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="data/Standard Scaler/standard_scaler_new.joblib",
        help="Path to the StandardScaler joblib file"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of local training epochs"
    )
    
    args = parser.parse_args()
    

    if args.data_path is None:
        args.data_path = f"data/Client {args.client_id}"
    

    if not os.path.exists(args.data_path):
        print(f"Error: Data path '{args.data_path}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(args.scaler_path):
        print(f"Error: Scaler path '{args.scaler_path}' does not exist.")
        sys.exit(1)
    

    print(f"Starting client {args.client_id} with server address {args.server_address}")
    print(f"Using data from {args.data_path}")
    
    start_client(
        client_id=args.client_id,
        server_address=args.server_address,
        data_path=args.data_path,
        scaler_path=args.scaler_path,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()