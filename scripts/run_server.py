import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.server.server import start_server


def main():
    parser = argparse.ArgumentParser(description="Start a NANCY Federated Learning server")
    
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
        "--min-clients",
        type=int,
        default=3,
        help="Minimum number of clients to start training"
    )
    
    parser.add_argument(
        "--input-shape",
        type=int,
        default=76,
        help="Number of input features for the model"
    )
    
    parser.add_argument(
        "--num-classes",
        type=int,
        default=7,
        help="Number of output classes for the model"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/final_model.keras",
        help="Path to save the final model"
    )
    
    args = parser.parse_args()
    
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print(f"Starting server on port {args.port}")
    print(f"Running for {args.rounds} rounds with minimum {args.min_clients} clients")
    print(f"Final model will be saved to {args.output_path}")
    
    try:
        start_server(
            port=args.port,
            rounds=args.rounds,
            min_clients=args.min_clients,
            input_shape=args.input_shape,
            num_classes=args.num_classes,
            output_path=args.output_path
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()