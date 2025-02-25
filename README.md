# 🌟 NANCY Federated Learning Framework 🌟

Welcome to the NANCY Federated Learning Framework!

This repository contains a complete implementation of a federated learning system for network traffic anomaly detection, built using Flower (Flwr) framework. The system enables distributed training across multiple nodes while keeping data localized, ensuring privacy while benefiting from collaborative model training.

## 📊 Network Anomaly Detection Component

This framework is designed to train deep learning models that classify network flows into seven categories:

* Benign Traffic
* Reconnaissance Attack
* TCP Scan
* SYN Scan
* SYN Flood
* HTTP Flood
* Slowrate DoS

## 📁 Dataset

The models are trained on the [NANCY SNS JU Project - Cyberattacks on O-RAN 5G Testbed Dataset](https://zenodo.org/records/14811122) available on Zenodo. This dataset contains network traffic flows with various features including packet lengths, inter-arrival times, flag counts, and other network-specific metrics.

## 🏗️ Architecture

The system follows a federated learning architecture with:

- **Central Server**: Coordinates the federated learning process, aggregates model updates, and distributes the global model
- **Clients (1-3)**: Train local models on their private data, send model updates to the server
- **Model Storage**: Saves the final trained model for deployment and inference

The implementation uses the FedAdagrad strategy to optimize the model aggregation process across clients.

## 🛠️ Installation

1. Clone this repository:

```bash
git clone https://github.com/username/nancy-federated-learning.git
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Download the dataset from Zenodo and place it in the appropriate directories (see Data Setup section).

## 🐳 Docker Deployment

The system can be deployed using Docker containers for easy setup and reproducibility:

### Build Images

```bash
# Build all images using docker-compose
docker-compose build

# Or build images individually
docker build -t nancy-fl-server -f docker/server.Dockerfile .
docker build -t nancy-fl-client -f docker/client.Dockerfile .
```

### Run with Docker Compose (Recommended)

The easiest way to deploy the system is with Docker Compose:

```bash
# Start the server and all clients
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f
```
This will:

- Start the server with 10 training rounds
- Launch 3 clients that connect to the server
- Mount the data directories from your local machine
- Save the trained model to your local models directory

### Run Containers Manually

Alternatively, you can run the containers individually:

```bash
# Start server
docker run -p 8080:8080 -v $(pwd)/models:/app/models nancy-fl-server --rounds 10 --min-clients 3

# Start clients (in separate terminals)
docker run -v $(pwd)/data/Client\ 1:/app/data -v $(pwd)/data/Standard\ Scaler:/app/scaler nancy-fl-client --client-id 1 --server-address server:8080 --data-path /app/data --scaler-path /app/scaler/standard_scaler_new.joblib

docker run -v $(pwd)/data/Client\ 2:/app/data -v $(pwd)/data/Standard\ Scaler:/app/scaler nancy-fl-client --client-id 2 --server-address server:8080 --data-path /app/data --scaler-path /app/scaler/standard_scaler_new.joblib

docker run -v $(pwd)/data/Client\ 3:/app/data -v $(pwd)/data/Standard\ Scaler:/app/scaler nancy-fl-client --client-id 3 --server-address server:8080 --data-path /app/data --scaler-path /app/scaler/standard_scaler_new.joblib
```
### Checking Results

After training completes:

- The server will save the model to models/final_model.keras
- You can inspect the model file in your local directory
- Client logs will show metrics for the training process
- Server logs will show the completion of rounds and the final model saving

#### Notes

- The server container will exit with code 0 after completing all rounds
- Clients may show connection errors after the server exits - this is normal
- Mount volumes to persist data and models between container runs
- Use Docker Compose for the simplest deployment experience

## 📂 Data Setup

The system expects data to be organized as follows:

```
data/
├── Client 1/
│   ├── Train.csv
│   └── Test.csv
├── Client 2/
│   ├── Train.csv
│   └── Test.csv
└── Client 3/
    ├── Train.csv
    └── Test.csv
```

You can specify custom data paths when running the clients using command line arguments.

## 🚀 Usage

### Running the Server

Start the central federated learning server:

```bash
python scripts/run_server.py --port 8080 --rounds 10 --min-clients 3
```

Arguments:
- `--port`: Port number for the server (default: 8080)
- `--rounds`: Number of federated learning rounds (default: 10)
- `--min-clients`: Minimum number of clients to start training (default: 3)
- `--output-path`: Path to save the final model (default: "models/final_model")

### Running the Clients

Run each client on different machines or in different terminals:

```bash
# Client 1
python scripts/run_client.py --client-id 1 --server-address 127.0.0.1:8080 --data-path data/Client\ 1/ --scaler-path data/Standard\ Scaler/standard_scaler_new.joblib

# Client 2
python scripts/run_client.py --client-id 2 --server-address 127.0.0.1:8080 --data-path data/Client\ 2/ --scaler-path data/Standard\ Scaler/standard_scaler_new.joblib

# Client 3
python scripts/run_client.py --client-id 3 --server-address 127.0.0.1:8080 --data-path data/Client\ 3/ --scaler-path data/Standard\ Scaler/standard_scaler_new.joblib
```

Arguments:
- `--client-id`: ID of the client (required)
- `--server-address`: Address of the server (default: "127.0.0.1:8080")
- `--data-path`: Path to the client data directory (default: based on client ID)
- `--scaler-path`: Path to the StandardScaler joblib file (required)
- `--epochs`: Number of local training epochs (default: 5)

### Running a Complete Experiment

For local testing, you can run the entire system (server + 3 clients) using:

```bash
python scripts/run_experiment.py --rounds 10 --epochs 5
```

This script will start the server and all clients in separate processes.

## 🧪 Testing

To run the tests:

```bash
pytest tests/
```

This will validate the functionality of individual components and their communication.

## 🔄 Model Export

After training, the final model is saved and can be exported for integration with other components. The server automatically saves the final global model after training completes.

## 📜 License

MIT License

Copyright (c) 2025 NANCY Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.