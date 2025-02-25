import os
import argparse
import flwr as fl
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from typing import Dict, List, Tuple, Optional
from ..model.neural_network import create_model
from .metrics import evaluation_metrics

class NancyClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        train_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
        num_classes: int,
        epochs: int = 5
    ):
        self.client_id = client_id
        self.X_train, self.y_train = train_data
        self.X_test, self.y_test = test_data
        self.num_classes = num_classes
        self.epochs = epochs
        self.model = create_model(input_shape=self.X_train.shape[1], num_classes=num_classes)
        
    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return self.model.get_weights()

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        self.model.set_weights(parameters)
        
        epochs = int(config.get("epochs", self.epochs))
        batch_size = int(config.get("batch_size", 32))
        
        history = self.model.fit(
            self.X_train, 
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return self.model.get_weights(), len(self.X_train), {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
            "client_id": self.client_id
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict[str, float]]:
        self.model.set_weights(parameters)
        
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        predicted_test = self.model.predict(self.X_test)
        predicted_classes = np.argmax(predicted_test, axis=1)
        
        tpr, fpr, f1, auc = evaluation_metrics(self.y_test, predicted_classes, predicted_test)
        
        return loss, len(self.X_test), {
            "accuracy": accuracy,
            "tpr": tpr,
            "fpr": fpr,
            "f1": f1, 
            "auc": auc,
            "client_id": self.client_id
        }
    
    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")


def load_data(data_path: str, scaler_path: str) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    df_train = pd.read_csv(os.path.join(data_path, "Train.csv"))
    df_test = pd.read_csv(os.path.join(data_path, "Test.csv"))
    
    columns_to_drop = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port',
                      'Protocol', 'Timestamp']
    df_train_reduced = df_train.drop(columns_to_drop, axis=1)
    df_test_reduced = df_test.drop(columns_to_drop, axis=1)
    
    df_train_reduced.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_train_no_na = df_train_reduced.dropna(axis=0)
    
    df_test_reduced.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test_no_na = df_test_reduced.dropna(axis=0)
    
    y_train = df_train_no_na['Label']
    X_train = df_train_no_na.drop(['Label'], axis=1)
    
    y_test = df_test_no_na['Label']
    X_test = df_test_no_na.drop(['Label'], axis=1)
    
    scaler = joblib.load(scaler_path, mmap_mode=None)
    
    X_scaled_train = scaler.transform(X_train)
    X_scaled_train = pd.DataFrame(X_scaled_train, columns=X_train.columns)
    
    X_scaled_test = scaler.transform(X_test)
    X_scaled_test = pd.DataFrame(X_scaled_test, columns=X_test.columns)
    
    return (X_scaled_train, y_train), (X_scaled_test, y_test)


def start_client(
    client_id: int,
    server_address: str,
    data_path: str,
    scaler_path: str,
    epochs: int = 5
) -> None:
    import time  
    
    (X_train, y_train), (X_test, y_test) = load_data(data_path, scaler_path)
    
    num_classes = len(np.unique(y_train))
    print(f"Client {client_id} initialized with {len(X_train)} training samples and {num_classes} classes")
    
    client = NancyClient(
        client_id=client_id,
        train_data=(X_train, y_train),
        test_data=(X_test, y_test),
        num_classes=num_classes,
        epochs=epochs
    )
    
    
    max_retries = 5
    retry_delay = 5  
    
    for attempt in range(max_retries):
        try:
            print(f"Connecting to server (attempt {attempt+1}/{max_retries})...")
            fl.client.start_numpy_client(
                server_address=server_address,
                client=client
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Connection failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to connect after {max_retries} attempts: {e}")
                raise