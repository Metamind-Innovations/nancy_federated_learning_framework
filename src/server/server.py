import os
import flwr as fl
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from ..model.neural_network import create_model
import mlflow


class SaveModelStrategy(fl.server.strategy.FedAdagrad):
    
    def __init__(
        self,
        min_available_clients: int,
        initial_parameters,
        output_path: str = "models/final_model.keras",
        num_rounds: int = 10,
        **kwargs
    ):
        
        self.input_shape = kwargs.pop("input_shape", 76)
        self.num_classes = kwargs.pop("num_classes", 7)
        self.output_path = output_path
        self.num_rounds = num_rounds
        
        
        super().__init__(
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            **kwargs
        )
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
        ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        print(f"Completed round {server_round} of {self.num_rounds}")
        
        if aggregated_parameters is not None and server_round == self.num_rounds:
            try:
                print(f"Attempting to save model to {self.output_path}...")
                
                model_path = self.output_path
                if not model_path.endswith('.keras') and not model_path.endswith('.h5'):
                    model_path = f"{model_path}.keras"
                
                aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
                
                model = create_model(input_shape=self.input_shape, num_classes=self.num_classes)
                model.set_weights(aggregated_weights)
                
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                model.save(model_path)
                print(f"SUCCESS: Model saved to {model_path}")
                
                # Save to MLflow
                mlflow.set_tracking_uri("http://localhost:5000")  # Use local MLflow server
                with mlflow.start_run() as run:
                    mlflow.keras.log_model(
                        model,
                        "model",
                        registered_model_name="nancy_federated_model"
                    )
                    if metrics:
                        mlflow.log_metrics(metrics)
                    mlflow.log_params({
                        "input_shape": self.input_shape,
                        "num_classes": self.num_classes,
                        "num_rounds": self.num_rounds
                    })
                print("SUCCESS: Model logged and registered to MLflow")
            except Exception as e:
                print(f"ERROR saving model: {e}")
                import traceback
                traceback.print_exc()
        
        return aggregated_parameters, metrics
        
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        if not results:
            return None, {}
        
        accuracy_aggregated = weighted_average([res.metrics["accuracy"] for _, res in results], 
                                               [res.num_examples for _, res in results])
        
        metrics = {
            "accuracy": accuracy_aggregated,
        }
        
        for idx, (client_proxy, res) in enumerate(results):
            client_id = res.metrics.get("client_id", f"unknown_{idx}")
            metrics[f"client_{client_id}_accuracy"] = res.metrics.get("accuracy", 0)
            metrics[f"client_{client_id}_tpr"] = res.metrics.get("tpr", 0)
            metrics[f"client_{client_id}_fpr"] = res.metrics.get("fpr", 0)
            metrics[f"client_{client_id}_f1"] = res.metrics.get("f1", 0)
            metrics[f"client_{client_id}_auc"] = res.metrics.get("auc", 0)
        
        return accuracy_aggregated, metrics


def weighted_average(metrics: List[float], weights: List[int]) -> float:
    return sum(m * w for m, w in zip(metrics, weights)) / sum(weights)


def start_server(
    port: int = 8080,
    rounds: int = 10,
    min_clients: int = 3,
    input_shape: int = 76,
    num_classes: int = 7,
    output_path: str = "models/final_model.keras"
) -> None:
    try:
        
        if not output_path.endswith('.keras') and not output_path.endswith('.h5'):
            output_path = f"{output_path}.keras"
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        model = create_model(input_shape=input_shape, num_classes=num_classes)
        initial_weights = model.get_weights()
        initial_parameters = fl.common.ndarrays_to_parameters(initial_weights)
        
        strategy = SaveModelStrategy(
            min_available_clients=min_clients,
            initial_parameters=initial_parameters,
            input_shape=input_shape,
            num_classes=num_classes,
            output_path=output_path,
            num_rounds=rounds
        )
        
        print(f"Starting Flower server on 0.0.0.0:{port}")
        fl.server.start_server(
            server_address=f"0.0.0.0:{port}",
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy
        )
    except Exception as e:
        print(f"Error in start_server: {e}")
        import traceback
        traceback.print_exc()
        raise