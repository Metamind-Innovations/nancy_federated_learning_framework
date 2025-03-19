import numpy as np
import mlflow
import ray
from ray import serve
from fastapi import FastAPI, HTTPException
from typing import List

app = FastAPI(debug=True)

NAOMI_IP = "localhost"

@serve.deployment(
    name="nancy_federated_model",
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
    autoscaling_config={"min_replicas": 1, "max_replicas": 1}
)
@serve.ingress(app)
class NancyPredictor:
    def __init__(self):
        mlflow.set_tracking_uri(f"http://{NAOMI_IP}:5000")
        self.model = mlflow.pyfunc.load_model("models:/nancy_federated_model/latest")
        print("Model loaded successfully")

    @app.post("/")
    async def predict(self, data: List[List[float]]):
        try:
            instances = np.array(data)
            predictions = self.model.predict(instances)
            return {"predictions": predictions.tolist()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    ray.init(f"ray://{NAOMI_IP}:10001", ignore_reinit_error=True)
    serve.run(NancyPredictor.bind(), name="nancy_federated_model", route_prefix="/nancy")
    print(f"Service deployed at: http://{NAOMI_IP}/ray-api/nancy/")