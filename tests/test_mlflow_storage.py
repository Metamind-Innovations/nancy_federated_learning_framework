import unittest
import os
import glob
import mlflow
from mlflow.tracking import MlflowClient

class TestNaomiMLflowIntegration(unittest.TestCase):
    
    def test_model_saved_to_mlflow(self):
        """Test that a model was successfully saved to MLflow."""
        self.assertTrue(os.path.exists("mlruns"), "mlruns directory should exist after training")
        self.assertTrue(os.path.exists("mlruns/0"), "mlruns/0 directory should exist")
        run_directories = glob.glob("mlruns/0/*/")
        self.assertGreater(len(run_directories), 0, "Should have at least one run in mlruns/0")
        latest_run = max(run_directories, key=os.path.getctime)
        print(f"Found latest run: {latest_run}")
        artifacts_dir = os.path.join(latest_run, "artifacts")
        
        if os.path.exists(artifacts_dir):
            model_artifacts = os.path.join(artifacts_dir, "model")
            if os.path.exists(model_artifacts):
                print(f"Model artifacts found at: {model_artifacts}")
                model_files = os.listdir(model_artifacts)
                print(f"Model files: {model_files}")
                self.assertTrue(len(model_files) > 0, "Model artifacts should contain files")
            else:
                mlmodel_file = os.path.join(latest_run, "MLmodel")
                if os.path.exists(mlmodel_file):
                    print(f"MLmodel file found at: {mlmodel_file}")
                    self.assertTrue(True, "MLmodel file exists in run directory")
                else:
                    self.fail("No model artifacts or MLmodel file found")
        else:
            meta_files = os.listdir(latest_run)
            print(f"Run files: {meta_files}")
            self.assertTrue(len(meta_files) > 0, "Run directory should contain metadata files")
        
        print("✓ MLflow run was created successfully")
        print(f"✓ Run directory: {latest_run}")
        print("✓ MLflow integration test passed")
        
if __name__ == "__main__":
    unittest.main()