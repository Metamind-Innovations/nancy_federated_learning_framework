import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import flwr as fl

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.client.client import load_data, start_client
from src.server.server import start_server


class TestCommunication(unittest.TestCase):
    
    def test_server_initialization(self):
        """Test server initialization without actually starting a server."""
        mock_model = MagicMock()
        mock_model.get_weights.return_value = [np.ones((5, 5)), np.zeros((3, 3))]

        with patch('src.model.neural_network.create_model', return_value=mock_model):
            with patch('flwr.server.start_server') as mock_start_server:
                
                start_server(
                    port=8090,
                    rounds=1,
                    min_clients=1,
                    input_shape=10,
                    num_classes=3,
                    output_path="test_dir/test_model.keras"
                )
    
    def test_client_initialization(self):
        """Test client initialization without actually starting a client."""
        with patch('flwr.client.start_numpy_client') as mock_start_client:
            with patch('src.client.client.load_data') as mock_load_data:

                X_train = np.random.rand(100, 10)
                y_train = np.random.randint(0, 3, 100)
                X_test = np.random.rand(50, 10)
                y_test = np.random.randint(0, 3, 50)
                
                mock_load_data.return_value = ((X_train, y_train), (X_test, y_test))
                

                start_client(
                    client_id=1,
                    server_address="127.0.0.1:8090",
                    data_path="fake_path",
                    scaler_path="fake_scaler.joblib",
                    epochs=1
                )
                

                mock_start_client.assert_called_once()
                args, kwargs = mock_start_client.call_args
                

                self.assertEqual(kwargs.get('server_address'), "127.0.0.1:8090")
    
    def test_run_experiment_imports(self):
        """Test that run_experiment script can be imported."""

        import scripts.run_experiment
        self.assertTrue(hasattr(scripts.run_experiment, 'main'))


if __name__ == '__main__':
    unittest.main()