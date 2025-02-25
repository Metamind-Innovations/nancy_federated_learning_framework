import os
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.client.client import NancyClient


class TestNancyClient(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.num_samples = 100
        self.num_features = 10
        self.num_classes = 3
        
        self.X_train = np.random.rand(self.num_samples, self.num_features)
        self.y_train = np.random.randint(0, self.num_classes, self.num_samples)
        
        self.X_test = np.random.rand(self.num_samples // 2, self.num_features)
        self.y_test = np.random.randint(0, self.num_classes, self.num_samples // 2)
        
        self.client = NancyClient(
            client_id=1,
            train_data=(self.X_train, self.y_train),
            test_data=(self.X_test, self.y_test),
            num_classes=self.num_classes,
            epochs=1
        )
    
    def test_get_parameters(self):
        params = self.client.get_parameters({})
        
        self.assertIsInstance(params, list)
        for param in params:
            self.assertIsInstance(param, np.ndarray)
    
    def test_fit(self):
        initial_params = self.client.get_parameters({})
        
        updated_params, num_examples, metrics = self.client.fit(initial_params, {"epochs": 1})
        
        self.assertIsInstance(updated_params, list)
        self.assertEqual(num_examples, self.num_samples)
        self.assertIsInstance(metrics, dict)
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
    
    def test_evaluate(self):
        params = self.client.get_parameters({})
        
        loss, num_examples, metrics = self.client.evaluate(params, {})
        
        self.assertIsInstance(loss, float)
        self.assertEqual(num_examples, self.num_samples // 2)
        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)
        self.assertIn("tpr", metrics)
        self.assertIn("fpr", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("auc", metrics)
    
    @patch('tensorflow.keras.models.Sequential.save')
    def test_save_model(self, mock_save):
        with patch('os.makedirs'):
            self.client.save_model("test/path")
            
            mock_save.assert_called_once_with("test/path")


if __name__ == '__main__':
    unittest.main()