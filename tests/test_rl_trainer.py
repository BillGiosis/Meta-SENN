import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add project root to path

import unittest
import torch
from meta_rl.rl_trainer import meta_rl_training, TaskEnvironment, MetaRLModel
from torch.utils.data import DataLoader, TensorDataset


class TestRLTrainer(unittest.TestCase):
    def setUp(self):
        # Create dummy data and model for testing
        self.n_features = 10
        self.n_classes = 5
        self.batch_size = 8

        # Dummy data
        self.dummy_data = torch.randn(16, 3, 32, 32)
        self.dummy_labels = torch.randint(0, self.n_classes, (16,))
        self.dataset = TensorDataset(self.dummy_data, self.dummy_labels)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size)

        # Dummy model (replace with a minimal, testable model)
        self.base_model = torch.nn.Linear(3*32*32, self.n_classes)  # Simple linear model
        self.meta_model = MetaRLModel(self.base_model, self.n_features)

        # Example state
        self.state = torch.randn(self.batch_size, self.n_features)

        # Task environment
        self.task = TaskEnvironment(self.data_loader)

    def test_meta_rl_training(self):
        # Call meta_rl_training with dummy data
        final_model, meta_rewards = meta_rl_training(
            final_model=self.base_model,  # Pass the base model
            train_loader=self.data_loader,
            test_loader=self.data_loader,
            log_dir="./test_logs",  # Dummy log directory
            n_features=self.n_features,
            rl_iterations=1,  # Reduced iterations for testing
            task_num=1,
            inner_steps=1,
            outer_steps=1
        )

        # Assert that the returned model is not None
        self.assertIsNotNone(final_model)
        self.assertIsInstance(meta_rewards, list)

    def test_get_episode(self):
        # Get episode and assert that reward is a float
        reward = self.task.get_episode(self.meta_model)
        self.assertIsInstance(reward, float)

    def test_policy_net_output_shape(self):
        # Check the output shape of the policy network
        action = self.meta_model.policy_net(self.state)
        self.assertEqual(action.shape, (self.batch_size, self.n_classes))


if __name__ == '__main__':
    unittest.main()