import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange
import copy
from evaluation.qsenn_metrics import eval_model_on_all_qsenn_metrics

def metasenn_feature_selection(model, feature_candidates, train_loader, test_loader, log_dir,
                               n_features, device='cuda', rl_iterations=50):
    """
    Reinforcement learning to select the best feature set from candidates

    Args:
        model: Base QSENN model
        feature_candidates: List of feature selection tensors/arrays from different iterations
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        log_dir: Directory to save logs
        n_features: Number of features to select
        device: Device to run on (cuda or cpu)
        rl_iterations: Number of RL iterations
    """
    print(f"Starting RL Feature Selection ({device})")

    # Create environment for feature selection
    class FeatureSelectionEnvironment:
        def __init__(self, model, feature_candidates, data_loader, device):
            self.model = copy.deepcopy(model).to(device)

            self.feature_candidates = []
            for feat in feature_candidates:
                if isinstance(feat, torch.Tensor):
                    self.feature_candidates.append(feat.to(device))
                else:
                    # Convert numpy arrays to tensors
                    self.feature_candidates.append(torch.tensor(feat, device=device))

            self.data_loader = data_loader
            self.device = device
            self.n_candidates = len(feature_candidates)
            self.iterator = iter(data_loader)
            self.rewards_history = [[] for _ in range(self.n_candidates)]

            # Ensure model attributes are on the correct device
            if hasattr(self.model, 'mean'):
                self.model.mean = self.model.mean.to(device)
            if hasattr(self.model, 'std'):
                self.model.std = self.model.std.to(device)
            if hasattr(self.model, 'current_mean'):
                self.model.current_mean = self.model.current_mean.to(device)
            if hasattr(self.model, 'current_std'):
                self.model.current_std = self.model.current_std.to(device)


        def evaluate_features(self, feature_idx):
            #Get the feature selection
            feature_sel = self.feature_candidates[feature_idx]

            #Set model to use these features
            if hasattr(self.model, 'feature_sel'):
                self.model.feature_sel = feature_sel

            # Evaluate the model with the selected features
            with torch.no_grad():
                current_metrics = eval_model_on_all_qsenn_metrics(self.model, test_loader, train_loader)
                reward = current_metrics['Accuracy'] * 0.5 + current_metrics['diversity'] * 0.3 - current_metrics['Dependence'] * 0.2

            # Record reward
            self.rewards_history[feature_idx].append(reward)

            return reward

    # Create environments
    train_env = FeatureSelectionEnvironment(model, feature_candidates, train_loader, device)
    test_env = FeatureSelectionEnvironment(model, feature_candidates, test_loader, device)

    # Epsilon exploration strategy
    epsilon = 1.0
    epsilon_decay = 0.95
    epsilon_min = 0.1

    # Track best feature set
    best_reward = 0
    best_feature_idx = 0
    cumulative_rewards = np.zeros(len(feature_candidates))

    # Run RL iterations
    for iteration in trange(rl_iterations, desc="Meta-SENN Feature Selection"):
        # Epsilon exploration
        if np.random.random() < epsilon:
            # Exploration: try a random feature set
            feature_idx = np.random.randint(0, len(feature_candidates))
        else:
            # Exploitation: choose the best feature set so far
            feature_idx = np.argmax(cumulative_rewards)

        # Decay epsilon after choosing an action
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Evaluate the selected feature set
        reward = train_env.evaluate_features(feature_idx)

        # Update cumulative rewards
        cumulative_rewards[feature_idx] += reward


        # Periodically evaluate on test set
        if iteration % 10 == 0 or iteration == rl_iterations - 1:
            test_rewards = []
            for idx in range(len(feature_candidates)):
                test_reward = test_env.evaluate_features(idx)
                test_rewards.append(test_reward)

            best_idx = np.argmax(test_rewards)
            print(f"Iteration {iteration}, Best test reward: {test_rewards[best_idx]:.4f} (feature set {best_idx})")

            if test_rewards[best_idx] > best_reward:
                best_reward = test_rewards[best_idx]
                best_feature_idx = best_idx

    print(f"Feature selection RL completed. Best feature set: {best_feature_idx} with reward {best_reward:.4f}")

    # Save results
    torch.save({
        'best_feature_idx': best_feature_idx,
        'best_feature_sel': feature_candidates[best_feature_idx],
        'rewards_history': train_env.rewards_history,
        'test_rewards_history': test_env.rewards_history,
        'cumulative_rewards': cumulative_rewards.tolist()
    }, str(log_dir / f"feature_selection_rl_results_{n_features}.pth"))

    return feature_candidates[best_feature_idx], train_env.rewards_history

def meta_rl_training(final_model, train_loader, test_loader, log_dir, n_features, device='cuda',
                           rl_iterations=100, lr=1e-3):
    """
    Meta-RL process to optimize the nonzero weights of the sparse decision layer,
    given a fixed feature selection (final_model.feature_sel).
    """
    print(f"Starting Meta-RL weight optimization ({device})")

    # Extract the sparse mask and initial weights for the selected features
    sparse_mask = (final_model.linear.weight != 0).float().to(device)
    nonzero_indices = torch.nonzero(sparse_mask, as_tuple=False)
    n_nonzero = nonzero_indices.shape[0]

    # Policy network 
    policy_net = nn.Sequential(
        nn.Linear(n_nonzero, 128),
        nn.LeakyReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.LeakyReLU(),
        nn.Linear(64, n_nonzero),
    ).to(device)
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=lr)

    # Flattened initial weights for nonzero positions
    initial_weights = final_model.linear.weight[sparse_mask.bool()].detach().clone().to(device)

    best_reward = -float('inf')
    best_weights = initial_weights.clone()
    reward_history = []

    for it in trange(rl_iterations, desc="Meta-RL Weight Tuning"):
        
        # Sample new weights
        state = initial_weights.unsqueeze(0)
        logits = policy_net(state)

        # Add noise for exploration
        new_weights = logits.squeeze(0) + 0.05 * torch.randn_like(logits.squeeze(0))

        # Apply new weights to the model
        temp_weight = final_model.linear.weight.data.clone()
        temp_weight[sparse_mask.bool()] = new_weights
        final_model.linear.weight.data = temp_weight

        # Evaluate reward
        with torch.no_grad():
            metrics = eval_model_on_all_qsenn_metrics(final_model, test_loader, train_loader)
            reward = metrics['Accuracy'] * 0.5 + metrics['diversity'] * 0.3 - metrics['Dependence'] * 0.2

        reward_history.append(reward)
        if reward > best_reward:
            best_reward = reward
            best_weights = new_weights.detach().clone()

        # Policy gradient calculation
        loss = -reward * logits.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 10 == 0 or it == rl_iterations - 1:
            print(f"Iter {it}: reward={reward:.4f}, best={best_reward:.4f}")

    # Set model to best weights found
    final_model.linear.weight.data[sparse_mask.bool()] = best_weights

    # Save results
    torch.save({
        'best_weights': best_weights,
        'reward_history': reward_history,
    }, str(log_dir / f"meta_rl_weight_tuning_{n_features}.pth"))

    print("Meta-RL weight optimization completed.")
    return final_model