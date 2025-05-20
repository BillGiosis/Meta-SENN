import os
import numpy as np
import torch

from finetuning.utils import train_n_epochs
from sparsification.qsenn import compute_qsenn_feature_selection_and_assignment
from meta_rl.rl_trainer import metasenn_feature_selection, meta_rl_training


def finetune_qsenn(model, train_loader, test_loader, log_dir, n_classes, seed, beta, optimization_schedule, n_features, n_per_class, meta_rl=False, meta_iterations=50, device='cuda'):
    # Initialize all arrays needed to store canditates for meta-RL feature selection
    feature_candidates = []
    sparse_candidates = []
    bias_candidates = []
    mean_candidates = []
    std_candidates = []

    # Run Q-SENN fine-tuning to get multiple candidate feature selections
    for iteration_epoch in range(4):
        print(f"Starting iteration epoch {iteration_epoch}")
        this_log_dir = log_dir / f"iteration_epoch_{iteration_epoch}"
        this_log_dir.mkdir(parents=True, exist_ok=True)
        feature_sel, sparse_layer, bias_sparse, current_mean, current_std = compute_qsenn_feature_selection_and_assignment(
            model, train_loader, test_loader, this_log_dir, n_classes, seed, n_features, n_per_class)

        # Store candidates
        feature_candidates.append(feature_sel)
        sparse_candidates.append(sparse_layer)
        bias_candidates.append(bias_sparse)
        mean_candidates.append(current_mean)
        std_candidates.append(current_std)

        # Set model with current candidates
        model.set_model_sldd(feature_sel, sparse_layer, current_mean, current_std, bias_sparse)

        if os.path.exists(this_log_dir / "trained_model.pth"):
            model.load_state_dict(torch.load(this_log_dir / "trained_model.pth"))
            _ = optimization_schedule.get_params()  # count up, to have get correct lr
            continue

        model = train_n_epochs(model, beta, optimization_schedule, train_loader, test_loader)
        torch.save(model.state_dict(), this_log_dir / "trained_model.pth")
        print(f"Finished iteration epoch {iteration_epoch}")

    # Apply meta-RL to optimize feature selection and tune non-zero weights
    if meta_rl:
        print("Starting Meta-RL training for feature selection and non-zero weights...")
        meta_log_dir = log_dir / "meta_rl"
        meta_log_dir.mkdir(parents=True, exist_ok=True)

        # Select the best feature set
        print("Optimizing feature selection...")
        best_feature_sel, feature_selection_rewards = metasenn_feature_selection(model=model, feature_candidates=feature_candidates, train_loader=train_loader, test_loader=test_loader, log_dir=meta_log_dir, n_features=n_features,
            device=device, rl_iterations=meta_iterations)

        # Get corresponding sparse layer for the best feature selection
        # Handle empty reward lists
        mean_rewards = []
        for r in feature_selection_rewards:
            if len(r) > 0:
                mean_rewards.append(np.mean(r))
            else:
                mean_rewards.append(0.0)  # Assign zero reward for empty lists

        # If all rewards are empty or zero, use the first candidate
        if all(m == 0 for m in mean_rewards) or not mean_rewards:
            best_idx = 0
            print("Warning: No valid rewards found. Using first feature candidate.")
        else:
            best_idx = np.argmax(mean_rewards)

        print(f"Selected feature set {best_idx} with mean reward {mean_rewards[best_idx] if best_idx < len(mean_rewards) else 0}")

        # Get best candidate components
        best_feature_sel = feature_candidates[best_idx]
        best_sparse_layer = sparse_candidates[best_idx].to(device)
        best_bias_sparse = bias_candidates[best_idx].to(device) if bias_candidates[best_idx] is not None else None
        best_current_mean = mean_candidates[best_idx].to(device) if mean_candidates[best_idx] is not None else None
        best_current_std = std_candidates[best_idx].to(device) if std_candidates[best_idx] is not None else None

        if isinstance(best_feature_sel, torch.Tensor):
            best_feature_sel = best_feature_sel.to(device)
        elif isinstance(best_feature_sel, (list, np.ndarray)):
            best_feature_sel = torch.tensor(best_feature_sel, device=device)

        # Set model with best feature selection
        model.set_model_sldd(best_feature_sel, best_sparse_layer, best_current_mean, best_current_std, best_bias_sparse)

        print("Optimizing sparse decision layer with meta-RL...")
        weights_log_dir = meta_log_dir / "weights_tuning"
        weights_log_dir.mkdir(parents=True, exist_ok=True)

        feature_dim = len(best_feature_sel) if hasattr(best_feature_sel, '__len__') else n_features
        model.feature_sel = best_feature_sel

        # Apply meta-RL for optimizing the sparse decision layer
        model = meta_rl_training(
            final_model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            log_dir=weights_log_dir,
            n_features=feature_dim,
            device=device,
            rl_iterations=meta_iterations*20,
            lr=0.001,
        )

        # Save the meta-RL tuned model
        torch.save(model.state_dict(), meta_log_dir / f"meta_rl_tuned_model_{n_features}.pth")
        torch.save({
            'feature_sel': best_feature_sel,
            'sparse_layer': model.linear.weight.data,
            'bias_sparse': model.linear.bias.data if model.linear.bias is not None else None,
            'current_mean': best_current_mean,
            'current_std': best_current_std,
            'feature_rewards': mean_rewards
        }, meta_log_dir / f"meta_rl_feature_selection_data_{n_features}.pth")

        print("Meta-RL training completed")

    return model