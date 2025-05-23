from finetuning.qsenn import finetune_qsenn
from finetuning.sldd import finetune_sldd


def finetune(key, model, train_loader, test_loader, log_dir, n_classes, seed, beta, optimization_schedule, per_class, n_features, meta_rl=False, meta_iterations=100, device='cuda'):
    if key == 'sldd':
        return finetune_sldd(model, train_loader, test_loader, log_dir, n_classes, seed, beta, optimization_schedule, per_class, n_features)
    elif key == 'qsenn':
        return finetune_qsenn(model, train_loader, test_loader, log_dir, n_classes, seed, beta, optimization_schedule, n_features, per_class, meta_rl, meta_iterations, device)
    else:
        raise ValueError(f"Unknown Finetuning key: {key}")