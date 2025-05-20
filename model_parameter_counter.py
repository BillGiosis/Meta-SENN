import torch
from argparse import ArgumentParser
from pathlib import Path
from architectures.model_mapping import get_model
from configs.dataset_params import dataset_constants

def detect_dataset_from_path(model_path):
    """Extract dataset name from model path"""
    path = str(Path(model_path))
    for dataset in dataset_constants.keys():
        if dataset in path:
            return dataset
    raise ValueError(f"Could not detect dataset from path: {model_path}")

def detect_model_type(model_path, state_dict):
    """Detect model type from path and state dict structure"""
    path = Path(model_path).name.lower()
    
    # Check state dict structure first
    if any("selection" in key for key in state_dict.keys()):
        if any("mean" in key for key in state_dict.keys()):
            return "qsenn"
        return "sldd"
    
    # Fallback to filename detection
    if "qsenn" in path:
        return "qsenn"
    elif "sldd" in path:
        return "sldd"
    return "dense"

def count_model_parameters(model_path, arch, reduced_strides=False):
    """Load model and count parameters"""
    try:
        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Detect dataset and get number of classes
        dataset = detect_dataset_from_path(model_path)
        n_classes = dataset_constants[dataset]["num_classes"]
        
        # Load state dict first
        state_dict = torch.load(model_path, map_location=device)
        
        # Initialize base model
        model = get_model(arch, n_classes, reduced_strides).to(device)
        
        # Try to load state dict
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"Warning: Could not load state dict exactly: {e}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        state_dict_params = sum(v.numel() for v in state_dict.values())
        
        return {
            'dataset': dataset,
            'n_classes': n_classes,
            'model_type': detect_model_type(model_path, state_dict),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'state_dict_parameters': state_dict_params,
            'state_dict_keys': list(state_dict.keys())
        }
    except Exception as e:
        print(f"Error analyzing model: {e}")
        return None

def print_model_info(params, arch):
    """Print formatted model information"""
    if not params:
        return
        
    print("\nParameter Counts:")
    print(f"Total parameters: {params['total_parameters']:,}")
    print(f"Trainable parameters: {params['trainable_parameters']:,}")
    
if __name__ == '__main__':
    parser = ArgumentParser(description='Analyze neural network model parameters')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to model state dict')
    parser.add_argument('--arch', type=str, default='resnet50',
                      choices=['resnet18', 'resnet50'],
                      help='Model architecture')
    parser.add_argument('--reduced_strides', action='store_true',
                      help='Use reduced strides')
    
    args = parser.parse_args()
    
    params = count_model_parameters(args.model_path, args.arch, args.reduced_strides)
    print_model_info(params, args.arch)