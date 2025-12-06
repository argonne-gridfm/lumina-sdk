import torch
import os
import argparse
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json
import re
import lightning as L
import ast
from torch.utils.data import Subset


class ACOPFEvaluationModule(L.LightningModule):
    """Lightning Module for ACOPF model evaluation"""

    def __init__(self, model, loss_manager, verbose=False):
        super().__init__()
        self.model = model
        self.loss_manager = loss_manager
        self.verbose = verbose
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        """Test step - run inference and compute losses"""
        # Convert batch to float32 to match model dtype
        batch = self._convert_batch_to_float32(batch)

        # Forward pass
        predictions = self.model(
            batch.x_dict,
            batch.edge_index_dict,
            batch.edge_attr_dict if hasattr(batch, 'edge_attr_dict') else None
        )

        # Compute losses
        loss_results = self.loss_manager.compute_loss(predictions, batch)

        batch_size = batch['bus'].batch.max().item() + 1 if hasattr(batch['bus'], 'batch') else 1

        # Log metrics
        self.log('test_loss', loss_results['total_loss'], prog_bar=True, batch_size=batch_size)
        for key, value in loss_results.items():
            if key != 'total_loss' and '_loss' in key:
                self.log(f'test_{key}', value, batch_size=batch_size)

        # Store predictions and targets for verbose output
        if self.verbose:
            output = {
                'predictions': {k: v.detach().cpu() for k, v in predictions.items()},
                'targets': {},
                'loss_results': {k: v.item() for k, v in loss_results.items()}
            }
            for node_type in predictions.keys():
                if hasattr(batch[node_type], 'y') and batch[node_type].y is not None:
                    output['targets'][node_type] = batch[node_type].y.detach().cpu()
            self.test_outputs.append(output)

        return loss_results
    
    def _convert_batch_to_float32(self, batch):
        """Convert all tensors in batch to float32"""
        for node_type in batch.node_types:
            if hasattr(batch[node_type], 'x') and batch[node_type].x is not None:
                batch[node_type].x = batch[node_type].x.float()
            if hasattr(batch[node_type], 'y') and batch[node_type].y is not None:
                batch[node_type].y = batch[node_type].y.float()
        
        for edge_type in batch.edge_types:
            if hasattr(batch[edge_type], 'edge_attr') and batch[edge_type].edge_attr is not None:
                batch[edge_type].edge_attr = batch[edge_type].edge_attr.float()
        
        return batch

    def on_test_epoch_end(self):
        """Called at the end of test epoch"""
        if self.verbose and self.test_outputs:
            self._print_detailed_results()

    def _print_detailed_results(self):
        """Print detailed predictions vs targets"""
        print(f"\n{'='*60}")
        print("Detailed Predictions vs Targets")
        print(f"{'='*60}")

        for idx, output in enumerate(self.test_outputs):
            predictions = output['predictions']
            targets = output['targets']

            for node_type in predictions.keys():
                if node_type in targets:
                    pred = predictions[node_type]
                    target = targets[node_type]

                    print(f"\n{node_type.upper()}:")
                    print(f"  Shape: {pred.shape}")
                    print(f"  Predictions (first 5 nodes):")
                    print(f"    {pred[:5]}")
                    print(f"  Targets (first 5 nodes):")
                    print(f"    {target[:5]}")

                    errors = torch.abs(pred - target)
                    mean_errors = errors.mean(dim=0)
                    max_errors = errors.max(dim=0)[0]

                    print(f"  Mean absolute errors per dimension: {mean_errors}")
                    print(f"  Max absolute errors per dimension: {max_errors}")


def convert_checkpoint_key_to_model_key(key):
    """Convert checkpoint key format to model's internal format"""
    pattern = r'<([^>]+)>'
    def replacer(match):
        content = match.group(1)
        parts = content.split('___')
        return f"('{parts[0]}', '{parts[1]}', '{parts[2]}')"
    return re.sub(pattern, replacer, key)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate ACOPF model',
    )
    
    # Model arguments
    parser.add_argument('--repo-id', type=str, default='argonne/LUMINA-1B',
                        help='HuggingFace repository ID')
    parser.add_argument('--hf-token', type=str, default=None,
                        help='HuggingFace token')
    
    # Dataset arguments
    parser.add_argument('--data-root', type=str, default='./opf_data',
                        help='Root directory for dataset')
    parser.add_argument('--case-name', type=str, default='pglib_opf_case14_ieee',
                        help='OPF case name')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Limit number of samples (None = use all)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--loss-type', type=str, default='mse',
                        choices=['mse', 'rmse', 'mae', 'mape', 'smooth_l1'],
                        help='Loss function type')
    
    # Lightning Trainer arguments
    parser.add_argument('--accelerator', type=str, default='cuda',
                        choices=['cpu', 'cuda', 'mps', 'tpu'],
                        help='Accelerator type')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of devices to use')
    parser.add_argument('--precision', type=str, default='32-true',
                        choices=['32-true', '16-mixed', 'bf16-mixed'],
                        help='Training precision')
    
    # Output arguments
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed predictions vs targets')
    parser.add_argument('--progress-bar', dest='progress_bar', action='store_true', default=True,
                        help='Show progress bar (default: on)')
    parser.add_argument('--no-progress-bar', dest='progress_bar', action='store_false',
                        help='Disable progress bar')
    
    return parser.parse_args()


def load_model_from_hub(repo_id, token):
    """Load model configuration and weights from HuggingFace Hub"""
    print("Downloading model files from HuggingFace...")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", token=token)
    safetensors_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", token=token)
    
    # Load config
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Convert metadata edge keys to tuples
    print("Converting metadata edge keys to tuples...")
    if 'edges' in config_data['metadata']:
        edges_dict = {}
        for key, value in config_data['metadata']['edges'].items():
            if isinstance(key, str) and key.startswith('('):
                key = ast.literal_eval(key)
            edges_dict[key] = value
        config_data['metadata']['edges'] = edges_dict
    
    return config_data, safetensors_path


def create_model(config_data):
    """Create model from configuration"""
    print("Creating model...")
    from lumina.model.opf.hetero_model import OPFHeteroGNN
    
    model = OPFHeteroGNN(
        metadata=config_data['metadata'],
        input_channels=config_data['input_channels'],
        hidden_channels=config_data['config']['models']['HeteroGNN']['hidden_channels'],
        num_layers=config_data['config']['models']['HeteroGNN']['num_layers'],
        backend=config_data['config']['models']['HeteroGNN']['backend'],
    )
    
    return model


def load_model_weights(model, safetensors_path):
    """Load model weights from safetensors file"""
    print("Loading model weights...")
    state_dict = load_file(safetensors_path)
    
    # Convert checkpoint keys
    checkpoint_dict = {}
    for k, v in state_dict.items():
        new_k = convert_checkpoint_key_to_model_key(k)
        checkpoint_dict[new_k] = v
    
    # Match with model parameters
    for param_name, param in model.named_parameters():
        checkpoint_key = convert_checkpoint_key_to_model_key(param_name)
        if checkpoint_key in checkpoint_dict:
            param.data.copy_(checkpoint_dict[checkpoint_key])
    
    # Copy buffers
    for buffer_name, buffer in model.named_buffers():
        checkpoint_key = convert_checkpoint_key_to_model_key(buffer_name)
        if checkpoint_key in checkpoint_dict:
            buffer.data.copy_(checkpoint_dict[checkpoint_key])
    
    print("Model weights loaded successfully!")
    return model


def load_dataset(data_root, case_name, num_samples=None):
    """Load OPF dataset"""
    print("Loading dataset...")
    from lumina.dataset.opf.opf_dataset import OPFDataset
    
    dataset = OPFDataset(root=data_root, case_name=case_name)
    
    # Limit dataset size if specified
    if num_samples is not None and num_samples < len(dataset):
        print(f"Limiting dataset to {num_samples} samples")
        dataset = Subset(dataset, range(num_samples))
    
    print(f"Dataset size: {len(dataset)}")
    
    return dataset


def main():
    # Parse arguments
    args = parse_args()
    
    # Get HuggingFace token
    token = args.hf_token
    if not token:
        raise ValueError(
            "HuggingFace token not provided"
        )
    
    # Load model from HuggingFace Hub
    config_data, safetensors_path = load_model_from_hub(args.repo_id, token)
    
    # Create and load model
    model = create_model(config_data)
    model = load_model_weights(model, safetensors_path)
    model.eval()
    
    # Load dataset
    dataset = load_dataset(args.data_root, args.case_name, args.num_samples)
    
    # Create dataloader
    from lumina.loader.opf.opf_loader import DataLoader
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create loss function
    from lumina.model.opf.losses import ACOPFLossFunction
    loss_manager = ACOPFLossFunction(loss_type=args.loss_type)
    
    # Create Lightning module
    lightning_module = ACOPFEvaluationModule(
        model=model,
        loss_manager=loss_manager,
        verbose=args.verbose,
    )
    
    # Create Lightning Trainer
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=args.progress_bar,
        enable_model_summary=False
    )
    
    # Print configuration
    print(f"\n{'='*60}")
    print("Evaluation Configuration")
    print(f"{'='*60}")
    print(f"Repository: {args.repo_id}")
    print(f"Case: {args.case_name}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Loss type: {args.loss_type}")
    print(f"Accelerator: {args.accelerator}")
    print(f"Devices: {args.devices}")
    print(f"Precision: {args.precision}")
    print(f"Verbose: {args.verbose}")
    
    # Run evaluation
    print(f"\n{'='*60}")
    print("Running Evaluation")
    print(f"{'='*60}")
    
    results = trainer.test(lightning_module, dataloaders=test_loader)
    
    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Results Summary")
    print(f"{'='*60}")
    if results:
        for metric_name, metric_value in results[0].items():
            print(f"{metric_name}: {metric_value:.6f}")
    
    print(f"\n{'='*60}")
    print("Evaluation Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
