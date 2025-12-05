import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json
import re
import lightning as L
import ast

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
        # Convert node features
        for node_type in batch.node_types:
            if hasattr(batch[node_type], 'x') and batch[node_type].x is not None:
                batch[node_type].x = batch[node_type].x.float()
            if hasattr(batch[node_type], 'y') and batch[node_type].y is not None:
                batch[node_type].y = batch[node_type].y.float()
        
        # Convert edge attributes
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

                    # Compute per-dimension errors
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
    key = re.sub(pattern, replacer, key)
    
    return key

token = "your_HF_token_here"  # Replace with your HuggingFace token
# Download model files
print("Downloading model files from HuggingFace...")
config_path = hf_hub_download(repo_id="argonne/LUMINA-1B", filename="config.json", token=token)
safetensors_path = hf_hub_download(repo_id="argonne/LUMINA-1B", filename="model.safetensors", token=token)

model_path = hf_hub_download(repo_id="argonne/LUMINA-1B", filename="model.pt", token=token)
checkpoint = torch.load(model_path, map_location='cpu')

# Load config
with open(config_path, 'r') as f:
    config_data = json.load(f)

# # print the content of config.json
# for key in config_data:
#     print(f"{key}: {config_data[key]}")

# FIX 1: Convert metadata edge keys from strings to tuples
print("Converting metadata edge keys to tuples...")
if 'edges' in config_data['metadata']:
    edges_dict = {}
    for key, value in config_data['metadata']['edges'].items():
        if isinstance(key, str) and key.startswith('('):
            key = ast.literal_eval(key)
        edges_dict[key] = value
    config_data['metadata']['edges'] = edges_dict

print("Creating model...")
from lumina.model.opf.hetero_model import OPFHeteroGNN

model = OPFHeteroGNN(
    metadata=config_data['metadata'],
    input_channels=config_data['input_channels'],
    hidden_channels=config_data['config']['models']['HeteroGNN']['hidden_channels'],
    num_layers=config_data['config']['models']['HeteroGNN']['num_layers'],
    backend=config_data['config']['models']['HeteroGNN']['backend'],
)

# print("Loading model weights...")
# state_dict = load_file(safetensors_path)
# new_state_dict = {convert_key(k): v for k, v in state_dict.items()}
# model.load_state_dict(new_state_dict)

print("Loading model weights...")
state_dict = load_file(safetensors_path)
checkpoint_dict = {}
for k, v in state_dict.items():
    new_k = convert_checkpoint_key_to_model_key(k)
    checkpoint_dict[new_k] = v

# Now match with model parameters
for param_name, param in model.named_parameters():
    # Convert model param name to checkpoint format
    checkpoint_key = convert_checkpoint_key_to_model_key(param_name)
    if checkpoint_key in checkpoint_dict:
        param.data.copy_(checkpoint_dict[checkpoint_key])

# Also copy buffers
for buffer_name, buffer in model.named_buffers():
    checkpoint_key = convert_checkpoint_key_to_model_key(buffer_name)
    if checkpoint_key in checkpoint_dict:
        buffer.data.copy_(checkpoint_dict[checkpoint_key])

print("Model weights loaded successfully!")

# model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# data 
print("Loading dataset...")
from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.loader.opf.opf_loader import DataLoader

dataset = OPFDataset(root='./opf_data', case_name='pglib_opf_case14_ieee')

test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# loss function
from lumina.model.opf.losses import ACOPFLossFunction
loss_manager = ACOPFLossFunction(loss_type="mse")


lightning_module = ACOPFEvaluationModule(
    model=model,
    loss_manager=loss_manager,
    verbose=False,
)

# Create Lightning Trainer for evaluation
trainer = L.Trainer(
    accelerator="cuda",
    devices=1,
    precision="32-true",
    logger=False,  # Disable logger for simple evaluation
    enable_checkpointing=False,  # No checkpointing needed
    enable_progress_bar=True,
    enable_model_summary=False
)

print(f"\n{'='*60}")
print("Running Evaluation")
print(f"{'='*60}")
print(f"Accelerator: cuda")
print(f"Devices: 1")
print(f"Precision: 32-true")

# Run evaluation
results = trainer.test(lightning_module, dataloaders=test_loader)

# Print results summary
print(f"\n{'='*60}")
print("Evaluation Results Summary")
print(f"{'='*60}")
if results:
    for metric_name, metric_value in results[0].items():
        print(f"{metric_name}: {metric_value:.6f}")

print(f"\n{'='*60}")
print("Evaluation Complete")
print(f"{'='*60}")
