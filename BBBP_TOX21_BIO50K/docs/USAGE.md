# Usage Guide - MolecularAI

Detailed examples for using the MolecularAI framework.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Custom Datasets](#custom-datasets)
3. [Featurization](#featurization)
4. [Training Models](#training-models)
5. [Evaluation](#evaluation)
6. [Advanced Examples](#advanced-examples)

## Basic Usage

### Run Benchmark Experiments

```bash
# Tox21 toxicity prediction with multi-modal fusion
python run.py --dataset tox21 --model fusion --mode audio_desc

# BBBP permeability with audio only
python run.py --dataset bbbp --model audio --mode audio_only

# ESOL solubility with descriptors
python run.py --dataset esol --model baseline --mode desc_only
```

### Configuration Options

```bash
python run.py \
  --dataset tox21 \
  --model fusion \
  --mode audio_desc \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --gpu 0
```

## Custom Datasets

### Load Your Own Data

```python
from datasets.molecule_dataset import MoleculeDataset
import pandas as pd

# Prepare your data
df = pd.DataFrame({
    'smiles': ['CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O', ...],
    'activity': [1, 0, 1, ...],
    'name': ['Ibuprofen', ...]
})

# Save as CSV
df.to_csv('data/my_dataset.csv', index=False)

# Load dataset
dataset = MoleculeDataset(
    csv_path='data/my_dataset.csv',
    smiles_col='smiles',
    label_col='activity',
    task='classification'  # or 'regression'
)

print(f"Dataset size: {len(dataset)}")
print(f"First molecule: {dataset[0]}")
```

### Split Data

```python
from sklearn.model_selection import train_test_split

# Get SMILES and labels
smiles_list = dataset.get_smiles()
labels = dataset.get_labels()

# Split
train_smiles, test_smiles, train_labels, test_labels = train_test_split(
    smiles_list, labels, test_size=0.2, random_state=42
)

# Create train/test datasets
train_dataset = MoleculeDataset.from_lists(train_smiles, train_labels)
test_dataset = MoleculeDataset.from_lists(test_smiles, test_labels)
```

## Featurization

### Audio Features (Molecular Sonification)

```python
from featurizers.audio_featurizer import MolecularAudioFeaturizer

# Initialize
audio_feat = MolecularAudioFeaturizer(
    model_name='facebook/wav2vec2-base',
    sample_rate=16000,
    duration=2.0  # seconds
)

# Single molecule
smiles = 'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O'
audio_vector = audio_feat.featurize(smiles)
print(f"Audio features shape: {audio_vector.shape}")  # (768,)

# Batch processing
smiles_list = ['CCO', 'CC(C)O', 'CCCC']
audio_features = audio_feat.featurize_batch(smiles_list)
print(f"Batch shape: {audio_features.shape}")  # (3, 768)
```

### Descriptor Features

```python
from featurizers.descriptor_featurizer import DescriptorFeaturizer

desc_feat = DescriptorFeaturizer(
    descriptors=['MolWt', 'LogP', 'TPSA', 'NumHDonors', 
                 'NumHAcceptors', 'NumRotatableBonds', 'NumAromaticRings']
)

# Featurize
smiles = 'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O'
descriptors = desc_feat.featurize(smiles)
print(f"Descriptors: {descriptors}")  # Shape: (7,)
```

### Structure Features (Graph)

```python
from featurizers.structure_featurizer import GraphFeaturizer

graph_feat = GraphFeaturizer(
    atom_features=['atomic_num', 'degree', 'hybridization'],
    bond_features=['bond_type', 'conjugated', 'in_ring']
)

# Get graph representation
smiles = 'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O'
node_features, edge_index, edge_features = graph_feat.featurize(smiles)

print(f"Nodes: {node_features.shape}")
print(f"Edges: {edge_index.shape}")
```

### Multi-Modal Features

```python
from featurizers import MultiModalFeaturizer

# Combine audio + descriptors
mm_feat = MultiModalFeaturizer(
    use_audio=True,
    use_descriptors=True,
    use_graph=False
)

# Featurize
smiles = 'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O'
features = mm_feat.featurize(smiles)

print(f"Audio features: {features['audio'].shape}")      # (768,)
print(f"Descriptor features: {features['desc'].shape}")  # (7,)
```

## Training Models

### Audio-Only Model

```python
from models.audio_model import AudioOnlyModel
from training.trainer import Trainer

# Initialize model
model = AudioOnlyModel(
    input_dim=768,
    hidden_dims=[256, 128],
    output_dim=1,
    dropout=0.3
)

# Train
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=100,
    batch_size=32,
    learning_rate=0.001
)

trainer.train()
```

### Multi-Modal Fusion Model

```python
from models.fusion_model import MultiModalFusionModel

# Initialize
model = MultiModalFusionModel(
    audio_dim=768,
    desc_dim=7,
    hidden_dim=256,
    output_dim=1,
    fusion_method='attention',  # or 'concat', 'gated'
    dropout=0.3
)

# Configure fusion
model.configure_fusion(
    attention_heads=4,
    gate_activation='sigmoid'
)

# Train
trainer = Trainer(model=model, ...)
trainer.train()
```

## Evaluation

### Predictions

```python
# Make predictions
predictions = trainer.predict(test_dataset)

# Get probabilities for classification
probs = trainer.predict_proba(test_dataset)

# Single molecule prediction
smiles = 'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O'
pred = model.predict_smiles(smiles)
print(f"Prediction: {pred}")
```

### Metrics

```python
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

# Classification metrics
y_true = test_dataset.get_labels()
y_pred = predictions
y_probs = probs

auc = roc_auc_score(y_true, y_probs)
acc = accuracy_score(y_true, (y_probs > 0.5).astype(int))

print(f"AUC: {auc:.3f}")
print(f"Accuracy: {acc:.3f}")

# Regression metrics
mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.3f}")
```

## Advanced Examples

### Custom Training Loop

```python
import torch
from torch.utils.data import DataLoader

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(100):
    model.train()
    for batch in train_loader:
        # Forward pass
        audio_features = batch['audio']
        desc_features = batch['descriptors']
        labels = batch['labels']
        
        outputs = model(audio_features, desc_features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch['audio'], batch['descriptors'])
            val_loss += criterion(outputs, batch['labels']).item()
    
    print(f"Epoch {epoch}: Val Loss = {val_loss/len(val_loader):.4f}")
```

### Attention Visualization

```python
# Get attention weights from fusion layer
model.eval()
with torch.no_grad():
    audio_feat = audio_featurizer.featurize(smiles)
    desc_feat = desc_featurizer.featurize(smiles)
    
    # Forward pass with attention
    output, attention_weights = model.forward_with_attention(
        audio_feat, desc_feat
    )

# Visualize attention
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.bar(['Audio'] + descriptor_names, attention_weights)
plt.title('Multi-Modal Attention Weights')
plt.ylabel('Attention')
plt.show()
```

### Transfer Learning Fine-tuning

```python
from models.audio_model import AudioOnlyModel

# Load pre-trained Wav2Vec 2.0
model = AudioOnlyModel(
    input_dim=768,
    hidden_dims=[256, 128],
    output_dim=1,
    pretrained_model='facebook/wav2vec2-base',
    freeze_encoder=True  # Freeze Wav2Vec weights initially
)

# Fine-tune in stages
# Stage 1: Train only classifier (encoder frozen)
trainer.train(epochs=20)

# Stage 2: Unfreeze and fine-tune entire model
model.freeze_encoder = False
trainer.train(epochs=80, learning_rate=0.0001)
```

### Hyperparameter Search

```python
from training.hyperparameter_search import GridSearch

# Define search space
param_grid = {
    'hidden_dim': [128, 256, 512],
    'dropout': [0.1, 0.3, 0.5],
    'learning_rate': [0.0001, 0.001, 0.01],
    'fusion_method': ['attention', 'concat', 'gated']
}

# Run grid search
search = GridSearch(
    model_class=MultiModalFusionModel,
    param_grid=param_grid,
    dataset=train_dataset,
    cv_folds=5
)

best_params, best_score = search.run()
print(f"Best parameters: {best_params}")
print(f"Best AUC: {best_score:.3f}")
```

### Export Model

```python
# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}, 'checkpoints/model_best.pth')

# Load model
checkpoint = torch.load('checkpoints/model_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Export to ONNX for deployment
dummy_audio = torch.randn(1, 768)
dummy_desc = torch.randn(1, 7)

torch.onnx.export(
    model,
    (dummy_audio, dummy_desc),
    'model_export.onnx',
    export_params=True,
    input_names=['audio', 'descriptors'],
    output_names=['output']
)
```

## Troubleshooting

### Memory Issues

```python
# Reduce batch size
trainer = Trainer(..., batch_size=16)

# Use gradient accumulation
trainer = Trainer(..., gradient_accumulation_steps=4)

# Use mixed precision training
trainer = Trainer(..., use_amp=True)
```

### Slow Training

```python
# Use multiple workers for data loading
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True
)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

## Need Help?

- Open an issue on GitHub
- Email: zhou@uchicago.edu or erzhou2@illinois.edu
- Documentation: https://github.com/molaudionet/molecularAI/wiki
