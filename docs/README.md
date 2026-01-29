# MolecularAI: Multi-Modal Molecular Intelligence Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

**Molecular Sonification: A Multi-Modal Approach for Enhanced AI in Drug Discovery**

This repository contains the official implementation of our molecular AI framework that integrates **molecular sonification technology** with transfer learning from voice AI models to achieve mechanistic understanding in drug discovery.

## 📄 Paper

**Authors:** Charles Jianping Zhou (University of Chicago) & Emily R. Zhou (Sound of Molecules LLC / UIUC)

**Published in:** *Medicinal Chemistry Research* (2025)

**Key Innovation:** Achieving 10×-100× faster training through audio-based molecular representation and transfer learning from pre-trained voice AI models (Wav2Vec 2.0, Whisper).

## 🎯 Key Features

- **Molecular Sonification:** Patent-protected technology (USP 9,018,506) that maps molecular properties to audio signals
- **Multi-Modal Learning:** Integrates audio, structure, spectroscopy, and physicochemical descriptors
- **Transfer Learning:** Leverages pre-trained Wav2Vec 2.0 models for 10×-100× computational efficiency
- **Benchmark Results:**
  - Tox21: AUC 0.751 (7,831 compounds)
  - BBBP: AUC 0.905 (blood-brain barrier penetration)
  - ESOL: Aqueous solubility prediction
- **Stereochemistry Encoding:** 89% R/S classification accuracy without experimental measurements

## 📁 Repository Structure

```
molecularAI/
├── configs/              # Configuration files for different experiments
├── data/                 # Raw datasets (CSV format)
├── datasets/            # Dataset processing and loading utilities
├── featurizers/         # Molecular featurization modules
│   ├── audio_featurizer.py      # Molecular sonification
│   ├── descriptor_featurizer.py  # Physicochemical descriptors
│   └── structure_featurizer.py   # Graph-based features
├── models/              # Neural network architectures
│   ├── wav2vec_model.py         # Wav2Vec 2.0 integration
│   ├── fusion_model.py          # Multi-modal fusion
│   └── baseline_models.py       # Descriptor/Graph baselines
├── training/            # Training scripts and utilities
├── utils/               # Helper functions and utilities
├── run.py               # Main training script
├── runall.sh            # Batch execution script
├── r.sh                 # Quick run script
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Required packages
pip install -r requirements.txt
```

### Installation

```bash
# Clone the repository
git clone https://github.com/molaudionet/molecularAI.git
cd molecularAI

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers
pip install rdkit
pip install pandas numpy scikit-learn
pip install librosa soundfile
```

### Basic Usage

```bash
# Run single experiment (Tox21 with multi-modal fusion)
python run.py --dataset tox21 --model fusion --mode audio_desc

# Run all benchmark experiments
bash runall.sh

# Quick test run
bash r.sh
```

## 📊 Datasets

### Included Datasets

All datasets are located in the `data/` directory in CSV format:

| Dataset | Task | Size | Metric | Our Result |
|---------|------|------|--------|------------|
| **Tox21** | Toxicity Classification | 7,831 | AUC | 0.751 |
| **BBBP** | Blood-Brain Barrier | 2,039 | AUC | 0.905 |
| **ESOL** | Aqueous Solubility | 1,128 | R² | - |

### Dataset Format

```csv
smiles,label
CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O,1
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,0
...
```

## 🎵 Molecular Sonification

The core innovation of this framework is **molecular sonification** - converting molecular structures to audio signals:

```python
from featurizers.audio_featurizer import MolecularAudioFeaturizer

# Initialize featurizer
audio_feat = MolecularAudioFeaturizer()

# Convert SMILES to audio features
smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"  # Ibuprofen
audio_features = audio_feat.featurize(smiles)

# Shape: (768,) - Wav2Vec 2.0 embeddings
print(audio_features.shape)
```

### How It Works

1. **Molecular Properties → Audio**: Maps atomic mass, electronegativity, bond polarity to 20Hz-20kHz range
2. **Dense Audio Stream**: Creates continuous signal preserving molecular vibrations
3. **Wav2Vec 2.0 Processing**: Extracts 768-dimensional embeddings
4. **Multi-Modal Fusion**: Combines with structural and physicochemical features

## 🧪 Experiments

### Run Individual Experiments

```bash
# Tox21 - Audio only
python run.py --dataset tox21 --model audio --mode audio_only

# BBBP - Multi-modal fusion
python run.py --dataset bbbp --model fusion --mode audio_desc

# ESOL - Descriptor baseline
python run.py --dataset esol --model baseline --mode desc_only
```

### Configuration

Edit `configs/default_config.yaml`:

```yaml
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  
model:
  audio_dim: 768        # Wav2Vec 2.0 output
  desc_dim: 7           # Physicochemical descriptors
  hidden_dim: 256
  dropout: 0.3
```

## 📈 Results

### Performance Comparison

| Dataset | Desc Only | Audio Only | **Fusion (Ours)** |
|---------|-----------|------------|-------------------|
| Tox21 (AUC) | 0.722 | 0.736 | **0.751** |
| BBBP (AUC) | 0.845 | 0.843 | **0.905** |

### Training Time

| Method | Training Time | Speedup |
|--------|--------------|---------|
| GNN from scratch | 120-180 hours | 1× |
| Descriptor baseline | 2-3 hours | 40-60× |
| **Audio + Transfer Learning** | **12-18 hours** | **10-15×** |

## 🔬 Advanced Usage

### Custom Datasets

```python
from datasets.molecule_dataset import MoleculeDataset

# Load your data
dataset = MoleculeDataset(
    csv_path='your_data.csv',
    smiles_col='smiles',
    label_col='activity'
)

# Featurize
from featurizers import MultiModalFeaturizer
featurizer = MultiModalFeaturizer(use_audio=True, use_desc=True)
features = featurizer.transform(dataset)
```

### Model Customization

```python
from models.fusion_model import MultiModalFusionModel

model = MultiModalFusionModel(
    audio_dim=768,
    desc_dim=7,
    hidden_dims=[512, 256, 128],
    dropout=0.3,
    fusion_method='attention'  # or 'concat', 'gated'
)
```

## 📖 Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{zhou2025molecular,
  title={Molecular Sonification: A Multi-Modal Approach for Enhanced AI in Drug Discovery},
  author={Zhou, Charles Jianping and Zhou, Emily R.},
  journal={Medicinal Chemistry Research},
  year={2025},
  publisher={Springer}
}
```

### Related Patents

```bibtex
@patent{zhou2014sonification,
  title={System and method for creating audible sound representations of atoms and molecules},
  author={Zhou, Emily R. and Zhou, Charles J.},
  number={US 9,018,506},
  year={2014},
  nationality={United States}
}
```

## 🙏 Acknowledgments

This work is dedicated to **Professor Richard B. Silverman** on his 80th birthday and 50 years at Northwestern University. We thank Professor Silverman for his mentorship and pioneering work in mechanism-based drug design.

We honor the memory of **Professor Philip E. Eaton** (1936-2023), whose cubyl chemistry inspired our understanding of molecular mechanisms.

Emily R. Zhou is grateful for the University of Illinois iVenture Accelerator grant supporting Sound of Molecules LLC.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note:** Molecular sonification technology is protected by US Patent 9,018,506. Commercial use requires licensing from Sound of Molecules LLC.

## 🔗 Links

- **Paper:** [Medicinal Chemistry Research](https://link.springer.com/) (DOI: TBD)
- **Sound of Molecules:** [Website](https://soundofmolecules.com)
- **Data Repository:** [Zenodo](https://doi.org/10.5281/zenodo.8425713)

## 📧 Contact

- **Charles Jianping Zhou** - University of Chicago - zhou@uchicago.edu
- **Emily R. Zhou** - Sound of Molecules LLC / UIUC - erzhou2@illinois.edu

## 🐛 Issues & Contributions

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

For bugs or feature requests, please open an issue on GitHub.

## 📝 Updates

- **January 2025:** Initial release with Tox21, BBBP, ESOL benchmarks
- **Future:** Additional datasets (HIV, MUV, SIDER), stereochemistry module, real-time sonification tool

---

**Made with 🎵 by Sound of Molecules**

*Transforming molecules into music, music into understanding.*
