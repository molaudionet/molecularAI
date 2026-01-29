# MolecularAI - Molecular Sonification Framework

Multi-modal molecular AI using audio-based molecular representation for drug discovery.

## Quick Start

```bash
# Install
git clone https://github.com/molaudionet/molecularAI.git
cd molecularAI
pip install -r requirements.txt

# Run
python run.py --dataset tox21 --model fusion
```

## Key Results

- **Tox21:** AUC 0.751 (7,831 compounds)
- **BBBP:** AUC 0.905 (blood-brain barrier)
- **Training Speed:** 10×-100× faster via transfer learning

## Repository Structure

```
├── configs/       # Experiment configurations
├── data/         # Datasets (Tox21, BBBP, ESOL)
├── datasets/     # Data loaders
├── featurizers/  # Audio, descriptor, structure features
├── models/       # Neural networks (Wav2Vec 2.0, fusion)
├── training/     # Training utilities
└── run.py        # Main script
```

## Citation

```bibtex
@article{zhou2025molecular,
  title={Molecular Sonification: A Multi-Modal Approach for Enhanced AI in Drug Discovery},
  author={Zhou, Charles Jianping and Zhou, Emily R.},
  journal={Medicinal Chemistry Research},
  year={2025}
}
```

## License

MIT License (code) | US Patent 9,018,506 (sonification technology)

## Contact

- Jianping Zhou: zhou@uchicago.edu
- Emily Zhou: erzhou2@illinois.edu
- Web: soundofmolecules.com
