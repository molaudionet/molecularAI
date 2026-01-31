#!/bin/bash
# Download and prepare MoleculeNet benchmarks for sound-of-molecules

mkdir -p data && cd data

# BBBP (Blood-Brain Barrier Penetration)
echo " Downloading BBBP..."
wget -nc https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv
# Rename target column to match config expectation
if ! grep -q "p_np" BBBP.csv; then
  echo "  BBBP.csv format may need adjustment - check column names"
fi

# Tox21 (12 toxicity endpoints)
echo " Downloading Tox21..."
wget -nc https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz
gunzip -f tox21.csv.gz
# Tox21 has columns: mol_id,smiles,SR-HSE,...,NR-AR, etc.

# ESOL (you likely have this already)
echo " Downloading ESOL (if needed)..."
wget -nc https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv
mv delaney-processed.csv esol.csv

# FreeSolv (hydration energy)
echo " Downloading FreeSolv..."
wget -nc https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv
mv SAMPL.csv freesolv.csv

# HIV (optional - large dataset)
# wget -nc https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv

echo -e "\n Datasets ready in ./data/"
ls -lh *.csv
