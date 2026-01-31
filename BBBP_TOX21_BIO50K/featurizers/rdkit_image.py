from __future__ import annotations
from typing import Optional
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


def smiles_to_png(smiles: str, out_path: str, size: int = 224) -> Optional[str]:
    """
    Render a SMILES as a 2D depiction PNG using RDKit.
    Returns the output path if successful, else None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        # still attempt to draw; RDKit often handles it anyway
        pass

    out_path = str(Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()

    with open(out_path, "wb") as f:
        f.write(png_bytes)

    return out_path

