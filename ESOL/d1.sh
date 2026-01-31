# Generate complete project map (safe to share)
find . -type f \( -name "*.py" -o -name "*.json" -o -name "*.yaml" -o -name "*.md" \) \
  -not -path "*/.git/*" -not -path "*/__pycache__/*" -not -path "*/.venv/*" \
  | head -100 > project_structure.txt

# Share key files I should review first:
cat project_structure.txt

# Core architecture
cat run.py
cat config/*.yaml 2>/dev/null || cat config/*.json 2>/dev/null

# Data pipeline
find . -name "*dataset*.py" -o -name "*loader*.py" | head -5 | xargs -I {} cat {}

# Sonification (if exists)
find . -name "*sonic*.py" -o -name "*audio*.py" -o -name "*wave*.py" | head -5 | xargs -I {} cat {}


find . -type f \( -name "*.py" -o -name "*.json" -o -name "*.yaml" \) \
  -not -path "*/.git/*" -not -path "*/__pycache__/*" | head -50
