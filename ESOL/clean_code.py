import re
import sys
from pathlib import Path

def clean_code_content(text):
    """
    Cleans code by removing emojis and converting AI-generated 
    special characters (smart quotes, etc.) into code-safe ASCII.
    """
    # 1. Map 'Smart' characters to standard ASCII
    replacements = {
        '\u201c': '"', '\u201d': '"',  # Smart double quotes
        '\u2018': "'", '\u2019': "'",  # Smart single quotes
        '\u2013': '-', '\u2014': '--', # En/Em dashes
        '\u00a0': ' ',                 # Non-breaking space
        '\u2026': '...',               # Ellipsis
    }
    
    for char, rep in replacements.items():
        text = text.replace(char, rep)

    # 2. Regex to strip Emojis and miscellaneous symbols
    # This covers the most common emoji and symbol blocks
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags
        "\U00002702-\U000027b0"  # dingbats
        "\U000024c2-\U0001f251"  # circled letters/extra symbols
        "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub('', text)

    # 3. Optional: Remove non-ASCII non-printable characters 
    # (keeps standard characters, numbers, and common punctuation)
    # text = "".join(i for i in text if ord(i) < 128) 

    return text

def process_file(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: {file_path} not found.")
        return

    print(f"Processing {path.name}...")
    
    # Read with utf-8, ignoring errors to prevent script crash
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    cleaned = clean_code_content(content)

    # Create backup
    backup = path.with_suffix(path.suffix + '.original')
    path.replace(backup)

    # Write cleaned file
    with open(path, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    
    print(f"Done! Cleaned code saved to {path.name}. Original backed up to {backup.name}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_code_final.py <filename.py/html/php>")
    else:
        process_file(sys.argv[1])

