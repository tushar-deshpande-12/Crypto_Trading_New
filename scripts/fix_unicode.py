"""
Fix unicode encoding issues in source files
Replace unicode symbols with ASCII alternatives for Windows console compatibility
"""

import os
from pathlib import Path

# Mapping of unicode characters to ASCII replacements
REPLACEMENTS = {
    '✓': '[OK]',
    '✗': '[X]',
    '⚠': '[!]',
    '♥': '[*]',
    '●': '*',
    '○': 'o',
    '→': '->',
    '↻': '[R]',
}

def fix_file(filepath):
    """Fix unicode issues in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original = content

        # Apply replacements
        for unicode_char, ascii_replacement in REPLACEMENTS.items():
            content = content.replace(unicode_char, ascii_replacement)

        # Only write if changes were made
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True

        return False

    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Fix all Python files in src directory"""
    src_dir = Path(__file__).parent.parent / 'src'

    print("Fixing unicode encoding issues...")
    print(f"Scanning: {src_dir}")
    print("-" * 60)

    fixed_count = 0

    # Find all Python files
    for py_file in src_dir.rglob('*.py'):
        if fix_file(py_file):
            fixed_count += 1

    print("-" * 60)
    print(f"Fixed {fixed_count} files")

if __name__ == "__main__":
    main()
