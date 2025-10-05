#!/usr/bin/env python3
"""
Read case study documents and extract requirements.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from docx import Document
except ImportError:
    print("Installing python-docx...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx"])
    from docx import Document

def read_docx(filepath):
    """Read a .docx file and return the text content."""
    try:
        doc = Document(filepath)
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text.strip())
        return '\n'.join(text)
    except Exception as e:
        return f'Error reading {filepath}: {e}'

def main():
    # Read main case study
    print('=== MAIN CASE STUDY ===')
    main_text = read_docx('data/CASESTUDY_FALL25.docx')
    print(main_text)
    
    print('\n' + '='*50 + '\n')
    
    # Read Fund 1 case study
    print('=== FUND 1 CASE STUDY ===')
    fund1_text = read_docx('data/CASESTUDY_FALL25_FUND1.docx')
    print(fund1_text)

if __name__ == "__main__":
    main()
