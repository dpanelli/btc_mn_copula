import sys
from pypdf import PdfReader

try:
    reader = PdfReader("specs/s40854-024-00702-7.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    print("--- SEARCHING FOR KSS / KAPETANIOS ---")
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if "kapetanios" in line.lower() or "kss" in line.lower():
            start = max(0, i-5)
            end = min(len(lines), i+5)
            print(f"\nContext (Line {i}):")
            print('\n'.join(lines[start:end]))

    print("\n--- SEARCHING FOR EXIT / THRESHOLD ---")
    for i, line in enumerate(lines):
        if "exit" in line.lower() or "threshold" in line.lower() or "alpha" in line.lower() or "confidence band" in line.lower():
            start = max(0, i-5)
            end = min(len(lines), i+5)
            print(f"\nContext (Line {i}):")
            print('\n'.join(lines[start:end]))

except Exception as e:
    print(f"Error: {e}")
