import sys
from pypdf import PdfReader

try:
    reader = PdfReader("specs/s40854-024-00702-7.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    print("--- SEARCHING FOR PAIR SELECTION CRITERIA ---")
    keywords = ["select", "formation", "criteria", "cointegration test", "step"]
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        if any(k in line.lower() for k in keywords) and ("EG" in line or "KSS" in line or "both" in line or "either" in line):
            start = max(0, i-5)
            end = min(len(lines), i+5)
            print(f"\nContext (Line {i}):")
            print('\n'.join(lines[start:end]))

    print("\n--- SEARCHING FOR 'BOTH' OR 'COMBINED' ---")
    for i, line in enumerate(lines):
        if "both" in line.lower() and "test" in line.lower():
             start = max(0, i-5)
             end = min(len(lines), i+5)
             print(f"\nContext (Line {i}):")
             print('\n'.join(lines[start:end]))

except Exception as e:
    print(f"Error: {e}")
