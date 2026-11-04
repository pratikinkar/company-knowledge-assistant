
from pathlib import Path

# Find the document
file_path = Path("documents/company_info.txt")

# Check whether the file exists
if not file_path.exists():
    print("Document not found! Check the file path.")
else:
    # Read the document
    document_text = file_path.read_text(encoding="utf-8")

    print("Document loaded successfully!\n")
    print(document_text)
 