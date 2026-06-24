import os
import shutil
from pathlib import Path
from llama_index.core import SimpleDirectoryReader

class DealRoomParser:
    def __init__(self, sandbox_dir: str = "backend/app/engine/data_workspace"):
        self.sandbox_dir = Path(sandbox_dir)
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    def process_files(self, input_dir: str):
        """
        Scans an input directory. Routes CSVs to the sandbox 
        and extracts PDFs for text indexing.
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

        # 1. Route Tabular Files (CSV/XLSX) straight to the safe sandbox
        for tabular_file in list(input_path.glob("*.csv")) + list(input_path.glob("*.xlsx")):
            destination = self.sandbox_dir / tabular_file.name
            shutil.copy(tabular_file, destination)
            print(f"🔒 Sandboxed tabular file: {tabular_file.name}")

        # 2. Extract Text Documents (PDFs) for the vector database
        # We explicitly exclude CSV/XLSX from the text reader
        reader = SimpleDirectoryReader(
            input_dir=str(input_path),
            required_exts=[".pdf", ".txt", ".docx"],
            recursive=False
        )
        
        documents = reader.load_data()
        print(f"📚 Successfully loaded {len(documents)} text pages/nodes for vector processing.")
        return documents