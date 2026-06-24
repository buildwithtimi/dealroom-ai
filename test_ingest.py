# test_ingest.py
import os
import importlib

# 🟢 Dynamically import modules from folders starting with numbers
parser_module = importlib.import_module("backend.app.engine.01_ingestion.parser")
indexer_module = importlib.import_module("backend.app.engine.01_ingestion.indexer")

# Extract the classes from the dynamically loaded modules
DealRoomParser = parser_module.DealRoomParser
DealRoomIndexer = indexer_module.DealRoomIndexer

# --- Rest of your script remains exactly the same ---

# 1. Setup sample raw folders
os.makedirs("sample_input_files", exist_ok=True)

# Make a fake CSV and fake text file to test routing
with open("sample_input_files/burn_rate.csv", "w") as f:
    f.write("Month,Burn_Rate,Runway\nJan,50000,12\nFeb,48000,11")
    
with open("sample_input_files/pitch_deck.txt", "w") as f:
    f.write("Welcome to DealRoom AI. We are transforming startup due diligence analytics.")

print("🚀 Running ingestion pipeline test...")
parser = DealRoomParser()
indexer = DealRoomIndexer()

# Run the system
documents = parser.process_files("sample_input_files")
index = indexer.build_vector_store(documents)

print("✅ Ingestion works! Check 'backend/app/engine/data_workspace/' for your sandboxed CSV.")