import os
import yaml
import shutil
import logging
import dask.dataframe as dd

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ingest_data(config_path="config/config.yaml"):
    """
    Reads raw CSVs from data/raw, converts them to Parquet, 
    and saves them to data/processed.
    """
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    raw_dir = config['directories']['raw_data']
    processed_dir = config['directories']['processed_data']
    
    # Map friendly names to actual filenames from config
    files = {
        'train': config['files']['train'],
        'members': config['files']['members'],
        'transactions': config['files']['transactions'],
        'user_logs': config['files']['user_logs']
    }

    # 2. Process each file
    for key, filename in files.items():
        input_path = os.path.join(raw_dir, filename)
        output_path = os.path.join(processed_dir, f"{key}.parquet")
        
        # Validation: Check if input exists
        if not os.path.exists(input_path):
            logging.error(f"❌ Input file not found: {input_path}")
            continue

        logging.info(f"🚀 Processing {key} ({filename})...")

        try:
            # 3. Read with Dask
            # We explicitly set dtypes for columns that often cause issues in mixed CSVs
            dtype_options = {
                'gender': 'object', 
                'msno': 'object',
                'payment_method_id': 'object',
                'city': 'object',
                'registered_via': 'object'
            }
            
            ddf = dd.read_csv(
                input_path, 
                blocksize="64MB", 
                dtype=dtype_options,
                assume_missing=True # Helps with int columns containing NaNs
            )
            
            # Standardize columns (Your Logic: lower + strip)
            ddf.columns = [c.lower().strip() for c in ddf.columns]

            # 4. Write to Parquet (Clean old run first)
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
                
            ddf.to_parquet(
                output_path, 
                engine='pyarrow', 
                compression='zstd', 
                write_index=False
            )
            
            logging.info(f"✅ Successfully converted {key} to Parquet.")

        except Exception as e:
            logging.error(f"❌ Failed to ingest {key}: {e}")
            raise e

if __name__ == "__main__":
    # Test run
    ingest_data()