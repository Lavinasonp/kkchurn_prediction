import logging
import argparse
from src.ingestion import ingest_data
from src.feature_engineering import run_feature_engineering
from src.training import train_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(step):
    """
    Main execution entry point.
    """
    try:
        if step == "all" or step == "ingest":
            logging.info(">>>>>> STAGE 1: Data Ingestion (CSV -> Parquet) <<<<<<")
            ingest_data()
            
        if step == "all" or step == "features":
            logging.info(">>>>>> STAGE 2: Feature Engineering (Dask Aggregation) <<<<<<")
            run_feature_engineering()
            
        if step == "all" or step == "train":
            logging.info(">>>>>> STAGE 3: Model Training (LightGBM) <<<<<<")
            train_model()
            
    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")
        raise e

if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["ingest", "features", "train", "all"], default="all",
                        help="Which step of the pipeline to run.")
    args = parser.parse_args()
    
    main(args.step)