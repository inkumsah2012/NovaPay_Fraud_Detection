import pandas as pd
from pathlib import Path
import re
import numpy as np
import logging
from config.constant import Input_Data

logging.basicConfig(level=logging.INFO)

def data_ingestion():
    try:
        df = pd.read_csv(Input_Data)
        logging.info("Data successfully loaded")
        print(df.head(5))
        return df
    except Exception as e:
        logging.error(f"error occurred while loading the data {e}")

if __name__ == "__main__":
    data = data_ingestion()
    print(data.head())
#data_ingestion()