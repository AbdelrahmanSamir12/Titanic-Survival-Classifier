import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_titanic_data():
    # Ensure data directory exists
    os.makedirs("data/raw", exist_ok=True)
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download Titanic dataset
    api.competition_download_files(
        "titanic",
        path="data/raw",
        unzip=True
    )
    
    print("Data downloaded to data/raw/")

if __name__ == "__main__":
    download_titanic_data()