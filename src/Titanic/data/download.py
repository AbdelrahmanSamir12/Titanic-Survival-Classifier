import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_titanic_data():
    # Set your specific data path
    raw_data_path = "/teamspace/studios/this_studio/mine/mlops/data/raw"
    
    # Ensure data directory exists
    os.makedirs(raw_data_path, exist_ok=True)
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download Titanic dataset
    api.competition_download_files(
        "titanic",
        path=raw_data_path
    )
    
    # Unzip the downloaded file
    zip_path = os.path.join(raw_data_path, "titanic.zip")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_data_path)
    
    # Remove the zip file
    os.remove(zip_path)
    
    print(f"Data downloaded and extracted to {raw_data_path}")

if __name__ == "__main__":
    download_titanic_data()