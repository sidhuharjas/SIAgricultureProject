import os
import kagglehub

# Force kagglehub to download to your current working directory
os.environ["KAGGLEHUB_CACHE_DIR"] = os.getcwd()

# Download the latest version
path = kagglehub.dataset_download("ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes")

print("Path to dataset files:", path)
