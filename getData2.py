import kagglehub

# Download latest version
path = kagglehub.dataset_download("swish9/weeds-detection")

print("Path to dataset files:", path)
