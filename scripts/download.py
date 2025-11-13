import os
import zipfile
import kaggle

dataset = "evilspirit05/visdrone"
download_path = os.path.join(os.path.dirname(__file__), "../data")

def ensure_kaggle_config():
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_path):
        raise FileNotFoundError(
            "⚠️ Kaggle API token not found.\n"
            "Please place your kaggle.json file in ~/.kaggle/ or C:\\Users\\<user>\\.kaggle\\"
        )
    os.chmod(kaggle_path, 0o600)
    print("✅ Kaggle API token found.")

def download():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=download_path, unzip=True, quiet=False)

if __name__ == "__main__":
    ensure_kaggle_config()
    download()