import requests
import zipfile
import time
from pathlib import Path


class KaggleDatasetDownloader:
    """
    Simple class to handle downloading of Kaggle datasets.
    This detects if dataset is already downloaded and skips the process if so.
    """
    def __init__(self, url: str, dest: str = "../../data") -> None:
        """
        Initializes the downloader instance.

        Args:
            url (str): URL of the Kaggle dataset to download.
            dest (str): Destination directory to save the dataset.
        """
        self._url = url
        self._dest = Path(dest)
        self._zip_path = self._dest / "dataset.zip"

    def download(self) -> None:
        """
        Checks if dataset is already downloaded.
        If not, downloads and then unzips it.
        """
        if self._dest.exists() and any(self._dest.iterdir()):
            print("Dataset already exists, skipping download.")
            return

        self._dest.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {self._url} to {self._dest}...")
        response = requests.get(self._url, stream=True)

        if response.status_code == 200:
            total_size = int(response.headers.get('Content-Length', 0))
            with open(self._zip_path, "wb") as f:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    self._print_progress(downloaded_size, total_size)
            print("\nDownload completed successfully.")
        else:
            print(f"Failed to download dataset: {response.status_code}")
            return

        time.sleep(0.5)  # nice touch, feels good to have
        self._unzip()

    def _unzip(self) -> None:
        """
        Unzips the downloaded dataset.
        Then it cleans up the zip file and its main extracted directory.

        Note: this assumes that the downloaded dataset has exactly
        the structure of our selected Kaggle dataset for simplicity.
        """
        print("Extracting downloaded dataset")
        with zipfile.ZipFile(self._zip_path, 'r') as zip_ref:
            zip_ref.extractall(self._dest)

        root = next(p for p in self._dest.iterdir() if p.is_dir())
        for sub in ("images", "labels"):
            (root / sub).rename(self._dest / sub)

        root.rmdir()
        self._zip_path.unlink()
        print("Extraction and cleanup completed successfully.")

    @staticmethod
    def _print_progress(downloaded: int, total: int) -> None:
        percent = downloaded / total * 100
        bar_length = 40
        filled_length = int(bar_length * percent // 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"\r|{bar}| {percent:.2f}%", end='')


if __name__ == "__main__":
    DATASET_URL = ("https://www.kaggle.com/api/v1/datasets/"
                   "download/mikhailma/test-dataset")
    DEST_DIR = "../../data"
    KaggleDatasetDownloader(DATASET_URL, DEST_DIR).download()
