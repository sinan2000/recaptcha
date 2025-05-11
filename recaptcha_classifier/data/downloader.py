import requests
import zipfile
import time
from pathlib import Path
from alive_progress import alive_bar


class DatasetDownloader:
    """
    Simple utility class for downloading and unzipping datasets.
    This detects if dataset is already downloaded and skips the process if so.

    It follows Single Responsibility Principle (SRP) as it only handles
    dataset downloading operation, with no other responsibilities.
    """
    def __init__(self,
                 url: str = ("https://www.kaggle.com/api/v1/datasets/"
                             "download/mikhailma/test-dataset"),
                 dest: str = "data") -> None:
        """
        Initializes the downloader instance.

        Args:
            url (str): URL of the Kaggle dataset to download.
            dest (str): Destination directory to save the dataset.
        """
        self._url: str = url
        self._dest: Path = Path(dest)
        self._zip_path: Path = self._dest / "dataset.zip"

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

            with (open(self._zip_path, "wb") as f,
                  alive_bar(total_size, title="Downloading dataset") as bar):
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar(len(chunk))
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
