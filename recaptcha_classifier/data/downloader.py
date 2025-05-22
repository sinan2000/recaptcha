import requests
import zipfile
import logging
from pathlib import Path
from alive_progress import alive_bar

logger = logging.getLogger(__name__)


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
        self._progress = alive_bar

    def download(self) -> None:
        """
        Checks if dataset is already downloaded.
        If not, downloads and then unzips it.
        """
        if self._is_downloaded():
            logger.info("Dataset already exists, skipping download.")
            return

        self._prepare_dest()
        logger.info(f"Downloading {self._url} to {self._dest}...")
        self._download_zip()
        self._unzip_and_cleanup()

    def _is_downloaded(self) -> bool:
        """
        Checks if the dataset is already downloaded.

        Returns:
            bool: True if the dataset is already downloaded, False otherwise.
        """
        return self._dest.exists() and any(self._dest.iterdir())

    def _prepare_dest(self) -> None:
        """
        Creates the destination directory if it doesn't exist.
        """
        self._dest.mkdir(parents=True, exist_ok=True)

    def _fetch_stream(self) -> requests.Response:
        """
        Fetches the dataset stream from the URL.

        Returns:
            requests.Response: Response object containing the dataset stream.
        """
        return requests.get(self._url, stream=True)

    def _download_zip(self) -> None:
        """
        Downloads the dataset zip file.
        This method handles the download process and shows a progress bar.
        """
        resp = self._fetch_stream()
        resp.raise_for_status()
        total = int(resp.headers.get('Content-Length', 0))
        with (open(self._zip_path, "wb") as f,
              self._progress(total, title="Downloading") as bar):
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar(len(chunk))
        logger.info("Download completed successfully.")

    def _unzip_and_cleanup(self) -> None:
        """
        Unzips the downloaded dataset.
        Then it cleans up the zip file and its main extracted directory.

        Note: this assumes that the downloaded dataset has exactly
        the structure of our selected Kaggle dataset for simplicity.
        """
        logger.info("Extracting...")
        with zipfile.ZipFile(self._zip_path) as z:
            z.extractall(self._dest)

        root = next(p for p in self._dest.iterdir() if p.is_dir())
        for sub in ("images", "labels"):
            (root / sub).rename(self._dest / sub)

        root.rmdir()
        self._zip_path.unlink()
        print("Extraction and cleanup completed successfully.")
