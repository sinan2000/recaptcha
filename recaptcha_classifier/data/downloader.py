import requests
import zipfile
import shutil
from pathlib import Path
from alive_progress import alive_bar
from recaptcha_classifier.detection_labels import DetectionLabels


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
        self._expected_folder_names = DetectionLabels.dataset_classnames()

    def download(self) -> None:
        """
        Checks if dataset is already downloaded.
        If not, downloads and then unzips it.
        """
        if self._is_downloaded():
            print("Dataset already exists, skipping download.")
            return

        self._prepare_dest()
        print(f"Downloading {self._url} to {self._dest}...")
        self._download_zip()
        self._extract_zip()
        self._move_subfolders()
        self._delete_labels()
        self._flatten_images_folder()
        self._zip_path.unlink()
        print("Download and extraction completed successfully.")

    def _is_downloaded(self) -> bool:
        """
        Checks if the dataset is already downloaded and in the expected format.

        Returns:
            bool: True if the dataset is already downloaded, False otherwise.
        """
        if not self._dest.exists():
            return False
        folders = {p.name for p in self._dest.iterdir() if p.is_dir()}
        expected = set(self._expected_folder_names)
        return expected.issubset(folders)

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
    
    def _extract_zip(self) -> None:
        """
        Extracts the downloaded zip file to the destination directory.
        """
        with zipfile.ZipFile(self._zip_path) as z:
            z.extractall(self._dest)

    def _move_subfolders(self) -> None:
        """
        Moves the main images and labels subfolders
        from the extracted directory to the destination directory.
        Finally, it removes the main extracted directory that is now empty.
        """
        root = next(p for p in self._dest.iterdir() if p.is_dir() and p.name not in self._expected_folder_names)
        
        for sub in ("images", "labels"):
            source = root / sub
            if source.exists():
                target = self._dest / sub
                if target.exists():
                    shutil.rmtree(target)
                source.rename(target)
        
        if root.exists() and root.is_dir():
            root.rmdir()
    
    def _delete_labels(self) -> None:
        """
        Deletes the labels directory if it exists.
        """
        label_dir = self._dest / "labels"
        if label_dir.exists() and label_dir.is_dir():
            shutil.rmtree(label_dir)
    
    def _flatten_images_folder(self) -> None:
        images_dir = self._dest / "images"
        if not images_dir.exists():
            return

        for subfolder in images_dir.iterdir():
            if subfolder.is_dir():
                target_path = self._dest / subfolder.name
                if target_path.exists():
                    shutil.rmtree(target_path)
                subfolder.rename(target_path)
        
        if not any(images_dir.iterdir()):
            images_dir.rmdir()
