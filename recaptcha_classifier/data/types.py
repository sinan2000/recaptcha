from typing import List, Tuple, Dict
from PIL import Image
from pathlib import Path
from torch import Tensor

# A list of image paths, for more items in the dataset
ImagePathList = List[Path]

# A dictionary where the keys are class names and
# the values are lists of all image paths for that class in the dataset
ClassToImgPaths = Dict[str, ImagePathList]

# A nested dictionary where main keys are train/val/test
# and the subkeys are class names, containing all image paths of that class
DatasetSplitMap = Dict[str, ClassToImgPaths]

# A dataset item, where the image is now loaded from disk
LoadedImg = Image.Image

# A final dataset item, that now contains the image tensor,
# and the class id; ready for model training
DataItem = Tuple[Tensor, Tensor]

# Output of the dataloader, a batch of data items
DataBatch = Tuple[Tensor, Tensor]
