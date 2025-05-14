from typing import List, Tuple, Dict
from PIL import Image
from pathlib import Path
from torch import Tensor

# OBJECT DETECTION TASK CLASSES
# A (image, label) pair containing their system paths/ locations
FilePair = Path  # Tuple[Path, Path]

# A list of (image, label) pairs, for more items in the dataset
FilePairList = List[FilePair]

# A dictionary where the keys are class names and
# the values are lists of (image, label) pairs, elements of that class
ClassFileDict = Dict[str, FilePairList]

# A nested dictionary where main keys are train/val/test
# and the subkeys are class names
DatasetSplitDict = Dict[str, ClassFileDict]

# YOLO bounding box format (x_center, y_center, width, height)
BBox = Tuple[float, float, float, float]

# List of bounding boxes for one image from the dataset
BBoxList = List[BBox]

# A dataset item, of (image, annotations), where both features
# now loaded in memory, instead of Paths like in FilePair
DataPair = Tuple[Image.Image, BBoxList]

# A final dataset item, that now contains the image tensor,
# the bounding boxes and the class id; ready for model training
DataItem = Tuple[Tensor, BBoxList, int]

# Output of the dataloader, a batch of data items
DataBatch = Tuple[Tensor, List[BBoxList], Tensor]
