from typing import List, Tuple
from PIL import Image
from pathlib import Path
from torch import Tensor

BoundingBoxList = List[Tuple[float, float, float, float]]

DatasetItem = Tuple[Image.Image, BoundingBoxList]

LoaderPair = Tuple[Path, Path]

HandlerItem = Tuple[Tensor, BoundingBoxList, int]
