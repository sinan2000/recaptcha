from typing import List, Tuple
from PIL import Image

BoundingBoxList = List[Tuple[float, float, float, float]]

DatasetItem = Tuple[Image.Image, BoundingBoxList]
