from typing import List, Tuple


class BoundingBoxScaler:
    """
    This class handles the complexity of adjusting bounding box coordinates
    when resizing images. It acts as an adapter between the raw YOLO bounding
    box format and the resized target image format.

    Design Pattern: Adapter
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 target_size: Tuple[int, int]):
        """
        Initialize the BoundingBoxScaler

        Args:
            input_size (Tuple[int, int]): Original dimensions (width, height)
            target_size (Tuple[int, int]): Target resized dimensions
            (width, height)
        """
        self.input_size = input_size
        self.target_size = target_size
        self.width_ratio = target_size[0] / input_size[0]
        self.height_ratio = target_size[1] / input_size[1]

    def rescale(self, bounding_box: List[float]) -> List[float]:
        """
        Rescale a single YOLO bounding box.

        Args:
            bounding_box (List[float]): YOLO bounding box
            [class, x_center, y_center, width, height]

        Returns:
            List[float]: Rescaled bounding box, in the same format.
        """
        cls, x_center, y_center, width, height = bounding_box
        x_center *= self.width_ratio
        y_center *= self.height_ratio
        width *= self.width_ratio
        height *= self.height_ratio
        return [cls, x_center, y_center, width, height]

    def rescale_batch(self,
                      bounding_boxes: List[List[float]]
                      ) -> List[List[float]]:
        """
        Rescale a batch of YOLO bounding boxes.

        Args:
            bounding_boxes (List[List[float]]): List of YOLO bounding boxes

        Returns:
            List[List[float]]: List of rescaled bounding boxes
        """
        return [self.rescale(box) for box in bounding_boxes]
