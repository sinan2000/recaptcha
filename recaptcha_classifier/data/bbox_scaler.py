from typing import Tuple
from .types import BBoxList


class BoundingBoxScaler:
    """
    This class is responsible for scaling the bounding boxes
    based on the transform applied to the image. It is only
    used inside the AugmentationPipeline class, to adjust
    the coordinates of the bounding boxes after applying
    transformations to the image.
    """

    @staticmethod
    def scale_for_flip(bboxes: BBoxList) -> BBoxList:
        """
        Adjusts the bounding boxes for horizontal flip.

        Args:
            bboxes (BBoxList): List of bounding boxes in YOLO format
            (x_center, y_center, width, height).

        Returns:
            BBoxList: List of scaled bounding boxes.
        """
        return [(1 - x, y, w, h) for (x, y, w, h) in bboxes]

    @staticmethod
    def scale_for_rotation(bboxes: BBoxList,
                           angle: float,
                           size: Tuple[int, int]) -> BBoxList:
        """
        Adjusts the bounding boxes for rotation.

        Args:
            bboxes (BBoxList): List of bounding boxes in YOLO format
            (x_center, y_center, width, height).
            angle (float): Angle of rotation.
            size (Tuple[int, int]): Size of the image. (width, height)

        Returns:
            BBoxList: List of scaled bounding boxes.
        """
        import math
        width, height = size
        angle_rad = math.radians(angle)
        c_x, c_y = width / 2, height / 2  # center coordinates of the image

        n_ann = []  # the new bounding boxes, that we will return

        for x, y, w, h in bboxes:
            # calculate pixel coordinates from bb
            x0, y0 = x * width, y * height
            bw, bh = w * width, h * height

            # calculate corner coordinates
            corners = [
                (x0 - bw / 2, y0 - bh / 2),
                (x0 + bw / 2, y0 - bh / 2),
                (x0 + bw / 2, y0 + bh / 2),
                (x0 - bw / 2, y0 + bh / 2)
            ]

            # rotate the corners
            new_corners = []
            for cx, cy in corners:
                # rotate the corners around the center of the image
                # formula at https://en.wikipedia.org/wiki/Rotation_matrix
                x_rot = (math.cos(angle_rad) * (cx - c_x) -
                         math.sin(angle_rad) * (cy - c_y) +
                         c_x)
                y_rot = (math.sin(angle_rad) * (cx - c_x) +
                         math.cos(angle_rad) * (cy - c_y) +
                         c_y)

                new_corners.append((x_rot, y_rot))

            # calculate new bounding box
            x_min = min(c[0] for c in new_corners)
            x_max = max(c[0] for c in new_corners)
            y_min = min(c[1] for c in new_corners)
            y_max = max(c[1] for c in new_corners)

            new_x = (x_min + x_max) / (2 * width)
            new_y = (y_min + y_max) / (2 * height)
            new_w = (x_max - x_min) / width
            new_h = (y_max - y_min) / height

            n_ann.append((new_x, new_y, new_w, new_h))

        return n_ann
