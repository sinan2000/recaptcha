from enum import Enum


class DetectionLabels(Enum):
    """
    Enum for improving readability of the object classes.
    """

    """
    OBJECT DETECTION TASK CLASSES
    CROSSWALK = 0
    CHIMNEY = 1
    STAIR = 2
    """
    BICYCLE = 0
    BRIDGE = 1
    BUS = 2
    CAR = 3
    CHIMNEY = 4
    CROSSWALK = 5
    HYDRANT = 6
    MOTORCYCLE = 7
    PALM = 8
    STAIR = 9
    TRAFFIC_LIGHT = 10
    OTHER = 11

    @classmethod
    def from_name(cls, name: str) -> int:
        """
        Converts a class name to its corresponding integer value.

        Args:
            name (str): The name of the class.

        Returns:
            int: The integer value of the class.
        """
        upper = name.upper().replace(" ", "_")
        if upper not in cls.__members__:
            raise ValueError(f"Class '{name}' does not exist. Please define it"
                             "into the labels.py file.")
        return cls[upper].value

    @classmethod
    def all(cls) -> list:
        """
        Returns the list of all classes.

        Returns:
            list: List of all classes.
        """
        return list(cls)

    @classmethod
    def to_class_map(cls) -> dict:
        """
        Convert the enum to a dictionary.

        Returns:
            dict: Dictionary representation of the enum.
        """
        return {cl.name.capitalize().replace("_", " "):
                cl.value for cl in cls}

    @classmethod
    def from_id(cls, id: int) -> str:
        """
        Converts an integer ID to its corresponding class name.

        Args:
            id (int): The ID of the class.

        Returns:
            str: The name of the class.
        """
        for name, member in cls.__members__.items():
            if member.value == id:
                return name.capitalize().replace("_", " ")

        raise ValueError(f"Class ID {id} does not exist in DetectionLabels.")

    @classmethod
    def dataset_classnames(cls) -> list:
        """
        Returns a list of class names, only with first letter capitalized.
        We use it for the pair loader, as that is the format of the folders
        downloaded from the dataset.

        Returns:
            list: List of class names.
        """
        return [name.capitalize().replace("_", " ")
                for name in cls.__members__.keys()]
