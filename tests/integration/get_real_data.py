from src.data.pipeline import DataPreprocessingPipeline


def get_real_dataloaders(batch_size=4):
    class_map = {
        'BICYCLE': 0,
        'BRIDGE': 1,
        'BUS': 2,
        'CAR': 3,
        'CHIMNEY': 4,
        'CROSSWALK': 5,
        'HYDRANT': 6,
        'MOTORCYCLE': 7,
        'PALM': 8,
        'STAIR': 9,
        'TRAFFIC_LIGHT': 10,
        'OTHER': 11
    }

    pipeline = DataPreprocessingPipeline(
        class_map=class_map,
        batch_size=batch_size,
        show_plots=False
    )

    return pipeline.run()
