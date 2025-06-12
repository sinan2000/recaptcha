import unittest
import torch
import os
from recaptcha_classifier.models.main_model import MainCNN
from recaptcha_classifier.train.training import Trainer
from recaptcha_classifier.data.pipeline import DataPreprocessingPipeline
from recaptcha_classifier.detection_labels import DetectionLabels


class TestCheckpointIntegration(unittest.TestCase):
    def test_checkpoint_save_load(self):
        pipeline = DataPreprocessingPipeline(
            DetectionLabels,
            batch_size=2,
            num_workers=0
            )
        
        loaders = pipeline.run()
        
        model = MainCNN(n_layers=1, kernel_size=3, num_classes=len(DetectionLabels))
        # optimizer = torch.optim.Adam(model.parameters())
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        trainer = Trainer(
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            device=device,
            epochs=1,
            save_folder="checkpoints"
        )
        
        trainer.train(model)
        
        model_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        model_new = MainCNN(n_layers=1, kernel_size=3, num_classes=len(DetectionLabels))
        
        trainer.load_checkpoint_states(model_new)
        
        for key in model_state:
            self.assertTrue(
                torch.equal(model_state[key], model_new.state_dict()[key])
            )
            
        trainer.delete_checkpoints()