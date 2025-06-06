import os
import pickle

import cv2
import numpy as np
import quantus
import torch
from PIL import Image
from gradcam.utils import visualize_cam
from matplotlib import pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import Subset, DataLoader

from recaptcha_classifier import DetectionLabels, DataPreprocessingPipeline, MainCNN


class Explainability(object):
    """ Class for generating and evaluating explanations. """

    def __init__(self, model: MainCNN, n_samples:int = 400):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        # -1 gets last block; 0 gets the conv2d layer specifically:
        self._target_layers = [self.model.layers[-1][0]]
        self.folder = "Explanations"
        self.n_samples = n_samples
        self._test_dataloader = None
        self._get_test_dataset()
        self._config_folders()


    def _get_test_dataset(self) -> None:
        """ Getting test dataset from DataPreprocessingPipeline. """
        if self._test_dataloader is None:
            data = DataPreprocessingPipeline(
                DetectionLabels, balance=True)
            self._test_dataloader = data.run()['test']
          # number of images
        self._dataset = Subset(self._test_dataloader.dataset,
                               indices=list(range(self.n_samples)))
        self._test_dataloader = DataLoader(self._dataset, batch_size=1)
        print(f"Test subset len: {len(self._dataset)}")

    def _calculate_mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        psum = torch.tensor([0.0, 0.0, 0.0])
        psum_sq = torch.tensor([0.0, 0.0, 0.0])
        nb_samples = 0

        for inputs, _ in self._test_dataloader:
            # inputs shape: (batch_size, 3, height, width)
            psum += inputs.sum(dim=[0, 2, 3])  # Sum over batch, height, width
            psum_sq += (inputs ** 2).sum(dim=[0, 2, 3])
            nb_samples += inputs.shape[0] * inputs.shape[2] * inputs.shape[3]  # batch * height * width

        mean = psum / nb_samples
        var = (psum_sq / nb_samples) - (mean ** 2)
        std = torch.sqrt(var)

        mean_tensor = mean.view(-1, 1, 1)  # shape (3, 1, 1)
        std_tensor = std.view(-1, 1, 1)  # shape (3, 1, 1)

        return mean_tensor.cpu().numpy(), std_tensor.cpu().numpy()


    def _get_input_tensors(self):
        mean, stds = self._calculate_mean_std()

        input_tensors = []

        for img_tensor, _ in self._dataset:
            img_tensor = img_tensor.reshape(224, 224, 3)
            rgb_img = img_tensor.cpu().numpy()
            # rgb_img = np.float32(image) / 255.0  # scaling pixels to [0,1]
            input_tensor = preprocess_image(rgb_img, mean=mean, std=stds)
            input_tensors.append(input_tensor)

        # Concat all input tensors along batch dimension
        batch_tensor = torch.cat(input_tensors, dim=0)  # shape: (N, 3, H, W)
        return batch_tensor



    def gradcam_generate_explanations(self):

        explanations = []
        input_tensors = self._get_input_tensors()

        self.model.eval()

        self.model.to(self.device)

        with GradCAM(model=self.model, target_layers=self._target_layers) as cam:
            for i, tensor in enumerate(input_tensors):
                if (i + 1) % 25 == 0:
                    print(f'Processing image {i + 1}/{len(input_tensors)}')

                tensor = tensor.to(self.device)
                expanded_tensor = tensor.unsqueeze(0)  # Add batch dimension

                # Normalize for visualization (between 0 and 1)
                img = tensor.permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min())

                # Automatically use top predicted class
                outputs = self.model.forward(expanded_tensor)
                pred_class = outputs.argmax(dim=1).item()
                targets = [ClassifierOutputTarget(pred_class)]

                # Generate CAM
                cam_output = cam(input_tensor=expanded_tensor,
                                    targets=targets)[0, :]
                explanations.append(cam_output)

                # Overlay CAM on image
                visualization = show_cam_on_image(img, cam_output, use_rgb=True)

                # Save visualization
                vis_name = f'img_{i+1}.jpg'
                if vis_name not in os.listdir(self.folder_vis): # applicable to code above
                    vis_img = Image.fromarray(visualization)
                    vis_img.save(os.path.join(self.folder_vis, vis_name))

        # Save all raw explanations
        np.save(os.path.join(self.folder_explain, 'explanations.npy'), np.asarray(explanations))
        print(f"Saved {len(explanations)} GradCAM visualizations to {self.folder_vis}")

    def _config_folders(self):
        os.makedirs(self.folder, exist_ok=True)
        save_folder = os.path.join(self.folder, 'outputs')
        self.folder_vis = os.path.join(save_folder, 'visualizations')
        self.folder_explain = os.path.join(save_folder, 'explain')
        os.makedirs(self.folder_vis, exist_ok=True)
        os.makedirs(self.folder_explain, exist_ok=True)
        return self.folder_explain, self.folder_vis


    def overlay_image(self, index: int = 0, alpha: float = 0.4) -> None:
        if not os.path.exists(self.folder_vis):
            print(f"{self.folder_vis} not found. Run gradcam_generate_explanations() first.")
            return

        original_img = self._dataset[index][0].cpu().numpy()

        heatmap = cv2.imread(os.path.join(self.folder_vis, f'img_{index+1}.jpg'))  # Loaded as BGR by default
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Assume original_img is a NumPy array in RGB format, heatmap is single-channel float [0,1]
        heatmap_hw3 = heatmap if heatmap.shape[0] == 224 else np.transpose(heatmap, (1, 2, 0))
        original_hw3 = np.transpose(original_img, (1, 2, 0)) if original_img.shape[0] == 3 else original_img

        heatmap_clipped = np.clip(heatmap_hw3, 0, 1)

        # Scale to [0,255] and round before converting to uint8
        heatmap_uint8 = (heatmap_clipped * 255).round().astype(np.uint8)

        # Now apply colormap
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Convert original image to BGR if needed (OpenCV uses BGR)
        original_bgr = cv2.cvtColor(original_hw3, cv2.COLOR_RGB2BGR)

        if original_bgr.dtype == np.float32 or original_bgr.dtype == np.float64:
            original_bgr = (original_bgr * 255).round().astype(np.uint8)

        # Overlay heatmap with alpha transparency (e.g., alpha=0.4)
        overlayed_img = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_color, alpha, 0)

        # Convert back to RGB if you want to display with matplotlib or PIL
        overlayed_img_rgb = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)

        plt.imshow(overlayed_img_rgb)
        # plt.colorbar()


    def evaluate_explanations(self):
        self.model.eval()

        if not os.path.exists(self.folder_explain):
            print(f"{self.folder_explain} not found. Run gradcam_generate_explanations() first.")
            return

        a_batch_saliency_ce_model = np.load(os.path.join(self.folder_explain,'explanations.npy'))
        a_batch_test = a_batch_saliency_ce_model[:32]

        dataloader = DataLoader(dataset=self._dataset, batch_size=32, shuffle=False)
        x_batch, y_batch = next(iter(dataloader))
        x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()


        metric = quantus.IROF(return_aggregate=False)
        # print(metric.get_params)

        scores = metric(
            model=self.model,
            channel_first=True,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch_test,
            device=self.device.type,
            explain_func=quantus.explain,
            explain_func_kwargs={"method": "Saliency"}
        )
        print(len(scores))
        np.savetxt(os.path.join(self.folder, "mainCNN_eval_scores.csv"), scores, delimiter=",")
