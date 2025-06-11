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
        self._scores = None
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
            if img_tensor.shape[0] == 3:
                rgb_img = img_tensor.permute(1, 2, 0).cpu().numpy()  # to (224, 224, 3)
            else:
                rgb_img = img_tensor.cpu().numpy()
            # im = Image.fromarray(rgb_img)
            # print(im)
            #plt.show()
            # rgb_img = np.float32(image) / 255.0  # scaling pixels to [0,1]
            input_tensor = preprocess_image(rgb_img, mean=mean, std=stds)
            input_tensors.append(input_tensor)

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

                # normalization for visualization (between 0 and 1)
                img = tensor.permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min())

                outputs = self.model.forward(expanded_tensor)
                pred_class = outputs.argmax(dim=1).item()
                targets = [ClassifierOutputTarget(pred_class)]

                cam_output = cam(input_tensor=expanded_tensor,
                                    targets=targets)[0, :]
                explanations.append(cam_output)

                visualization = show_cam_on_image(img, cam_output, use_rgb=True)

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



    def overlay_image(self, index: int = 1, img_opacity:int = 0.5) -> None:
        if not os.path.exists(self.folder_vis):
            print(f"{self.folder_vis} not found. Run gradcam_generate_explanations() first.")
            return

        if index < 1:
            raise ValueError("Index should be greater than 0.")

        original_img = self._dataset[index-1][0].cpu().numpy()

        heatmap = cv2.imread(os.path.join(self.folder_vis, f'img_{index}.jpg'))  # Loaded as BGR by default
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB

        heatmap = heatmap if heatmap.shape[0] == 224 else np.transpose(heatmap, (1, 2, 0))
        original_img = np.transpose(original_img, (1, 2, 0)) if original_img.shape[0] == 3 else original_img

        vis = show_cam_on_image(original_img, heatmap, use_rgb=True, image_weight=img_opacity)
        plt.imshow(vis)
        plt.axis('off')
        # plt.title(f"True Class: {DetectionLabels.from_id(self._dataset[index-1][1])}")
        plt.show()


    def evaluate_explanations(self, n:int = 100):
        self.model.eval()

        if not os.path.exists(self.folder_explain):
            print(f"{self.folder_explain} not found. Run gradcam_generate_explanations() first.")
            return

        a_batch_saliency_ce_model = np.load(os.path.join(self.folder_explain,'explanations.npy'))
        a_batch_test = a_batch_saliency_ce_model[:n]

        dataloader = DataLoader(dataset=self._dataset, batch_size=n, shuffle=False)
        x_batch, y_batch = next(iter(dataloader))
        x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()


        metric = quantus.IROF(return_aggregate=False)

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
        np.savetxt(os.path.join(self.folder, "mainCNN_eval_scores.csv"), scores, delimiter=",")

    def aggregate_eval(self):
        if self._scores is None:
            try:
                self._scores = np.loadtxt(os.path.join(self.folder, "mainCNN_eval_scores.csv"))
            except ValueError as e:
                raise ValueError("No scores found. Run evaluate_explanations() first.")
        self._scores = np.array(self._scores)
        print("=== Explainability Evaluation Stats: ===")
        print(f"Mean: {np.mean(self._scores)}")
        print(f"STD: {self._scores.std()}")
        print(f"Median: {np.median(self._scores)}")
        print(f"Variance: {np.var(self._scores)}")

    def evaluate_explanations_index(self, index: int):
        """Evaluate explanations for a specific dataset index."""
        self.model.eval()

        if not os.path.exists(self.folder_explain):
            print(f"{self.folder_explain} not found. Run gradcam_generate_explanations() first.")
            return

        explanations = np.load(os.path.join(self.folder_explain, 'explanations.npy'))
        if index < 0 or index >= len(explanations):
            print(f"Index {index} out of range [0-{len(explanations) - 1}]")
            return

        a_batch_test = explanations[index:index + 1]  # Shape: (1, H, W)

        x_single, y_single = self._dataset[index]
        x_batch = x_single.unsqueeze(0).cpu().numpy()  # Add batch dimension
        y_batch = np.array([y_single])  # Create batch of size 1

        x = x_single
        explanation = a_batch_test[index] if a_batch_test.shape[0] > 1 else a_batch_test[0]
        self.visualize_sample(x, explanation, title_prefix=f"Sample {index}: ")

        metric = quantus.IROF(return_aggregate=False, abs=True)
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

        print(f"Score for index {index}: {scores[0]}")
        return scores[0]


    def visualize_sample(self, x, explanation, title_prefix=""):
        """
        Visualize an input image and its corresponding explanation/saliency map.
        Args:
            x: Input image as a numpy array (shape: H x W x 3 or 3 x H x W)
            explanation: Saliency map as a numpy array (shape: H x W)
            title_prefix: Optional string for plot titles
        """
        # H x W x 3 format
        if x.shape[0] == 3 and len(x.shape) == 3:
            img = np.transpose(x, (1, 2, 0))
        else:
            img = x

        if img.max() > 1.0:
            img_disp = img / 255.0
        else:
            img_disp = img

        exp_disp = explanation
        if exp_disp.max() > 0:
            exp_disp = exp_disp / exp_disp.max()

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(img_disp)
        plt.title(f"{title_prefix}Input Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(exp_disp, cmap='jet')
        plt.title(f"{title_prefix}Saliency Map")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        # Overlay saliency map on image
        plt.imshow(img_disp, alpha=0.6)
        plt.imshow(exp_disp, cmap='jet', alpha=0.4)
        plt.title(f"{title_prefix}Overlay")
        plt.axis('off')

        plt.tight_layout()
        plt.show()



