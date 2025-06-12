import os
import cv2
import numpy as np
import quantus
import torch
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import Tensor
from torch.utils.data import Subset, DataLoader
from src import (
    DetectionLabels, DataPreprocessingPipeline, MainCNN)
from src.XAI.utils_wrapped_model import WrappedModel


class Explainability(object):
    """ Class for generating and evaluating explanations. """

    def __init__(self, model: MainCNN, n_samples: int = 400):
        """ Constructor for Explainability class.
        Args:
            model (MainCNN): Main CNN model for image classification.
            n_samples (int): Number of samples to evaluate.
        """
        self._scores = None
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)

        # -1 gets last block; 0 gets the conv2d layer specifically:
        try:
            self._target_layers = [self.model.layers[-1][0]]
        except AttributeError:
            try:
                self._target_layers = [self.model.res_blocks[-1]]
            except ModuleNotFoundError:
                raise ModuleNotFoundError("Model's last layers were not found.")
        self.folder = "Explanations"
        self.n_samples = n_samples
        self._test_dataloader = None
        self._get_test_dataset()
        self._config_folders()

    def run(self, eval_percent_samples: float = 0.5) -> None:
        """Runs the main class methods (pipeline method).
        Args:
            eval_percent_samples (float): fraction of explanations to evaluate
        """
        if eval_percent_samples < 0 or eval_percent_samples > 1:
            raise ValueError("eval_percent_samples should be between 0 and 1")
        self.gradcam_generate_explanations()
        self.evaluate_explanations(n=int(
            eval_percent_samples * self.n_samples))
        self.aggregate_eval()

    def gradcam_generate_explanations(self) -> None:
        """Generates GradCam explanations (rgb).
        Results saved in Explainability folder."""
        self.model.eval()
        self.model.to(self.device)
        explanations = []
        input_tensors = self._get_input_tensors()

        with GradCAM(
             model=self.model, target_layers=self._target_layers) as cam:
            for i, tensor in enumerate(input_tensors):
                if (i + 1) % 25 == 0:
                    print(f'Processing image {i + 1}/{len(input_tensors)}')

                tensor = tensor.to(self.device)
                # Add batch dimension
                expanded_tensor = tensor.unsqueeze(0)

                # normalization for visualization (between 0 and 1)
                img = tensor.permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min())

                outputs = self.model.forward(expanded_tensor)
                pred_class = outputs.argmax(dim=1).item()
                targets = [ClassifierOutputTarget(pred_class)]

                cam_output = cam(input_tensor=expanded_tensor,
                                 targets=targets)[0, :]
                explanations.append(cam_output)

                visualization = show_cam_on_image(img, cam_output,
                                                  use_rgb=True)

                vis_name = f'img_{i+1}.jpg'
                if vis_name not in os.listdir(self.folder_vis):
                    vis_img = Image.fromarray(visualization)
                    vis_img.save(os.path.join(self.folder_vis, vis_name))

        np.save(os.path.join(self.folder_explain, 'explanations.npy'),
                np.asarray(explanations))
        print(f"Saved {len(explanations)} GradCAM visualizations to {
            self.folder_vis}")

    def evaluate_explanations(self, n: int = 100) -> list | None:
        """Evaluate explanations of a specific amount.
        Args:
            n: number of explanations to evaluate
        Returns:
            None if explanations path is not found.
            List of scores for each explanation otherwise.
        """
        self.model.eval()

        if not os.path.exists(self.folder_explain):
            print(f"{self.folder_explain} not found. "
                  f"Run gradcam_generate_explanations() first.")
            return None

        a_batch_saliency_ce_model = np.load(os.path.join(
            self.folder_explain, 'explanations.npy'))
        a_batch_test = a_batch_saliency_ce_model[:n]

        dataloader = DataLoader(dataset=self._dataset,
                                batch_size=n,
                                shuffle=False)
        x_batch, y_batch = next(iter(dataloader))
        x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

        scores = self._run_evaluation_irof(a_batch_test, n, x_batch, y_batch)

        if scores:
            np.savetxt(os.path.join(self.folder,
                                    "mainCNN_eval_scores.csv"),
                       scores,
                       delimiter=",")
            print(f"Saved {len(scores)} valid evaluation scores.")
        else:
            print("No valid scores to save.")
        return scores

    def evaluate_explanations_index(self, index: int) -> None:
        """Evaluate explanations for a specific dataset index.
        Args:
            index: Index of preexisting image & saliency map. (From 1)
        """
        self.model.eval()
        n = 1
        index = index - 1

        if not os.path.exists(self.folder_explain):
            print(f"{self.folder_explain} not found. "
                  f"Run gradcam_generate_explanations() first.")
            return

        explanations = np.load(os.path.join(self.folder_explain,
                                            'explanations.npy'))
        if index < 0 or index >= len(explanations):
            print(f"Index {index} out of range [0-{len(explanations) - 1}]")
            return

        a_batch_test = explanations[index:index + 1]  # Shape: (1, H, W)
        x_single, y_single = self._dataset[index]

        # batch dimension:
        x_batch = x_single.unsqueeze(0).cpu().numpy()
        # batch of size 1
        y_batch = np.array([y_single])

        x = x_single
        explanation = a_batch_test[
            index] if a_batch_test.shape[0] > 1 else a_batch_test[0]
        self._visualize_sample(x, explanation, title_prefix=f"Sample {
            index}: ")

        scores = self._run_evaluation_irof(a_batch_test, n, x_batch, y_batch)

        if not len(scores) == 0:
            print(f"Score for index {index}: {scores[0]}")

    def aggregate_eval(self) -> None:
        """Aggregating all saved evaluation scores by
        mean, std, median, and variance."""
        if self._scores is None:
            try:
                self._scores = np.loadtxt(os.path.join(
                    self.folder, "mainCNN_eval_scores.csv"))
            except ValueError:
                raise ValueError(
                    "No scores found. Run evaluate_explanations() first.")
        self._scores = np.array(self._scores)
        print(f"=== Explainability Evaluation Stats ({
            len(self._scores)} samples): ===")
        print(f"Mean: {np.mean(self._scores)}")
        print(f"STD: {self._scores.std()}")
        print(f"Median: {np.median(self._scores)}")
        print(f"Variance: {np.var(self._scores)}")

    def overlay_image(self, index: int = 1) -> None:
        """
        Visualize an input image and its corresponding
        explanation/saliency map.
        Args:
            index: Index of preexisting image & saliency map. (From 1)
        """
        if not os.path.exists(self.folder_vis):
            print(f"{self.folder_vis} not found. "
                  f"Run gradcam_generate_explanations() first.")
            return

        if index < 1:
            raise ValueError("Index should be greater than 0.")

        original_img = self._dataset[index - 1][0].cpu().numpy()

        # Loaded as BGR by default:
        heatmap = cv2.imread(os.path.join(self.folder_vis, f'img_{index}.jpg'))
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB

        self._visualize_sample(
            original_img, heatmap, title_prefix=f"Sample {index}: ")

    def _get_test_dataset(self) -> None:
        """ Getting test dataset from DataPreprocessingPipeline. """
        if self._test_dataloader is None:
            data = DataPreprocessingPipeline(
                DetectionLabels, balance=True)
            self._test_dataloader = data.run()['test']
        self._dataset = Subset(self._test_dataloader.dataset,
                               indices=list(range(self.n_samples)))
        self._test_dataloader = DataLoader(self._dataset, batch_size=1)
        print(f"Test subset len: {len(self._dataset)}")

    def _calculate_mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates mean and standard deviation of the image channels.
        GradCam preprocessing.
        Returns:
             tuple[np.ndarray, np.ndarray]: mean and std arrays
        """
        psum = torch.tensor([0.0, 0.0, 0.0])
        psum_sq = torch.tensor([0.0, 0.0, 0.0])
        nb_samples = 0

        for inputs, _ in self._test_dataloader:
            # inputs shape: (batch_size, 3, height, width)
            psum += inputs.sum(dim=[0, 2, 3])  # Sum over batch, height, width
            psum_sq += (inputs ** 2).sum(dim=[0, 2, 3])
            # batch * height * width:
            nb_samples += inputs.shape[0] * inputs.shape[2] * inputs.shape[3]

        mean = psum / nb_samples
        var = (psum_sq / nb_samples) - (mean ** 2)
        std = torch.sqrt(var)

        mean_tensor = mean.view(-1, 1, 1)  # shape (3, 1, 1)
        std_tensor = std.view(-1, 1, 1)  # shape (3, 1, 1)

        return mean_tensor.cpu().numpy(), std_tensor.cpu().numpy()

    def _get_input_tensors(self) -> Tensor:
        """
        Gets input tensors. GradCam preprocessing.
        Returns:
             Tensor: a batch of input image tensors. (shape: (N, 3, H, W))
        """

        mean, stds = self._calculate_mean_std()
        input_tensors = []

        for img_tensor, _ in self._dataset:
            if img_tensor.shape[0] == 3:
                rgb_img = img_tensor.permute(
                    1, 2, 0).cpu().numpy()  # to (224, 224, 3)
            else:
                rgb_img = img_tensor.cpu().numpy()
            input_tensor = preprocess_image(rgb_img, mean=mean, std=stds)
            input_tensors.append(input_tensor)

        batch_tensor = torch.cat(input_tensors, dim=0)  # shape: (N, 3, H, W)
        return batch_tensor

    def _config_folders(self) -> tuple[str, str]:
        """
        Folder/File path handling.
        Returns:
             tuple[str, str]: Folder path and file path.
        """
        os.makedirs(self.folder, exist_ok=True)
        save_folder = os.path.join(self.folder, 'outputs')
        self.folder_vis = os.path.join(save_folder, 'visualizations')
        self.folder_explain = os.path.join(save_folder, 'explain')
        os.makedirs(self.folder_vis, exist_ok=True)
        os.makedirs(self.folder_explain, exist_ok=True)
        return self.folder_explain, self.folder_vis

    def _run_evaluation_irof(self,
                             a_batch_test: np.ndarray,
                             n: int,
                             x_batch: np.ndarray,
                             y_batch: np.ndarray) -> list:
        """
        Runs quantus IROF evaluation loop.
        Args:
            a_batch_test: explanations batch
            n: number of explanations to evaluate
            x_batch: array of x data
            y_batch: array of y data
        Returns:
            list: list of scores
        """
        wrapped_model = WrappedModel(self.model)
        metric = quantus.IROF(return_aggregate=False)
        scores = []
        for i in range(n):
            x = x_batch[i:i + 1]
            y = y_batch[i]
            a = a_batch_test[i:i + 1]

            # checking prediction confidence before evaluation
            probs = wrapped_model.predict(x)
            y_pred = probs[0, y]

            if y_pred < 0.01:
                print(f"Skipping index {i}: model confidence for "
                      f"true label {y} is too low ({y_pred:.6f})")
                continue

            score = metric(
                model=wrapped_model,
                channel_first=True,
                x_batch=x,
                y_batch=np.array([y]),
                a_batch=a,
                device=self.device.type,
                explain_func=quantus.explain,
                explain_func_kwargs={"method": "Saliency"}
            )[0]

            print(f"Score for index {i}: {score}")
            scores.append(score)
        return scores

    def _visualize_sample(self, x, explanation, title_prefix="") -> None:
        """
        Visualize an input image and its corresponding
        explanation/saliency map.

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
