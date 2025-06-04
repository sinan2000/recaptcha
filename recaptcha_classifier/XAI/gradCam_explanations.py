import os
import numpy as np
import torch
from matplotlib import pyplot as plt

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from PIL.Image import Image

from recaptcha_classifier import MainCNN
#from recaptcha_classifier.server.load_model import load_main_model

#model = load_main_model()

# target_layers = # [model.layers[-1]] last layer
# #input_tensor = # Create an input tensor image for your model...
# # Note: input_tensor can be a batch tensor with several images!

# # Construct the CAM object once, and then re-use it on many images:
# # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda)

# # You can also use it within a with statement, to make sure it is freed,
# # In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
# #   ...

# # We have to specify the target we want to generate
# # the Class Activation Maps for.
# # If targets is None, the highest scoring category
# # will be used for every image in the batch.
# # Here we use ClassifierOutputTarget, but you can define your own custom targets
# # That are, for example, combinations of categories, or specific outputs in a non standard model.

# targets = [ClassifierOutputTarget(281)]

# # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
# grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# # In this example grayscale_cam has only one image in the batch:
# grayscale_cam = grayscale_cam[0, :]
# visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)






















def gradcam_explain(model,
                    target_layers,
                    folder: str = '',
                    input_tensors = []):

    os.makedirs(folder, exist_ok=True)

    save_folder = os.path.join(folder, 'outputs')
    folder_vis = os.path.join(save_folder, 'visualizations')
    folder_explain = os.path.join(save_folder, 'explain')

    os.makedirs(folder_vis, exist_ok=True)
    os.makedirs(folder_explain, exist_ok=True)

    explanations = []

    model.eval()

    device = next(model.parameters()).device

    with GradCAM(model=model, target_layers=target_layers, use_cuda=device.type == 'cuda') as cam:
        for i, tensor in enumerate(input_tensors):
            if (i + 1) % 25 == 0:
                print(f'Processing image {i + 1}/{len(input_tensors)}')

            tensor = tensor.to(device)
            input_tensor = tensor.unsqueeze(0)  # Add batch dimension

            # Normalize for visualization (between 0 and 1)
            img = tensor.permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())

            # Automatically use top predicted class
            outputs = model(input_tensor)
            pred_class = outputs.argmax(dim=1).item()
            targets = [ClassifierOutputTarget(pred_class)]

            # Generate CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            explanations.append(grayscale_cam)

            # Overlay CAM on image
            visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

            # Save visualization
            vis_img = Image.fromarray(visualization)
            vis_img.save(os.path.join(folder_vis, f'img_{i}.jpg'))

    # Save all raw explanations
    np.save(os.path.join(folder_explain, 'explanations.npy'), np.asarray(explanations))
    print(f"Saved {len(explanations)} GradCAM visualizations to {folder_vis}")

    # for i, tensor in enumerate(input_tensors):

    #     if (i + 1) % 100 == 0:                               #############
    #         print(f'Processing image {i + 1}')
    #         print(np.asarray(explanations).shape)
    #     # generate img as above
    #     img = tensor.permute(1, 2, 0).numpy()
    #     #img = np.repeat(img, 3, axis=-1) # to fit rgb format

    #     with (EigenCAM(model=model, target_layers=target_layers) as cam): # change EigenCAM to desired cams
    #         grayscale_cam = cam(input_tensor=tensor.unsqueeze(0))
    #         explanations.append(grayscale_cam)
    #         grayscale_cam = grayscale_cam[0, :]

    #         visualization = show_cam_on_image(img, grayscale_cam, use_rgb=False)

    #         img = Image.fromarray(visualization)
    #         file_name = str(f'img_{i}.jpg')
    #         img.save(os.path.join(folder_vis, file_name))
    #         # model_outputs = cam.outputs

    # np.save(os.path.join(folder_explain, 'explanations.npy'), np.asarray(explanations))