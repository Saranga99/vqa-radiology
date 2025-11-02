from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import matplotlib.pyplot as plt

def visualize_cam(model, input_tensor, target_layers):
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    rgb_img = np.transpose(input_tensor.squeeze().cpu().numpy(), (1, 2, 0))
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    plt.imshow(cam_image)
    plt.axis('off')
    plt.show()