import torch
from colorization.colorizers.eccv16 import eccv16
from colorization.colorizers.siggraph17 import siggraph17
from colorization.colorizers.util import *
import matplotlib.pyplot as plt


def visualize_images(original, grayscale, colorized):
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(grayscale)
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(colorized)
    plt.title("Colorized (ECCV 16)")
    plt.axis("off")
    plt.show()


def colorize_image(img_path, model, device):
    img_color = load_img(img_path)
    tens_l_orig, tens_l_rs = preprocess_img(img_color)
    tens_l_orig, tens_l_rs = tens_l_orig.to(device), tens_l_rs.to(device)
    img_gray = postprocess_tens(
        tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1)
    )
    img_rgb_pred = postprocess_tens(tens_l_orig, model(tens_l_rs))
    visualize_images(img_color, img_gray, img_rgb_pred)


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model and load the trained weights
    model = eccv16(pretrained=False)
    model.load_state_dict(torch.load("eccv16_flowers_90.pth"))
    model.eval()
    model.to(device)

    # Define the image path you want to colorize
    img_path = (
        r"/home/jamal.aljaberi/Desktop/ai702_project/oxford_flowers/jpg/image_03838.jpg"
    )

    # Colorize and visualize the image
    colorize_image(img_path, model, device)
