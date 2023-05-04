import os
import time
import warnings

import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from skimage import color
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import kornia.color as kc
from colorization.colorizers.siggraph17 import siggraph17

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


# Define a function to preprocess an image
def preprocess_img(img):
    # Reverse the order of the color channels from BGR to RGB using PyTorch's flip function
    img = torch.flip(img, dims=[-1])
    # Resize the image to (256, 256) using PyTorch's Resize transform
    img = transforms.Resize((256, 256), antialias=True)(img)
    # Convert the RGB image to Lab color space using scikit-image's rgb2lab function
    img_lab = color.rgb2lab(img.numpy().transpose(1, 2, 0))
    # Extract the L channel (luminance) from the Lab image using NumPy's slicing syntax
    img_l = img_lab[:, :, 0]
    # Extract the ab channels (color) from the Lab image using NumPy's slicing syntax
    img_ab = img_lab[:, :, 1:]
    # Convert the L channel to a PyTorch tensor, add a batch dimension using unsqueeze, and convert the data type to float
    img_l = torch.from_numpy(img_l).unsqueeze(0).float()
    # Convert the ab channels to a PyTorch tensor, transpose the dimensions to match the expected shape, and convert the data type to float
    img_ab = torch.from_numpy(img_ab).transpose(2, 0).transpose(2, 1).float()
    # Return the L and ab tensors as a tuple, ready for use in a neural network model
    return img_l, img_ab


# Define a function to load an image
def load_img(img_path):
    # Open the image at the specified path using PIL's Image module, convert it to a PyTorch tensor, and store it in out_tensor
    out_tensor = transforms.ToTensor()(Image.open(img_path))
    # If the tensor is 2D (grayscale), repeat the 2D tensor along the third axis (RGB), and store the result in out_tensor
    if out_tensor.ndim == 2:
        out_tensor = torch.stack([out_tensor] * 3, dim=0)
    # Return the tensor, which is now RGB (3D) even if it was originally grayscale (2D)
    return out_tensor


# Define a custom dataset class for colorization
class CustomDataset(torch.utils.data.Dataset):
    # Initialize the dataset with the file containing the train/test split, the data type (train or test)
    def __init__(self, split_file, data_type):
        # Get a list of image file paths for the specified data type
        self.image_files = self.get_image_files(split_file, data_type)

    # Reads the split file and returns a list of image file paths for the specified data type (train or test)
    def get_image_files(self, split_file, data_type):
        image_files = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                file_data_type, file_path = line.strip().split()
                # If the data type matches the specified type (train or test)
                if file_data_type == data_type:
                    image_files.append(file_path)
        return image_files

    # Get an image and its corresponding ab channels by index
    def __getitem__(self, idx):
        # Load the image at the specified index using the load_img function and the corresponding file path from the list of image file paths
        img_file_name = os.path.join(self.image_files[idx])
        img = load_img(img_file_name)
        # Preprocess the image using the preprocess_img function to get the L and ab channels
        img_l, img_ab = preprocess_img(img)
        # Return the preprocessed L and ab tensors as a tuple
        return img_l, img_ab, img_file_name

    # Return the length of the list of image file paths, which corresponds to the number of images in the dataset
    def __len__(self):
        return len(self.image_files)


def clip_lab_values(lab_image):
    # Separate the L, A, and B channels from the input LAB image tensor
    l, a, b = lab_image[:, 0, :, :], lab_image[:, 1, :, :], lab_image[:, 2, :, :]
    # Clamp the L channel values to the range [0, 100]
    l = torch.clamp(l, 0, 100)
    # Clamp the A channel values to the range [-128, 127]
    a = torch.clamp(a, -128, 127)
    # Clamp the B channel values to the range [-128, 127]
    b = torch.clamp(b, -128, 127)
    # Stack the clamped L, A, and B channels back together along the channel dimension
    return torch.stack([l, a, b], dim=1)


def convert_lab_to_rgb(img_l, img_ab):
    img_lab = torch.cat((img_l, img_ab), dim=1)
    img_lab_np = img_lab.cpu().detach().numpy()
    img_lab_np = img_lab_np.transpose(
        0, 2, 3, 1
    )  # Change order of dimensions: CxHxW to HxWxC
    img_rgb = []
    for lab in img_lab_np:
        img_rgb.append(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2BGR))
    img_rgb = np.stack(img_rgb)
    img_rgb = img_rgb.transpose(
        0, 3, 1, 2
    )  # Change order of dimensions: HxWxC to CxHxW
    return img_rgb


def save_images(img, original, grayscale, colorized):
    # Normalize grayscale image to range [0, 1]
    grayscale = grayscale / 255.0

    # Create a new figure with a specified size
    plt.figure(figsize=(12, 8))

    # Display the original image in the first subplot
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    # Display the grayscale image in the second subplot
    plt.subplot(1, 3, 2)
    plt.imshow(grayscale, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    # Display the colorized image in the third subplot
    plt.subplot(1, 3, 3)
    plt.imshow(colorized)
    plt.title("Colorized (SIGGRPAH 17)")
    plt.axis("off")

    # Create a new array for the grayscale image with three channels
    grayscale_rgb = np.stack((grayscale,) * 3, axis=-1)

    # Concatenate the original, grayscale, and colorized images horizontally
    combined = np.concatenate((original, grayscale_rgb, colorized), axis=1)

    # Normalize the combined image by clipping values between 0 and 1
    combined_normalized = np.clip(combined, 0, 1)

    # Create a folder to save the images if it doesn't exist
    folder_path = os.path.join(os.getcwd(), f"siggraph_trained_flowers")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Determine the save path for the combined image
    save_path = os.path.join(folder_path, img[0].split(os.sep)[-1])

    # Save the combined image to the specified path
    plt.imsave(save_path, combined_normalized)

    # Close the current figure to avoid memory issues
    plt.close()


# Training function
def train_siggraph17(model, train_loader, device, num_epochs=100, lr=1e-4):
    model.train()  # Set model to training mode
    model.to(device)  # Move the model to the specified device (GPU or CPU)

    train_psnr = []
    train_ssim = []
    train_lpips = []

    # Define the optimizer (Adam) with the specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define the loss function (Mean Squared Error)
    criterion = nn.MSELoss()

    # Create a SummaryWriter object
    writer = SummaryWriter(log_dir="runs/siggraph17_flowers")

    for epoch in range(num_epochs):
        # Wrap the train_loader with tqdm to display a progress bar
        progress_bar = tqdm(
            enumerate(train_loader), total=len(train_loader), leave=True
        )

        # Move input tensors to the device
        for i, (img_l, img_ab, img_file_name) in progress_bar:
            img_l, img_ab = img_l.to(device), img_ab.to(device)

            # Clear the gradients of the model parameters
            optimizer.zero_grad()

            # Forward pass: compute the predicted ab channels
            img_ab_pred = model(img_l)

            # Compute the loss between the predicted ab channels and the ground truth ab channels
            loss = criterion(img_ab_pred, img_ab)
            # Backward pass: compute gradients for the model parameters
            loss.backward()
            # Update model parameters using the computed gradients
            optimizer.step()

            # Convert the predicted ab channels and the ground truth ab channels to RGB
            img_rgb_pred = convert_lab_to_rgb(img_l, img_ab_pred)
            img_rgb = convert_lab_to_rgb(img_l, img_ab)

            # Calculate the PSNR between the ground truth and predicted RGB images
            cur_psnr = psnr(img_rgb, img_rgb_pred)
            # Calculate the SSIM between the ground truth and predicted RGB images
            try:
                cur_ssim, _ = ssim(
                    img_rgb, img_rgb_pred, channel_axis=1, full=True, data_range=1
                )
            except ValueError:
                pass

            # Combine the lightness channel (img_l) with the ground truth ab channels to create an Lab image
            img_lab = torch.cat((img_l, img_ab), dim=1)
            # Combine the lightness channel (img_l) with the predicted ab channels to create a predicted Lab image
            img_lab_pred = torch.cat((img_l, img_ab_pred), dim=1)
            # Clip the predicted Lab image values to be within the valid Lab color space range
            img_lab_pred_clipped = clip_lab_values(img_lab_pred)
            # Convert the clipped predicted Lab image to an RGB image
            img_rgb_pred = kc.lab_to_rgb(img_lab_pred_clipped)
            # Convert the ground truth Lab image to an RGB image
            img_rgb = kc.lab_to_rgb(img_lab)
            # Calculate the LPIPS loss between the ground truth and predicted RGB images
            cur_lpips = lpips_loss(img_rgb, img_rgb_pred).mean().item()

            train_psnr.append(cur_psnr)
            train_ssim.append(cur_ssim)
            train_lpips.append(cur_lpips)

            # Update the progress bar with the current loss value
            progress_bar.set_description(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, PSNR: {cur_psnr:.2f}, SSIM: {cur_ssim:.4f}, LPIPS: {cur_lpips:.4f}",
                refresh=True,
            )

            # Move the images back to the CPU
            img_rgb, img_rgb_pred, img_l = (
                img_rgb.cpu().detach().numpy(),
                img_rgb_pred.cpu().detach().numpy(),
                img_l.cpu().detach().numpy(),
            )

            # Save the images
            save_images(
                img_file_name,
                img_rgb[0].transpose(1, 2, 0),
                img_l[0][0],
                img_rgb_pred[0].transpose(1, 2, 0),
            )

            # Log the PSNR, SSIM and LPIPS to TensorBoard
            writer.add_scalar("PSNR/train", cur_psnr, epoch * len(train_loader) + i)
            writer.add_scalar("SSIM/train", cur_ssim, epoch * len(train_loader) + i)
            writer.add_scalar("LPIPS/train", cur_lpips, epoch * len(train_loader) + i)

            # Log the loss value to TensorBoard
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)

            # # Log the learning rate to TensorBoard
            # writer.add_scalar("Learning Rate", lr, epoch)

        if epoch % 10 == 0:
            # Save the model checkpoint
            torch.save(model.state_dict(), f"siggraph17_flowers_{epoch}.pth")

    # Close the SummaryWriter object
    writer.close()

    # Save training metrics summary to a text file
    with open("siggraph_flowers_train.txt", "w") as f:
        f.write(f"GPU: {torch.cuda.get_device_name()}\n")
        f.write(f"Average PSNR: {sum(train_psnr)/len(train_psnr):.4f}\n")
        f.write(f"Average SSIM: {sum(train_ssim)/len(train_ssim):.4f}\n")
        f.write(f"Average LPIPS: {sum(train_lpips)/len(train_lpips):.4f}\n")


if __name__ == "__main__":
    start_time = time.time()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using {torch.cuda.get_device_name()} GPU")
    else:
        print("Using CPU")

    lpips_loss = lpips.LPIPS(net="alex").to(device)

    # Define the training data path and create the custom dataset
    train_dataset = CustomDataset("./oxford_flowers/train_test_split.txt", "train")

    # Create the DataLoader
    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Instantiate the model
    model = siggraph17(pretrained=True)

    # Train the model
    train_siggraph17(model, train_loader, device, num_epochs=100)

    # Save the trained model
    torch.save(model.state_dict(), "siggraph17_flowers_trained.pth")

    end_time = time.time()
    training_time = end_time - start_time

    # Save training summary to a text file
    with open(
        f"siggraph_flowers_train.txt",
        "a",
    ) as f:
        f.write("\nTraining Summary\n")
        f.write(
            f"Total training time: {training_time // 60} minutes {training_time % 60:.3f} seconds\n"
        )
