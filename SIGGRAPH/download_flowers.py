import os
import urllib.request
import tarfile
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract_dataset(url, dest_dir):
    # Create the destination directory if it does not exist
    os.makedirs(dest_dir, exist_ok=True)

    # Download the dataset
    tar_filename = os.path.join(dest_dir, "oxford_flowers.tar")

    print("Downloading the dataset...")

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=url.split("/")[-1],
    ) as t:
        urllib.request.urlretrieve(url, tar_filename, reporthook=t.update_to)

    print("Extracting the dataset...")

    # Extract the dataset
    with tarfile.open(tar_filename, "r") as tar_file:
        tar_file.extractall(dest_dir)

    # Remove the tar file
    os.remove(tar_filename)


def save_train_test_split(train_files, test_files, output_file):
    with open(output_file, "w") as f:
        for file_path in train_files:
            f.write("train " + file_path + "\n")
        for file_path in test_files:
            f.write("test " + file_path + "\n")


if __name__ == "__main__":
    dataset_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    dest_dir = "./oxford_flowers"

    # Download and extract the dataset
    download_and_extract_dataset(dataset_url, dest_dir)

    # Get all image file paths
    all_image_files = []
    for root, _, files in os.walk(dest_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                all_image_files.append(os.path.join(root, file))

    # Split the data into train and test sets
    random.seed(42)
    train_files, test_files = train_test_split(
        all_image_files, test_size=0.2, random_state=42
    )

    # Save the train/test split
    save_train_test_split(train_files, test_files, "train_test_split.txt")
