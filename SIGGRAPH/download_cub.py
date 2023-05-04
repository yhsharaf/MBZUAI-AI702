import urllib.request
import tarfile
from tqdm import tqdm


def download_file_with_progress_bar(url, local_filename):
    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
        with open(local_filename, "wb") as file:
            while True:
                data = response.read(block_size)
                if not data:
                    break
                progress_bar.update(len(data))
                file.write(data)

        progress_bar.close()
        if total_size != 0 and progress_bar.n != total_size:
            print("ERROR: Download failed.")
            return False
        return True


def extract_tgz_file(file_path, output_path):
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=output_path)


if __name__ == "__main__":
    url = (
        "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
    )
    file_name = "CUB_200_2011.tgz"

    print("Downloading CUB dataset...")
    if download_file_with_progress_bar(url, file_name):
        print("Download complete.")
        print("Extracting CUB dataset...")
        extract_tgz_file(file_name, ".")
        print("Extraction complete.")
    else:
        print("Download failed.")
