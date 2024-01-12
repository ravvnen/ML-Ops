import os
import random
import hydra
import torch
import logging

from omegaconf import OmegaConf
from torchvision import transforms
from torch.utils.data import Dataset, Subset
from PIL import Image, UnidentifiedImageError, ImageOps
from sklearn.model_selection import train_test_split


class PadAndResize:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        # Resize if any of the dimensions are greater than target size
        if (
            img.size[0] > self.target_size[0] or img.size[1] > self.target_size[1]
        ):
            img.thumbnail(self.target_size, Image.Resampling.LANCZOS)

        # Calculate padding
        padding_l = (self.target_size[0] - img.size[0]) // 2
        padding_t = (self.target_size[1] - img.size[1]) // 2
        padding_r = self.target_size[0] - img.size[0] - padding_l
        padding_b = self.target_size[1] - img.size[1] - padding_t
        paddings = (padding_l, padding_t, padding_r, padding_b)

        # Add padding
        img = ImageOps.expand(img, border=paddings, fill=0)
        return img


class WikiArt(Dataset):
    def __init__(
        self,
        root_dir,
        selected_styles,
        num_images_per_style=None,
        transform=None,
    ):
        """
        Args:
            root_dir (string): Directory with all the images and subdirectories for art styles.
            selected_styles (list): List of art styles that need to be classified.
            num_images_per_style (int, optional): Number of images from each style.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.style_counts = {}
        self.style_to_idx = {
            style: idx for idx, style in enumerate(selected_styles)
        }

        # Load images and labels
        for style in selected_styles:
            style_dir = os.path.join(root_dir, style, style)
            assert os.path.isdir(style_dir)

            if os.path.isdir(style_dir):
                all = os.listdir(style_dir)
                cnt = len(all)
                self.style_counts[style] = cnt
                # Randomly select num_images_per_style images
                if num_images_per_style is None or num_images_per_style > cnt:
                    selected_images = all
                else:
                    selected_images = random.sample(
                        all, min(num_images_per_style, cnt)
                    )
                for img in selected_images:
                    img_path = os.path.join(style_dir, img)
                    self.images.append(img_path)
                    self.labels.append(style)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        try:
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except UnidentifiedImageError:
            print(
                f"Warning: Skipping file {img_name} as it couldn't be identified."
            )
            # Return the next image
            return self.__getitem__((idx + 1) % self.__len__())

        # Convert label to index
        label_idx = self.style_to_idx[self.labels[idx]]
        return image, label_idx

    def get_random_image_by_style(self, style):
        style_dir = os.path.join(self.root_dir, style, style)
        if os.path.isdir(style_dir):
            images = os.listdir(style_dir)
            if images:
                random_image = random.choice(images)
                return os.path.join(style_dir, random_image)
        return None


print(os.getcwd())


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.1"
)
def main(config):
    # Init Logger - Hydra sets log dirs to outputs/ by default
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info(f"configuration: \n {OmegaConf.to_yaml(config)}")
    data_cfg = config.dataset

    # Ensure Reproducibility
    torch.manual_seed(data_cfg.seed)
    random.seed(data_cfg.seed)

    resize_target = (data_cfg.input_shape[1], data_cfg.input_shape[2])

    # Error due to the pad_resize function in trasnforms.Lambda -> Temporary Fix Below
    #     Traceback (most recent call last):
    #   File "c:\Users\Hasan\OneDrive\Desktop\Projects\ML-Ops\ml_art\data\make_dataset.py", line 157, in main
    #     torch.save(train_dataset,os.path.join(data_cfg.processed_path,"train_set.pt"))
    #   File "C:\Users\Hasan\miniconda3\envs\ML-Art\Lib\site-packages\torch\serialization.py", line 619, in save
    #     _save(obj, opened_zipfile, pickle_module, pickle_protocol, _disable_byteorder_record)
    #   File "C:\Users\Hasan\miniconda3\envs\ML-Art\Lib\site-packages\torch\serialization.py", line 831, in _save
    #     pickler.dump(obj)
    # AttributeError: Can't pickle local object 'main.<locals>.<lambda>'

    # Transformations

    if data_cfg.input_shape[0] not in [1, 3]:  # RGB or Grayscale
        raise ValueError("Use RGB or GrayScale Images")

    # transform = transforms.Compose(
    #     [transforms.Resize((resize_target)), transforms.ToTensor()]
    # )
    transform = transforms.Compose(
        [PadAndResize(resize_target), transforms.ToTensor()]
    )

    if data_cfg.input_shape[0] == 1:  # Grayscale
        transform = transforms.Compose(
            [transform, transforms.Grayscale(num_output_channels=1)]
        )

    print("Selected Styles: ", data_cfg.styles)

    # Create the dataset
    dataset = WikiArt(
        root_dir=data_cfg.raw_path,
        selected_styles=data_cfg.styles,
        num_images_per_style=data_cfg.imgs_per_style,
        transform=transform,
    )

    # Split the dataset into training and test sets
    train_idx, test_idx = train_test_split(
        list(range(len(dataset))),
        test_size=data_cfg.test_size,
        random_state=data_cfg.seed,
    )

    # Create subset for training and test from the indices
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    hydra_log_dir = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    torch.save(train_dataset, os.path.join(hydra_log_dir, "train_set.pt"))
    torch.save(test_dataset, os.path.join(hydra_log_dir, "test_set.pt"))

    logger.info(
        f"Processed raw data into a .pt file stored in {hydra_log_dir}"
    )


if __name__ == "__main__":
    # Get the data and process it
    main()
