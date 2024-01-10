import os
import random
import hydra
import torch

import torchvision.transforms as transforms

from torch.utils.data import Dataset,Subset
from PIL import Image, UnidentifiedImageError
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv



# Here we can change the input size of given images
def pad_and_resize(img, target_size=(512, 512)):
    # Resize if any of the dimensions are greater than target size
    if img.size[0] > target_size[0] or img.size[1] > target_size[1]:
        # Scale down yet keeping the aspect ratio
        img.thumbnail(target_size, Image.Resampling.LANCZOS)

    # Calculate padding
    padding_l = (target_size[0] - img.size[0]) // 2
    padding_t = (target_size[1] - img.size[1]) // 2
    padding_r = target_size[0] - img.size[0] - padding_l
    padding_b = target_size[1] - img.size[1] - padding_t
    paddings = (padding_l, padding_t, padding_r, padding_b)

    return transforms.functional.pad(img, paddings, padding_mode="constant", fill=0)


class ArtDataset(Dataset):
    def __init__(
        self, root_dir, selected_styles, num_images_per_style=None, transform=None
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
        self.style_to_idx = {style: idx for idx,
                             style in enumerate(selected_styles)}

        # Load images and labels
        for style in selected_styles:
            style_dir = os.path.join(root_dir, style, style)
            if os.path.isdir(style_dir):
                all = os.listdir(style_dir)
                cnt = len(all)
                self.style_counts[style] = cnt
                # Randomly select num_images_per_style images
                if num_images_per_style is None or num_images_per_style > cnt:
                    selected_images = all
                else:
                    selected_images = random.sample(
                        all, min(num_images_per_style, cnt))
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
                f"Warning: Skipping file {img_name} as it couldn't be identified.")
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
@hydra.main(config_path="config", config_name="data_config.yaml",version_base="1.1")
def main(config):
    # Load Global Env Variables defined in .env
    load_dotenv()

    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    data_cfg = config.dataset

    # Ensure Reproducibility
    torch.manual_seed(data_cfg.seed)
    random.seed(data_cfg.seed)


    # Transformations
    transform = transforms.Compose(
        [
            transforms.Lambda(pad_and_resize),
            transforms.ToTensor()
        ]
    )

    # Create the dataset
    dataset = ArtDataset(
        root_dir=os.path.join(os.getenv("LOCAL_PATH"),data_cfg.path),
        selected_styles=data_cfg.styles,
        num_images_per_style=data_cfg.imgs_per_style,
        transform=transform,
    )


    # Split the dataset into training and test sets
    train_idx, test_idx = train_test_split(
        list(range(len(dataset))), test_size=data_cfg.test_size, random_state=data_cfg.seed
    )

    # Create subset for training and test from the indices
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    torch.save(train_dataset,"data/processed/train_set.pt")
    torch.save(test_dataset,"data/processed/test_set.pt")



if __name__ == "__main__":
    main()
