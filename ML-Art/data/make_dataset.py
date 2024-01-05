import os
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import random
import click
import pandas as pd




class ArtDataset(Dataset):
    def __init__(self, root_dir, selected_styles, num_images_per_style=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and subdirectories for art styles.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        ## TODO : Add if train == True/False, make sure training images are not chosen in testing set or else Data Leak
        self.root_dir = root_dir
        self.transform = transform
        # self.images = torch.empty
        # self.labels = []
        self.style_counts = {}

        # Tensor Transform PIL -> Tensor

        tensor_transform = transforms.Compose([transforms.ToTensor()])
        ## TODO Pick Appropriate Size or Filter Unwanted Resolution
        resize_transform = transforms.Resize((500,500))

        # Load images and labels
        for i,style in enumerate(selected_styles):
            style_dir = os.path.join(root_dir, style, style)
            if os.path.isdir(style_dir):
                all = os.listdir(style_dir)
                cnt = len(all)
                self.style_counts[style] = cnt
                # Randomly select num_images_per_style images
                if num_images_per_style is None or num_images_per_style > cnt:
                    selected_images = all
                else:
                    selected_images = random.sample(all, 
                                                    min(num_images_per_style, cnt))
                for j,img in enumerate(selected_images):
                    img_path = os.path.join(style_dir, img)
                    image = Image.open(img_path).convert("RGB")
                    resized_img = resize_transform(image)
                    tensor_img = tensor_transform(resized_img)

                    # Define Tensor on First Iteration
                    if (i == 0) and (j == 0):   # If you know a better way let me know, I hate this
                        self.images = tensor_img.unsqueeze(0)
                        self.labels = torch.tensor(i).unsqueeze(0)
                    # Then concatenate
                    else:
                        self.images = torch.cat((self.images,tensor_img.unsqueeze(0)),dim = 0)
                        self.labels = torch.cat((self.labels,torch.tensor(i).unsqueeze(0)),dim = 0)

        if self.transform:
            self.images = transform(self.images)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx],self.labels[idx]

    
    def get_random_image_by_style(self, style):
        style_dir = os.path.join(self.root_dir, style, style)
        if os.path.isdir(style_dir):
            images = os.listdir(style_dir)
            if images:
                random_image = random.choice(images)
                return os.path.join(style_dir, random_image)
        return None


@click.command()
@click.option("--raw_data_path", default="data/raw", help="Path To Raw Data")
@click.option("--processed_data_path", default="data/processed", help="Path To Raw Data")
def main(raw_data_path,processed_data_path):

    # Selected Styles

    styles = ["Academic_Art", "Realism", "Symbolism"]


    dataset = ArtDataset(root_dir = raw_data_path,
                        selected_styles = styles,
                        num_images_per_style = 64,
                        transform = None)
    
    torch.save(dataset,"data/processed/dataset.pt")
    
    
    # I'm thinking to save dataset and load it for reproducibility
    # Or else everytime the images are selected randomly ....

    dataset = torch.load("data/processed/dataset.pt")

    dataloader = DataLoader(dataset,batch_size = 64, shuffle = True)

    data_iter = iter(dataloader)

    img,target = next(data_iter)

    print("Images Shape: ",img.shape)
    print("Target Shape: ",target.shape)





if __name__ == '__main__':
    # Get the data and process it
    main()