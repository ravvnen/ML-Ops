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
        ## TODO : Add if train == True/False, make sure training images are not chosen in testing set or else data leak
        resize_h = 500
        resize_w = 500
        self.root_dir = root_dir
        self.transform = transform
        self.images = torch.empty((num_images_per_style * len(selected_styles),3,resize_h, resize_w))
        self.labels = torch.empty((num_images_per_style * len(selected_styles)))
        self.style_counts = {}

        # Tensor Transform PIL -> Tensor

        ## TODO Pick Appropriate Size or Filter Unwanted Resolution
        auto_transform = transforms.Compose([transforms.Resize((resize_h,resize_w)),transforms.ToTensor()])
        
        # Load images and labels
        for j,style in enumerate(selected_styles):
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
                for i,img in enumerate(selected_images):
                    img_path = os.path.join(style_dir, img)
                    image = Image.open(img_path).convert("RGB")
                   
                    tensor_img = auto_transform(image)

                    # Assign Image & Label to Tensor

                    self.images[i + (j * num_images_per_style),:,:,:] = tensor_img
                    self.labels[i + (j * num_images_per_style)] = j

        if self.transform:
            self.images = self.transform(self.images)

        print("Dataset Images Shape: ",self.images.shape)
        print("Dataset Targets Shape: ",self.labels.shape)


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
                        num_images_per_style = 10,
                        transform = None)
    
    torch.save(dataset,"data/processed/dataset.pt")
    
    
    # I'm thinking to save dataset and load it for reproducibility
    # Or else everytime the images are selected randomly ....

    dataset = torch.load("data/processed/dataset.pt")

    dataloader = DataLoader(dataset,batch_size = 10, shuffle = True)

    data_iter = iter(dataloader)

    img,target = next(data_iter)

    print("Images Batch Shape: ",img.shape)
    print("Target Batch Shape: ",target.shape)




if __name__ == '__main__':
    # Get the data and process it
    main()