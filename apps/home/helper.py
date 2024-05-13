from torch.utils.data import Dataset
from PIL import Image, ImageChops
from .preprocessing import *

import torch
import numpy as np

def combine_image(input, pred): 
    input = input.transpose((1, 2, 0))
    pred = pred.transpose((1, 2, 0))
    combine_image = np.zeros((224, 224, 3), dtype=input.dtype)
    for ch in range(combine_image.shape[2]):
        for i in range(combine_image.shape[0]):
            for j in range(combine_image.shape[1]):
                if ch == 0 and pred[i,j,0] > 0:
                    combine_image[i,j,ch] = 1
                else:
                    combine_image[i,j,ch] = input[i,j,0]

    combine_image = (combine_image * 255).astype(np.uint8)
    return(combine_image)

def trim_from_np(im):
    im = Image.fromarray(im)
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        im = im.crop(bbox)
    return np.array(im, dtype=np.uint8)

class SimDataset(Dataset):
    def __init__(self, image):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        self.input_image = image
        
    def __len__(self):
        return 1
        
    def __getitem__(self, idx):
                
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        img_as_img = Image.open(self.input_image)
        img_as_np = np.asarray(img_as_img)

        # Normalize the image
        img_as_np = clahe_equalized (img_as_np)
        img_as_np = adjust_gamma(img_as_np, 1.2)
        img_as_np = normalization(img_as_np, max=1, min=0)
        img_as_np = np.expand_dims(img_as_np, axis=0)  # add additional dimension
        img_as_tensor = torch.from_numpy(img_as_np).float()  # Convert numpy array to tensor

        return (img_as_tensor)
    
class SimDatasetFullImage(Dataset):
    def __init__(self, image):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        self.input_images = image
        
    def __len__(self):
        return 1
        
    def __getitem__(self, idx):
        images = []
        images_as_tensor = []

        image = self.input_images
        image = Image.open(image)
        width, height = image.size
        half_width, half_height = width // 2, height // 2

        # Crop the four parts for image
        images.append(image.crop((0, 0, half_width, half_height)))
        images.append(image.crop((half_width, 0, width, half_height)))
        images.append(image.crop((0, half_height, half_width, height)))
        images.append(image.crop((half_width, half_height, width, height)))

        for img in images:
            img_as_np = np.asarray(img)

            # Normalize the image
            img_as_np = make_square_to_np(img_as_np)
            img_as_np = resize_image(img_as_np, 224, 224)
            img_as_np = clahe_equalized(img_as_np)
            img_as_np = adjust_gamma(img_as_np, 0.9)
            img_as_np = normalization(img_as_np, max=1, min=0)
            img_as_np = np.expand_dims(img_as_np, axis=0)
            images_as_tensor.append(torch.from_numpy(img_as_np).float())
        

        return images_as_tensor