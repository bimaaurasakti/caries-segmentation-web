from PIL import Image

import numpy as np
import cv2

def normalization(image, max, min):
    image_new = (image - np.min(image))*(max - min)/(np.max(image)-np.min(image)) + min
    return image_new

def clahe_equalized(image):
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = clahe.apply(grayimg)
    
    return imgs_equalized

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_image = cv2.LUT(image, table)
    
    return new_image

def make_square(input_image_array, min_size=224):
    im = Image.fromarray(input_image_array)
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    new_im = new_im.convert('RGB')

    # Convert the resulting image to a NumPy array
    result_array = np.array(new_im)

    return result_array

def make_square_to_np(im, min_size=256, fill_color=(0, 0, 0, 0)):
    im = Image.fromarray(im)
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return np.array(new_im, dtype=np.uint8)

def resize_image(input_np_array, desired_width=1024, desired_height=1024):
    input_image = Image.fromarray(input_np_array)
    resized_image = input_image.resize((desired_width, desired_height))
    return np.array(resized_image)