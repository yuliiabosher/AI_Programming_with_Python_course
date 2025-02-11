import torch
import numpy as np
from PIL import Image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Load the image
    pil_image = Image.open(image)
    
    # Resize the image with the shortest side being 256 pixels
    pil_image.thumbnail((256, 256))
    
    # Crop the center 224x224 portion of the image
    width, height = pil_image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    pil_image = pil_image.crop((left, top, right, bottom))
    
    # Convert the image to a numpy array and normalize
    np_image = np.array(pil_image) / 255.0  # Convert to float in range 0-1
    
    # Normalize the image
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds
    
    # Rearrange dimensions to have color channel first
    np_image = np_image.transpose((2, 0, 1))  # Change from HxWxC to CxHxW
    
    tensor_image = torch.from_numpy(np_image).float()
    
    return tensor_image
