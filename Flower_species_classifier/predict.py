import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import get_model
from get_predict_args import get_predict_args
from load_data import load_data
from build_network import build_network
from process_image import process_image
from imshow import imshow
from predict_func import predict
from load_checkpoint import load_checkpoint

def main():
    args = get_predict_args()    
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    with open('arguments.json', 'r') as f:
        data = json.load(f)
        
    data_dir = data["data_dir"]
    arch = data["arch"]
    learning_rate = data["learning_rate"]
    hidden_units = data["hidden_units"]
                
    trainloader, validloader, testloader, train_data = load_data(data_dir)
    model = get_model(arch, weights="DEFAULT")
    criterion, optimizer, model = build_network(model, train_data, learning_rate, hidden_units, device)
    model = load_checkpoint(model, args.checkpoint)
    probs, classes = predict(args.path_to_image, model, args.top_k)
    probs = probs.cpu().numpy().flatten()  # Flatten to 1D array

    classes_flattened = classes.cpu().numpy().flatten() 
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    flower_names = [cat_to_name[str(cls+1)] for cls in classes_flattened]
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 6), nrows=2, ncols=1)
    image = process_image(args.path_to_image)
    imshow(image, ax=ax1)  
    ax1.set_title('Input Image')

    y_pos = np.arange(len(flower_names))
    ax2.barh(y_pos, probs, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(flower_names)
    ax2.invert_yaxis()  
    ax2.set_xlabel('Probability')
    ax2.set_title('Top 5 Predicted Classes')

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
