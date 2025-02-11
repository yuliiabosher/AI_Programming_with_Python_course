import torch
import json
import os
from torchvision.models import get_model
from get_train_args import get_train_args
from load_data import load_data
from build_network import build_network
from train_network import train_network
from test_network import test_network

def main():
    args = get_train_args()
    with open('arguments.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    trainloader, validloader, testloader, train_data = load_data(args.data_dir)
    model = get_model(args.arch, weights="DEFAULT")
    criterion, optimizer, model = build_network(model, train_data, args.learning_rate, args.hidden_units, device)
    model = train_network(trainloader, validloader, optimizer, criterion, model, args.epochs, device)
    test_network(trainloader, testloader, optimizer, criterion, model, device)
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'checkpoint.pth'))

if __name__ == "__main__":
    main()
