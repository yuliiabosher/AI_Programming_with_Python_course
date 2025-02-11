import torch
from torch import nn
from torch import optim

def build_network(model, train_data, learning_rate, hidden_units, device):
    num_classes = len(train_data.classes)    

    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(nn.Linear(25088, int(hidden_units)), 
                                     nn.ReLU(),                      
                                     nn.Dropout(0.2),
                                     nn.Linear(int(hidden_units), num_classes),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=float(learning_rate))

    model.to(device)
    return criterion, optimizer, model
