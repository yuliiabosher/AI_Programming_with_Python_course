import torch
from process_image import process_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''   
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    
    # Process the image
    image = process_image(image_path)
    image = image.unsqueeze(0)
    
    # Move the tensor to the same device as the model (if using GPU)
    if torch.cuda.is_available():
        image = image.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        ps = torch.exp(model(image)) 
        top_p, top_classes = ps.topk(int(topk), dim=1)
    return top_p, top_classes   
