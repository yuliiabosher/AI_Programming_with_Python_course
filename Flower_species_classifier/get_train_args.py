import argparse

def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', default='/home/yulia/Desktop/')
    parser.add_argument('--arch', default='vgg16')
    parser.add_argument('--learning_rate', default='0.003')
    parser.add_argument('--hidden_units', default='512')
    parser.add_argument('--epochs', default='5')
    parser.add_argument('--gpu', action='store_true')
    
    return parser.parse_args()
