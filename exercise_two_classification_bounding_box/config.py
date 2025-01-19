import torch

data_dir = '../data/kagglehub/datasets/andrewmvd/dog-and-cat-detection/versions/1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')