import torch

# Load the model
model_path = '/Users/pennywise/Documents/python/project/best.pt'
model = torch.load(model_path, map_location='cpu')

# Print the entire content (careful — might be large!)
print(model)
