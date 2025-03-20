import torch
from ultralytics.nn.tasks import DetectionModel

# Add DetectionModel to the safe globals
torch.serialization.add_safe_globals([DetectionModel])

# Now load the model
model = torch.load('/Users/pennywise/Documents/python/project/best.pt')
print(model)
# The model is now loaded successfully