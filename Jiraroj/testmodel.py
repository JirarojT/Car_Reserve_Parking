from ultralytics import YOLO

# Load best model
model = YOLO('/Users/pennywise/Documents/python/project/besttext.pt')

# Test on an image
results = model('/Users/pennywise/Documents/python/project/plate_0.jpg', conf=0.25)
results[0].show()
