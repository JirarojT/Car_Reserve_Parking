import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Thai character set for ก-ฮ and numbers
thai_characters = [
    'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 
    'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ฤ', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'ฮ',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

# Path to store generated images and labels
output_dir = 'generated_dataset'
os.makedirs(output_dir, exist_ok=True)

# Font path (change to a valid Thai font path on your system)
font_path = "/Library/Fonts/Thonburi.ttc"  # For Thai font in macOS (you can choose any Thai font installed)
font_size = 64  # Font size for the characters

# Function to generate a single image of a character
def generate_image(character):
    # Create a white canvas
    image = Image.new('RGB', (128, 128), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    try:
        # Load Thai font
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        raise Exception("Font file not found. Make sure the path is correct.")
    
    # Get the bounding box of the text (replaces textsize method)
    bbox = draw.textbbox((0, 0), character, font=font)
    text_width = bbox[2] - bbox[0]  # width = x2 - x1
    text_height = bbox[3] - bbox[1]  # height = y2 - y1
    
    # Draw the character on the image
    text_position = ((128 - text_width) // 2, (128 - text_height) // 2)  # Center the text
    draw.text(text_position, character, font=font, fill=(0, 0, 0))
    
    # Convert to numpy array (OpenCV uses BGR format)
    image = np.array(image)
    
    return image

# Function to save dataset and labels
def save_dataset(num_samples):
    labels = []
    for i in range(num_samples):
        # Randomly pick a character from the set
        char = random.choice(thai_characters)
        
        # Find the index of the character in the thai_characters list
        char_index = thai_characters.index(char)
        
        # Generate the image
        image = generate_image(char)
        
        # Save the image to disk
        image_filename = os.path.join(output_dir, f"{i}.jpg")
        cv2.imwrite(image_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for saving
        
        # Save the label (class index)
        label_filename = os.path.join(output_dir, f"{i}.txt")
        with open(label_filename, 'w', encoding='utf-8') as f:
            # Assuming we're placing the character in the center of the image, width and height are 1
            f.write(f"{char_index} 0.5 0.5 1.0 1.0\n")  # For YOLO format: class_id x_center y_center width height
        
        labels.append((image_filename, label_filename))
        
        if i % 100 == 0:
            print(f"Generated {i} images...")
    
    print(f"Dataset and labels saved to {output_dir}")

# Generate 1000 images (you can increase this number as needed)
num_samples = 1000
save_dataset(num_samples)
