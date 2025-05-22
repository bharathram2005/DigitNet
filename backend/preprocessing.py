from PIL import ImageOps, Image
import numpy as np
import torch
from torchvision import transforms

def preprocess_image(image: Image.Image) -> torch.Tensor:
    # Convert to grayscale and invert (so background is black)
    image = image.convert("L")
    image = ImageOps.invert(image)

    # Convert to numpy array and binarize
    img_array = np.array(image)
    img_array = (img_array > 50) * 255  # basic thresholding

    # Find bounding box
    coords = np.argwhere(img_array)
    if coords.shape[0] == 0:
        return None  # completely blank image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Crop to bounding box
    cropped = img_array[y0:y1, x0:x1]
    cropped_img = Image.fromarray(cropped.astype(np.uint8))

    # Resize to 20x20, then pad to 28x28
    resized = cropped_img.resize((20, 20), Image.LANCZOS)
    new_img = Image.new("L", (28, 28), color=0)
    new_img.paste(resized, ((28 - 20) // 2, (28 - 20) // 2))

    # Final transform: to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(new_img).unsqueeze(0)  # shape: [1, 1, 28, 28]
