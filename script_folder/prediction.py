import matplotlib.pyplot as plt
from train import *
from data import *
import numpy as np
import data
from PIL import Image
import cv2
import torch


# define prediction function
def predict_image(image_path, model, transform, class_names, device="cuda"):
    model.eval()  # set to evaluation mode
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = probs.argmax(dim=1).item()

    return class_names[pred_class], probs[0][pred_class].item()

transform = transform(train_dir="unused", test_dir="unused")

# Now call the method
test_transform = transform.test_transform()
# print(test_transform)
class_names = train_dataset.classes  # same order as during training


i = np.random.randint(822)
img_path = list(train_dir.glob("*/*.jpg"))
print(img_path[i])
pred, confidence = predict_image(img_path[i], cnn_model, transform.test_transform(), class_names, device)
print(f"Prediction: {pred} (confidence: {confidence:.2f})")


image = cv2.imread(img_path[i])

# Display the image
# cv2.imshow("Image", image)
label = f"{pred} ({confidence:.2f})"

# Define position and style
position = (10, 30)  # x, y coordinates
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (0, 255, 255)  # Green
thickness = 2

# Write on the image
cv2.putText(image, label, position, font, font_scale, color, thickness, cv2.LINE_AA)

# Show the image
cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

