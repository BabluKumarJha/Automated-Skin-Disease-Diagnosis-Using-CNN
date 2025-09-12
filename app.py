
import torch
import streamlit as st
import torchvision.transforms as transforms
# from script_folder.model import device # for streamlit cloud we have to comment, due to this getting error.
# from script_folder.data import class_names
from PIL import Image

# device agnositc code
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


# Class names
class_names = ['Acitinic Keratosis',
 'Basal Cell Carcinoma',
 'Dermatofibroma',
 'Melanoma',
 'Nevus',
 'Pigmented Benign Keratosis',
 'Seborrheic Keratosis',
 'Squamous Cell Carcinoma',
 'Vascular Lesion']

model = torch.load("resnet_model_07.pth", weights_only=False)
model.to(device)

# Prediction function
def predict_image(image_path, model, transform, class_names, device="cuda"):
    model.eval()  # set to evaluation mode
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = probs.argmax(dim=1).item()

    return class_names[pred_class], probs[0][pred_class].item()

# img_path = 'C://Users//BKJST//Desktop//python//Project//skin disease//output_dataset//train//Dermatofibroma//ISIC_0060422.jpg'



test_transform = transforms.Compose(
    [
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std = [0.22, 0.22, 0.22])
    ]
)


# pred, confidence = predict_image(img_path, model, test_transform, class_names, device)
# print("Model predicted: ", pred, "And Model confidence: ", confidence*100)

##------------------------------------------------------------------------------------------------------------##
###################################################
#### Streamlit code
#########################################
st.write("Note: This project is under research and development")
st.title("Apna Skin Clinic & Diagnosis Center ")
st.header("Identify your disease via using skin image.")

st.subheader("Upload your diseases part skin image.")




# Set page config
st.set_page_config(layout="centered")

# Constants
DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 600

# Step 1: Choose input type (outside the form)
st.write("Select input type:")
col1, col2 = st.columns(2)

if col1.button("Upload Image"):
    st.session_state.input_type = "Upload Image"

if col2.button("Take Picture"):
    st.session_state.input_type = "Take Picture"

# Step 2: Show form based on selection
if "input_type" in st.session_state:
    with st.form("upload_form"):
        image_data = None

        if st.session_state.input_type == "Upload Image":
            uploaded_file = st.file_uploader("Upload JPG image", type=["jpg"])
            if uploaded_file is not None:
                image_data = uploaded_file

        elif st.session_state.input_type == "Take Picture":
            camera_image = st.camera_input("Take a picture")
            if camera_image is not None:
                image_data = camera_image

        submitted = st.form_submit_button("Submit")

        if submitted:
            if image_data is not None:
                image = Image.open(image_data).convert("RGB")
                image = image.resize((DISPLAY_WIDTH, DISPLAY_HEIGHT))

                # Replace with your actual prediction function
                pred, confidence = predict_image(image_data, model, test_transform, class_names, device)
                st.write(f"Prediction: {pred} | Confidence: {confidence * 100:.2f}%")
                st.image(image, caption=f"Prediction: {pred} | Confidence: {confidence * 100:.2f}%", use_column_width=False)
            else:

                st.warning("Please upload or capture an image before submitting.")




