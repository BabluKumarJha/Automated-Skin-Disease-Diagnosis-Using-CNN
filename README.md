# Automated Skin Disease Classifier 🩺

**Author: Bablu Kumar Jha**

---

## 🔍 Overview

This project is a deep learning–based **skin disease classification system** that identifies dermatological conditions from skin images. It leverages **ResNet50**, which significantly outperformed a baseline CNN in experiments, achieving around **60% accuracy** on the [Skin Cancer 9 Classes (ISIC) dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic).  

The project includes:

- A **Streamlit web app** for image upload and prediction  
- Training and inference scripts (`data.py`, `model.py`, `train.py`, `prediction.py`)  
- Pretrained ResNet model weights  
- Dataset support for multiple skin disease categories  

⚠️ **Disclaimer:** This project is for **research and educational purposes only** and should not be used as a substitute for professional medical advice or diagnosis.  

---

## 📊 Dataset

- **Name:** [Skin Cancer 9 Classes (ISIC)](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)  
- **Classes Covered:**
  - Actinic keratosis  
  - Basal cell carcinoma  
  - Dermatofibroma  
  - Melanoma  
  - Nevus  
  - Seborrheic keratosis  
  - Squamous cell carcinoma  
  - Vascular lesion  
  - Normal skin  

Dataset preprocessing and loading are handled in **`data.py`**.  

---

## 🧠 Model

- **Baseline CNN**: Accuracy ~30–40%  
- **ResNet50 (Fine-tuned)**: Accuracy ~60%  

The ResNet50 architecture showed much better generalization for multi-class skin cancer classification. Model definition is in **`model.py`**.  

---

## 🛠️ Project Structure

```
Automated_skin_disease_classifier/
│
├── app.py                 # Streamlit app for deployment
├── resnet_model_07.pth    # Trained ResNet50 weights
├── skin_cancer_disease.ipynb  # Jupyter notebook for experiments
│
├── script_folder/
│   ├── data.py            # Dataset loading & preprocessing
│   ├── model.py           # Model architecture (CNN & ResNet50)
│   ├── prediction.py      # Inference pipeline
│   ├── train.py           # Training loop and evaluation
│   └── requirements.txt   # Dependencies
│
└── README.md              # Project documentation
```

---

## 🚀 Installation & Usage

### 1️⃣ Clone the repository

```bash
git clone https://github.com/BabluKumarJha/Automated_skin_disease_classifier.git
cd Automated_skin_disease_classifier
```

### 2️⃣ Create environment & install dependencies

```bash
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

pip install -r script_folder/requirements.txt
```

### 3️⃣ Train the model

```bash
python script_folder/train.py
```

### 4️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

### 5️⃣ Upload an image for prediction

- Upload a **JPG skin image**  
- Model outputs disease type + confidence score  

---

## 🌐 Live Demo

You can try the hosted version here:  
👉 [Streamlit App](https://bablukumarjhaautomatedskindiseaseclassifier.streamlit.app)  

---

## 📈 Results

- **Baseline CNN:** ~30–40% accuracy  
- **ResNet50 Fine-tuned:** ~60% accuracy  

Example Prediction (Seborrheic Keratosis, 90.89% confidence):

![Example Prediction](docs/example_prediction.png)

---

## 📌 Future Improvements

- Improve accuracy with **ResNet101 / EfficientNet / Vision Transformers**  
- Use **data augmentation** for better generalization  
- Add **Grad-CAM visualizations** for model explainability  
- Improve UI for clinical usability  

---

## 👨‍💻 Author

**Bablu Kumar Jha**  
- 💼 Researcher & Developer in AI and Computer Vision  
- 🔗 GitHub: [BabluKumarJha](https://github.com/BabluKumarJha)  
  


---

#Python, #BabluKumarJha, #DataScience, #BabluKumarJha, #github, #Machinelearning, #BabluKumarJha, #AI
