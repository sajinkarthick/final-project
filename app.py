import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

# EfficientNet model definition
from torchvision.models import efficientnet_b0

class EfficientNetEmotion(nn.Module):
    def __init__(self, num_classes=7):
        super(EfficientNetEmotion, self).__init__()
        self.model = efficientnet_b0(pretrained=False)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Class labels
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load model
@st.cache_resource
def load_model():
    model = EfficientNetEmotion(num_classes=7)
    model.load_state_dict(torch.load("emotion_effi_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Prediction with probabilities
def predict_emotion(img, model):
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1).squeeze().cpu().numpy()
        pred_class = int(torch.argmax(output, dim=1))
    return pred_class, probs

# Streamlit UI
st.title("Emotion Detection from Facial Image")
st.write("Upload a face image to detect the emotion and view class probabilities.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    model = load_model()
    pred_class, probs = predict_emotion(image, model)

    # Prediction result
    st.success(f"Predicted Emotion: **{class_names[pred_class]}**")

    # Display probabilities as table
    st.subheader("Class Probabilities")
    prob_df = pd.DataFrame({
        "Emotion": class_names,
        "Probability": [f"{p * 100:.2f}%" for p in probs]
    })
    st.dataframe(prob_df, use_container_width=True)

    # Optional: Bar chart
    st.subheader("Probability Distribution")
    fig, ax = plt.subplots()
    ax.bar(class_names, probs, color="skyblue")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Emotion Probabilities")
    st.pyplot(fig)
