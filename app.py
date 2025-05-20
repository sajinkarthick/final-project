import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model class
class EfficientNetEmotion(nn.Module):
    def __init__(self, num_classes=7):
        super(EfficientNetEmotion, self).__init__()
        self.model = efficientnet_b0(pretrained=True)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load the trained model
model = EfficientNetEmotion()
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.to(device)
model.eval()

# Emotion labels
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Face detection function
def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    return faces

# Predict function
def predict_emotion(image):
    faces = detect_face(image)
    if len(faces) == 0:
        return None, None

    # Crop the first face
    x, y, w, h = faces[0]
    face = image.crop((x, y, x + w, y + h))
    face_tensor = transform(face).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(face_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        predicted_label = class_names[np.argmax(probabilities)]

    return predicted_label, probabilities

# Streamlit UI
st.title("Emotion Detection from Facial Image")
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, probs = predict_emotion(image)

    if label is None:
        st.warning("No face detected in the image. Please upload a clear face image.")
    else:
        st.success(f"Predicted Emotion: **{label}**")

        # Show probabilities
        fig, ax = plt.subplots()
        ax.bar(class_names, probs, color='skyblue')
        ax.set_ylabel('Probability')
        ax.set_title('Emotion Probabilities')
        st.pyplot(fig)
