import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same model class used during training
from torchvision.models import efficientnet_b0

class EfficientNetEmotion(nn.Module):
    def __init__(self, num_classes=7):
        super(EfficientNetEmotion, self).__init__()
        self.model = efficientnet_b0(pretrained=False)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load model weights
model = EfficientNetEmotion().to(device)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()

# Class names
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.title("Emotion Detection from Face Image (.pth Model)")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        st.warning("No face detected. Please upload an image containing a visible face.")
    else:
        x, y, w, h = faces[0]  # Only first face
        face_img = img_np[y:y+h, x:x+w]
        face_pil = Image.fromarray(face_img)
        input_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = class_names[pred_idx]

        st.image(face_pil, caption=f"Detected Face ({pred_label})", use_column_width=True)
        st.success(f"Predicted Emotion: {pred_label}")

        # Show class probabilities
        st.subheader("Class Probabilities:")
        prob_dict = {class_names[i]: float(probs[0][i]) for i in range(len(class_names))}
        st.bar_chart(prob_dict)

