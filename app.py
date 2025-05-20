import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms

# Load model (make sure you load your .pth properly)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetEmotion()
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()
model.to(device)

# Define transforms same as training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']  # update with your classes

def detect_face(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

st.title("Emotion Detection with Probabilities")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    if detect_face(img):
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            top_prob, top_class = torch.max(probs, dim=1)
            
            st.write(f"**Predicted Emotion:** {class_names[top_class.item()]}")
            st.write(f"**Confidence:** {top_prob.item()*100:.2f}%")
            
            # Display all class probabilities
            st.write("### Class Probabilities:")
            for i, class_name in enumerate(class_names):
                st.write(f"{class_name}: {probs[0][i].item()*100:.2f}%")
    else:
        st.error("This is not a face image. Please upload a valid face photo.")
