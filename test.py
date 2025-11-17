import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io

# ---------------- MODEL ----------------
class Drowsiness_Detector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.efficientnet_b0(weights="DEFAULT")

        for params in self.model.parameters():
            params.requires_grad = False

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = Drowsiness_Detector(num_classes=4).to(device)
    model.load_state_dict(torch.load("Drowsiness_Detection_model.pth", map_location=device))
    model.eval()
    return model

model = load_model()
classes = ["Closed", "Open", "no_yawn", "yawn"]

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# ---------------- STREAMLIT UI ----------------
st.title("Drowsiness Detection - Image Classifier")
st.write("Upload an image and the model will classify eye/mouth state.")

uploaded_file = st.file_uploader("Upload an image", type=None)

if uploaded_file:
    # Load the image
    img_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)

    pred_class = classes[pred.item()]

    st.subheader(f"Prediction: {pred_class}")
