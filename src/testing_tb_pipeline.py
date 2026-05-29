# =========================================================
# ADVANCED STREAMLIT TB DETECTION SYSTEM
# =========================================================
#
# FIXED BINARY DIAGNOSTIC LOGIC
#
# IMPORTANT FIX:
#
# IF "Healthy vs Sick" predicts HEALTHY:
# -> DO NOT RUN "Sick vs TB"
#
# WHY?
# Because TB vs Sick model is trained ONLY on:
# - sick
# - tb
#
# So sending healthy images into that model
# creates meaningless predictions.
#
# =========================================================

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image

from torchvision import models
from torchvision import transforms

# =========================================================
# CONFIG
# =========================================================

IMG_SIZE_STAGE1 = 256
IMG_SIZE_CLASSIFIER = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "health",
    "sick",
    "tb"
]

BINARY_CLASSES = [
    "healthy",
    "abnormal"
]

TB_BINARY_CLASSES = [
    "generic disease",
    "tuberculosis"
]

# =========================================================
# MODEL PATHS
# =========================================================

MODEL_PATHS = {

    "ResNet18": {

        "multiclass":
        r"G:\Ai-CDD\classification_checkpoints\best_resnet18_multiclass.pth",

        "binary":
        r"G:\Ai-CDD\classification_checkpoints\best_resnet18_binary.pth",

        "tb_binary":
        r"G:\Ai-CDD\classification_checkpoints\best_resnet18_tb_vs_sick.pth"
    },

    "ConvNeXt": {

        "multiclass":
        r"G:\Ai-CDD\classification_checkpoints\best_convnext_multiclass.pth",

        "binary":
        r"G:\Ai-CDD\classification_checkpoints\best_convnext_binary.pth",

        "tb_binary":
        r"G:\Ai-CDD\classification_checkpoints\best_convnext_tb_vs_sick.pth"
    },

    "EfficientNet-B3": {

        "multiclass":
        r"G:\Ai-CDD\classification_checkpoints\best_efficientnetb3_multiclass.pth",

        "binary":
        r"G:\Ai-CDD\classification_checkpoints\best_efficientnetb3_binary.pth",

        "tb_binary":
        r"G:\Ai-CDD\classification_checkpoints\best_efficientnet_tb_vs_sick.pth"
    }
}

UNET_PATH = (
    r"G:\Ai-CDD\segmentation_checkpoints"
    r"\final_unet.pth"
)

# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="Advanced TB Detection",
    layout="wide"
)

st.title("🫁 Advanced TB Detection System")

st.markdown(
    """
    Upload a Chest X-ray image and analyze:
    
    - Lung Segmentation
    - ROI Extraction
    - 3-Class Classification
    - Binary Diagnostic Analysis
    - Grad-CAM Visualization
    """
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Model Settings")

selected_model = st.sidebar.selectbox(

    "Select Classification Model",

    [
        "ResNet18",
        "ConvNeXt",
        "EfficientNet-B3"
    ]
)

st.sidebar.write(f"Using: {selected_model}")

# =========================================================
# CLAHE
# =========================================================

clahe = cv2.createCLAHE(

    clipLimit=2.0,

    tileGridSize=(8, 8)
)

# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(img):

    if len(img.shape) == 3:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

    img = cv2.resize(
        img,
        (IMG_SIZE_STAGE1, IMG_SIZE_STAGE1),
        interpolation=cv2.INTER_AREA
    )

    img = clahe.apply(img)

    return img

# =========================================================
# U-NET
# =========================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.down1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(
            1024,
            512,
            2,
            stride=2
        )

        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(
            512,
            256,
            2,
            stride=2
        )

        self.conv2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.conv3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.conv4 = DoubleConv(128, 64)

        self.out = nn.Conv2d(
            64,
            1,
            1
        )

    def forward(self, x):

        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        bn = self.bottleneck(p4)

        u1 = self.up1(bn)
        u1 = torch.cat([u1, d4], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)
        u4 = self.conv4(u4)

        return self.out(u4)

# =========================================================
# LOAD U-NET
# =========================================================

@st.cache_resource
def load_unet():

    model = UNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
            UNET_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    return model

unet_model = load_unet()

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_resnet(path, num_classes):

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    model.load_state_dict(
        torch.load(
            path,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    return model

@st.cache_resource
def load_convnext(path, num_classes):

    model = models.convnext_tiny(
        weights=None
    )

    model.classifier[2] = nn.Linear(
        768,
        num_classes
    )

    model.load_state_dict(
        torch.load(
            path,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    return model

@st.cache_resource
def load_efficientnet(path, num_classes):

    model = models.efficientnet_b3(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    model.load_state_dict(
        torch.load(
            path,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    return model

# =========================================================
# MODEL GETTER
# =========================================================

def get_model(model_name, task):

    path = MODEL_PATHS[model_name][task]

    num_classes = 3 if task == "multiclass" else 2

    if model_name == "ResNet18":

        model = load_resnet(path, num_classes)

        target_layer = model.layer4[1].conv2

    elif model_name == "ConvNeXt":

        model = load_convnext(path, num_classes)

        target_layer = model.features[-1][-1].block[5]

    else:

        model = load_efficientnet(path, num_classes)

        target_layer = model.features[-1][0]

    return model, target_layer

# =========================================================
# TRANSFORM
# =========================================================

classification_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        (
            IMG_SIZE_CLASSIFIER,
            IMG_SIZE_CLASSIFIER
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# MASK CLEANING
# =========================================================

def keep_largest_components(mask):

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]

    largest = (
        np.argsort(areas)[-2:] + 1
    )

    clean = np.zeros_like(mask)

    for idx in largest:

        clean[labels == idx] = 255

    return clean

def fill_holes(mask):

    h, w = mask.shape

    flood = mask.copy()

    flood_mask = np.zeros(
        (h + 2, w + 2),
        np.uint8
    )

    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        255
    )

    flood_inv = cv2.bitwise_not(flood)

    return mask | flood_inv

# =========================================================
# ROI EXTRACTION
# =========================================================

def extract_roi(image, mask):

    mask = (mask > 127).astype(np.uint8)

    lung = image * mask

    coords = cv2.findNonZero(mask)

    if coords is None:

        return cv2.resize(
            image,
            (
                IMG_SIZE_CLASSIFIER,
                IMG_SIZE_CLASSIFIER
            )
        )

    x, y, w, h = cv2.boundingRect(coords)

    pad = 10

    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)

    x2 = min(
        x + w + pad,
        image.shape[1]
    )

    y2 = min(
        y + h + pad,
        image.shape[0]
    )

    cropped = lung[y1:y2, x1:x2]

    cropped = cv2.resize(

        cropped,

        (
            IMG_SIZE_CLASSIFIER,
            IMG_SIZE_CLASSIFIER
        )
    )

    return cropped

# =========================================================
# CORRECTED GRAD-CAM
# =========================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model

        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.backward_hook
            )
        )

    def forward_hook(
        self,
        module,
        input,
        output
    ):

        self.activations = output.detach()

    def backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor,
        class_idx
    ):

        self.model.zero_grad()

        output = self.model(input_tensor)

        score = output[:, class_idx]

        score.backward()

        gradients = self.gradients[0]

        activations = self.activations[0]

        weights = torch.mean(
            gradients,
            dim=(1, 2)
        )

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32
        ).to(DEVICE)

        for i, w in enumerate(weights):

            cam += w * activations[i]

        cam = torch.relu(cam)

        cam -= cam.min()

        cam /= (
            cam.max() + 1e-8
        )

        cam = cam.cpu().numpy()

        return cam

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(

    "Upload Chest X-Ray",

    type=["png", "jpg", "jpeg"]
)

# =========================================================
# MAIN PIPELINE
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    image_np = np.array(image)

    # =====================================================
    # STAGE 1
    # =====================================================

    preprocessed = preprocess_image(
        image_np
    )

    # =====================================================
    # STAGE 2 SEGMENTATION
    # =====================================================

    img_norm = (
        preprocessed.astype(np.float32) / 255.0
    )

    tensor = torch.tensor(
        img_norm,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        pred = unet_model(tensor)

        pred = torch.sigmoid(pred)

        pred = pred[0, 0].cpu().numpy()

    binary_mask = (
        pred > 0.5
    ).astype(np.uint8) * 255

    # =====================================================
    # STAGE 3 MASK CLEANING
    # =====================================================

    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel_open
    )

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    binary_mask = keep_largest_components(
        binary_mask
    )

    binary_mask = fill_holes(
        binary_mask
    )

    # =====================================================
    # STAGE 4 ROI EXTRACTION
    # =====================================================

    roi = extract_roi(
        preprocessed,
        binary_mask
    )

    # =====================================================
    # LOAD MODELS
    # =====================================================

    multiclass_model, target_layer = get_model(
        selected_model,
        "multiclass"
    )

    binary_model, _ = get_model(
        selected_model,
        "binary"
    )

    tb_binary_model, _ = get_model(
        selected_model,
        "tb_binary"
    )

    # =====================================================
    # CLASSIFICATION INPUT
    # =====================================================

    input_tensor = classification_transform(
        roi
    ).unsqueeze(0).to(DEVICE)

    # =====================================================
    # MULTICLASS PREDICTION
    # =====================================================

    with torch.no_grad():

        outputs = multiclass_model(
            input_tensor
        )

        probs = torch.softmax(
            outputs,
            dim=1
        )

        pred_class = torch.argmax(
            probs,
            dim=1
        ).item()

    prediction = CLASS_NAMES[pred_class]

    confidence_scores = (
        probs.cpu().numpy()[0]
    )

    # =====================================================
    # HEALTHY VS SICK
    # =====================================================

    with torch.no_grad():

        binary_out = binary_model(
            input_tensor
        )

        binary_prob = torch.softmax(
            binary_out,
            dim=1
        )

        binary_pred = torch.argmax(
            binary_prob,
            dim=1
        ).item()

    # =====================================================
    # CONDITIONAL TB ANALYSIS
    # =====================================================

    tb_prediction_text = "Not Required"

    if binary_pred == 1:

        with torch.no_grad():

            tb_out = tb_binary_model(
                input_tensor
            )

            tb_prob = torch.softmax(
                tb_out,
                dim=1
            )

            tb_pred = torch.argmax(
                tb_prob,
                dim=1
            ).item()

        tb_prediction_text = (
            TB_BINARY_CLASSES[tb_pred]
        )

    # =====================================================
    # GRAD-CAM
    # =====================================================

    gradcam_generator = GradCAM(
        multiclass_model,
        target_layer
    )

    cam = gradcam_generator.generate(
        input_tensor,
        pred_class
    )

    cam = cv2.resize(
        cam,
        (
            IMG_SIZE_CLASSIFIER,
            IMG_SIZE_CLASSIFIER
        )
    )

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    roi_color = cv2.cvtColor(
        roi.astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )

    overlay = cv2.addWeighted(
        roi_color,
        0.75,
        heatmap,
        0.25,
        0
    )

    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    # =====================================================
    # MAIN LAYOUT
    # =====================================================

    left_col, right_col = st.columns([1.4, 1])

    # =====================================================
    # LEFT PANEL
    # =====================================================

    with left_col:

        st.header("🔬 Pipeline Visualization")

        st.write("### 1. Original X-Ray")

        st.image(
            image_np,
            use_container_width=True
        )

        st.write("### 2. Preprocessed Image")

        st.image(
            preprocessed,
            use_container_width=True
        )

        st.write("### 3. Lung Segmentation")

        st.image(
            pred,
            use_container_width=True
        )

        st.write("### 4. Mask Cleaning")

        st.image(
            binary_mask,
            use_container_width=True
        )

        st.write("### 5. ROI Extraction")

        st.image(
            roi,
            use_container_width=True
        )

        st.write("### 6. Grad-CAM Visualization")

        st.image(
            overlay,
            use_container_width=True
        )

    # =====================================================
    # RIGHT PANEL
    # =====================================================

    with right_col:

        st.header("📊 Results Panel")

        st.write("---")

        st.subheader("Final Prediction")

        st.success(
            prediction.upper()
        )

        st.write("---")

        st.subheader("Confidence Scores")

        fig, ax = plt.subplots()

        ax.bar(
            CLASS_NAMES,
            confidence_scores
        )

        ax.set_ylim([0, 1])

        ax.set_ylabel("Probability")

        st.pyplot(fig)

        st.write("---")

        st.subheader("Model Comparison")

        st.info(
            f"Current Active Model: {selected_model}"
        )

        st.write("---")

        st.subheader("Binary Diagnostic Analysis")

        st.markdown(
            f"""
            ### Healthy vs Sick
            
            Prediction:
            **{BINARY_CLASSES[binary_pred]}**
            """
        )

        # ================================================
        # FIXED CONDITIONAL DISPLAY
        # ================================================

        if binary_pred == 1:

            st.markdown(
                f"""
                ### Sick vs TB
                
                Prediction:
                **{tb_prediction_text}**
                """
            )

        else:

            st.markdown(
                """
                ### Sick vs TB
                
                Prediction:
                **Skipped because image is Healthy**
                """
            )

        st.write("---")

        st.subheader("Prediction Confidence")

        for idx, cls in enumerate(CLASS_NAMES):

            st.write(
                f"{cls}: "
                f"{confidence_scores[idx]*100:.2f}%"
            )

# =========================================================
# GRAD-CAM INTERPRETATION
# =========================================================
#
# HEALTHY:
# - Mostly blue/green
# - Diffuse weak activation
# - No sharp focal hotspots
#
# SICK:
# - Moderate scattered activation
# - Broader yellow/orange regions
# - More diffuse than TB
#
# TB:
# - Strong localized hotspots
# - Red/yellow concentrated regions
# - Often asymmetric
# - Frequently upper-lung dominant
#
# =========================================================