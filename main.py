import streamlit as st
from PIL import Image, UnidentifiedImageError
import torch
import numpy as np
from datetime import datetime
import plotly.express as px
import pandas as pd
import os
import sys
from pathlib import Path
import platform
import pathlib

# Check the operating system and set the appropriate path type
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Create a path that's compatible with your OS
model_path = Path("best.pt")

# --- App Configuration ---
st.set_page_config(
    page_title="Pest Detection System",
    layout="wide",
    page_icon="🐞",
)

# --- Initialize Results Storage ---
if "results" not in st.session_state:
    st.session_state["results"] = []

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main { background-color: #fafafa; }
    .title { font-size: 48px; font-weight: bold; text-align: center; color: #1f4e79; margin-bottom: 20px; }
    .subheader { font-size: 16px; text-align: center; color: #444; margin-bottom: 20px; }
    .footer { font-size: 13px; text-align: center; color: #888; margin-top: 30px; }
    .card {
        background-color: white;
        padding: 20px;
        margin: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .card h3 { color: #1f4e79; margin-bottom: 10px; }
    .stButton>button {
        background-color: #1f4e79 !important;
        color: white !important;
        border-radius: 8px !important;
    }
    .metric { font-size: 24px; font-weight: bold; color: #1f4e79; }
    .pest-grid { display: flex; flex-wrap: wrap; justify-content: center; }
    .pest-card {
        background-color: #fff;
        padding: 10px;
        margin: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        width: 200px;
        text-align: center;
    }
    .pest-card h4 { color: #1f4e79; font-size: 18px; margin-bottom: 5px; }
    .result-card {
        background-color: white;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .result-card h4 { color: #1f4e79; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
tabs = st.sidebar.radio("Go to:", ["🏠 Home", "📄 Upload Image", "📜 Results History", "📊 Analytics", "ℹ️ About"])

# --- List of Detectable Pests ---
DETECTABLE_PESTS = [
    "Armyworm",
    "Legume Blister Beetle",
    "Red Spider",
    "Rice Gall Midge",
    "Rice Leaf Roller",
    "Rice Leafhopper",
    "Rice Water Weevil",
    "Wheat Phloeothrips",
    "White Backed Plant Hopper",
    "Yellow Rice Borer",
]

# --- Load YOLOv5 Model ---
@st.cache_resource
def load_model(model_path):
    try:
        # Import DetectMultiBackend
        from yolov5.models.common import DetectMultiBackend
        from yolov5.utils.torch_utils import select_device

        device = select_device('cpu')  # Use 'cpu' or '0' for GPU
        model = DetectMultiBackend(model_path, device=device)
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        return None

# --- Home Page ---
if tabs == "🏠 Home":
    st.markdown("<h1 class='title'>🐞 Automated Pest Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 class='subheader'>Effortless pest detection for improved agricultural outcomes.</h4>", unsafe_allow_html=True)

    # Display Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><div class="card-header">📄 Images Uploaded</div><p class="metric">{}</p></div>'.format(len(st.session_state["results"])), unsafe_allow_html=True)
    with col2:
        total_pests_detected = sum(len(result['pests']) for result in st.session_state["results"])
        st.markdown('<div class="card"><div class="card-header">🐛 Pests Detected</div><p class="metric">{}</p></div>'.format(total_pests_detected), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><div class="card-header">🧠 Model Version</div><p class="metric">YOLOv5</p></div>', unsafe_allow_html=True)

    st.markdown("<p class='footer'>© 2024 Pest Detection System. All rights reserved.</p>", unsafe_allow_html=True)

# --- Upload Image Tab ---
elif tabs == "📄 Upload Image":
    st.markdown("## Upload an Image for Pest Detection")
    st.markdown("### Detectable Pests:")

    cols = st.columns(4)
    for idx, pest in enumerate(DETECTABLE_PESTS):
        with cols[idx % 4]:
            st.markdown(f"""
                <div class="pest-card">
                    <h4>{pest}</h4>
                </div>
            """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image (jpg, jpeg, png, jfif):", type=["jpg", "jpeg", "png", "jfif"])

    if uploaded_file:
        try:
            # Try to open the image to check if it's a valid image file
            image = Image.open(uploaded_file)
            image.verify()  # Verify that it is, in fact, an image
            image = Image.open(uploaded_file)  # Reopen since verify() closes the file
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Run Detection"):
                with st.spinner("Running YOLOv5 Inference..."):
                    model = load_model(model_path)
                    if model is None:
                        st.stop()

                    device = model.device
                    imgsz = 640  # Inference size (pixels)

                    # Preprocess image
                    img = np.array(image.convert('RGB'))
                    # Letterbox the image to match the model's input size
                    from yolov5.utils.augmentations import letterbox
                    img0 = img.copy()
                    img = letterbox(img, imgsz, stride=model.stride, auto=True)[0]

                    # Convert to tensor
                    img = img.transpose((2, 0, 1))  # HWC to CHW
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(device)
                    img = img.float()
                    img /= 255  # Normalize to 0-1
                    if len(img.shape) == 3:
                        img = img[None]  # Add batch dimension

                    # Inference
                    pred = model(img)

                    # NMS
                    from yolov5.utils.general import non_max_suppression, scale_boxes
                    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

                    # Process predictions
                    for i, det in enumerate(pred):  # detections per image
                        if len(det):
                            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

                            # Annotate image
                            from yolov5.utils.plots import Annotator, colors
                            annotator = Annotator(img0, line_width=2, example=str(model.names))
                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # integer class
                                label = f'{model.names[c]} {conf:.2f}'
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            detected_image = annotator.result()

                            # Extract pests detected and their confidences
                            pests_detected = []
                            for *xyxy, conf, cls in det:
                                pests_detected.append({
                                    'name': model.names[int(cls)],
                                    'confidence': float(conf)
                                })

                            # Display results
                            st.success("Detection complete!")
                            st.image(detected_image, caption="Detection Results", use_container_width=True)

                            # Display the results
                            st.markdown("### Pests Detected:")
                            for pest in pests_detected:
                                st.markdown(f"""
                                    <div class="result-card">
                                        <h4>{pest['name']}</h4>
                                        <p>Confidence Level: {pest['confidence']:.2f}</p>
                                    </div>
                                """, unsafe_allow_html=True)

                            # Log results
                            st.session_state["results"].append({
                                "filename": uploaded_file.name,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "pests": pests_detected,
                                "image": detected_image,
                            })
                        else:
                            st.warning("No detectable pests found in the uploaded image.")
        except UnidentifiedImageError:
            st.error("The uploaded file is not a valid image. Please upload a valid image file.")
        except Exception as e:
            st.error("An error occurred while processing the image.")
            st.error(f"Error details: {e}")

# --- Results History Tab ---
elif tabs == "📜 Results History":
    st.markdown("## Results History")
    if not st.session_state["results"]:
        st.write("No results available yet. Upload an image to start detecting pests.")
    else:
        for idx, result in enumerate(reversed(st.session_state["results"])):
            # Basic info
            st.markdown(f"""
                <div class="card">
                    <p><b>File:</b> {result['filename']}</p>
                    <p><b>Uploaded At:</b> {result['timestamp']}</p>
                    <p><b>Pests Detected:</b> {', '.join([pest['name'] for pest in result['pests']])}</p>
                </div>
            """, unsafe_allow_html=True)
            # Expandable section to show image and details
            with st.expander("View Details"):
                st.image(result['image'], caption="Detection Image", use_container_width=True)
                st.write("### Pests Detected:")
                pests_df = pd.DataFrame(result['pests'])
                st.table(pests_df[['name', 'confidence']].rename(columns={'name': 'Pest', 'confidence': 'Confidence'}))

# --- Analytics Tab ---
elif tabs == "📊 Analytics":
    st.markdown("## Analytics Dashboard")
    if st.session_state["results"]:
        # Aggregate pest detection counts
        pest_counts = {}
        for result in st.session_state["results"]:
            for pest in result['pests']:
                pest_name = pest['name']
                pest_counts[pest_name] = pest_counts.get(pest_name, 0) + 1

        pest_counts_df = pd.DataFrame(list(pest_counts.items()), columns=['Pest', 'Count']).sort_values(by='Count', ascending=False)

        st.markdown("### Most Frequently Detected Pests")
        fig = px.pie(pest_counts_df, names='Pest', values='Count', title='Pest Detection Distribution')
        st.plotly_chart(fig, use_container_width=True)

        # Show detection history over time
        df = pd.DataFrame(st.session_state["results"])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.explode('pests')  # Expand the pests detected
        df['pest_name'] = df['pests'].apply(lambda x: x['name'])
        df['confidence'] = df['pests'].apply(lambda x: x['confidence'])

        if not df.empty:
            st.markdown("### Detection History Over Time")
            # Use a bar chart for simplicity
            detection_counts = df.groupby(['timestamp', 'pest_name']).size().reset_index(name='Counts')
            fig_time = px.bar(detection_counts, x='timestamp', y='Counts', color='pest_name', title='Pest Detections Over Time')
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.write("No pest detections to display over time.")

        # Show summary statistics
        st.markdown("### Summary Statistics")
        total_detections = len(df)
        average_confidence = df['confidence'].mean() if total_detections > 0 else 0
        st.write(f"Total Detections: **{total_detections}**")
        st.write(f"Average Confidence Level: **{average_confidence:.2f}**")

        # Top pests
        top_pests = pest_counts_df.head(3)['Pest'].tolist()
        st.write(f"Top Detected Pests: **{', '.join(top_pests)}**")
    else:
        st.write("No analytics data available yet.")

# --- About Tab ---
elif tabs == "ℹ️ About":
    st.markdown("## About This App")
    st.markdown("""
        ### Introduction
        The **Automated Pest Detection System** is an AI-powered application that uses cutting-edge **YOLOv5** technology to accurately detect pests in agricultural images.
        Our goal is to assist farmers and agricultural professionals in early pest detection, helping to save crops, reduce pesticide usage, and improve yield.

        ### Features
        - **Real-time Pest Detection:** Upload images and get instant results on pest presence.
        - **Comprehensive Pest Library:** Detects a wide range of common agricultural pests.
        - **Analytics Dashboard:** Monitor pest detection trends over time to make informed decisions.
        - **User-friendly Interface:** Easy to navigate, with a clean and intuitive design.

        ### How It Works
        The system analyzes uploaded images using a trained YOLOv5 model to identify pests. The model has been trained on thousands of images to recognize various pest species with high accuracy.

        ### Contact Us
        For inquiries or support, please reach out to us at [support@pestdetectionsystem.com](mailto:support@pestdetectionsystem.com).
    """)
    st.markdown("<p class='footer'>© 2024 Pest Detection System. All rights reserved.</p>", unsafe_allow_html=True)
