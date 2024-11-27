import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np
import os
from datetime import datetime
import plotly.express as px
import pandas as pd
# Remove the pathlib override
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

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
    /* Your existing CSS styles */
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

# Remove custom torch cache directory settings
# os.environ["TORCH_HOME"] = "/tmp/torch"  # Use a temporary directory

MODEL_PATH = "best.pt"

# --- Load YOLOv5 Model ---
from ultralytics import YOLO

@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        return None

# Load the model
model = load_model(MODEL_PATH)
if model is None:
    st.stop()

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
            # Open and display the uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Run Detection"):
                with st.spinner("Running YOLOv5 Inference..."):
                    image_np = np.array(image)

                    # Perform inference
                    results = model(image_np)

                    if not results or results[0].boxes.shape[0] == 0:
                        st.warning("No detectable pests found in the uploaded image.")
                    else:
                        # Render results
                        annotated_image = results[0].plot()
                        st.success("Detection complete!")
                        st.image(annotated_image, caption="Detection Results", use_container_width=True)

                        # Extract detected pests
                        pests_detected = []
                        for box in results[0].boxes:
                            pests_detected.append({
                                'name': model.names[int(box.cls)],
                                'confidence': float(box.conf),
                            })

                        # Display detected pests
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
                            "image": Image.fromarray(annotated_image),
                        })
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
