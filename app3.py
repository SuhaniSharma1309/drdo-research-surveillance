import streamlit as st
import cv2
import numpy as np
from PIL import Image
from detector import process_image, process_video, set_zone
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Air & Ground Surveillance System",
    page_icon="🚁",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #000000;
    background-image:
        linear-gradient(rgba(0,255,100,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,100,0.05) 1px, transparent 1px);
    background-size: 40px 40px;
}

[data-testid="stMetricValue"] {
    color: #00ff88 !important;
    font-weight: bold;
}

h1, h2, h3 {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<h1 style='text-align:center;'>
AI-Powered Air & Ground Surveillance and Threat Assessment System</h1>
<h5 style='text-align:center; color:gray;'>
DRDO Research Prototype | Real-Time Object Detection, Threat Assessment & Restricted Zone Monitoring</h5>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------

if "objects" not in st.session_state:
    st.session_state.objects = 0

if "threats" not in st.session_state:
    st.session_state.threats = 0

if "det" not in st.session_state:
    st.session_state.det = []

if "result_img" not in st.session_state:
    st.session_state.result_img = None

# --------------------------------------------------
# TOP METRICS
# --------------------------------------------------

metrics_placeholder = st.empty()

def render_metrics():
    with metrics_placeholder.container():
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Detected Objects", st.session_state.objects)
        with k2:
            st.metric("Threats", st.session_state.threats)
        with k3:
            restricted = "ACTIVE" if st.session_state.threats > 0 else "CLEAR"
            st.metric("Restricted Zone", restricted)
        with k4:
            st.metric("System Status", "ONLINE")


# --------------------------------------------------
# CONTROLS
# --------------------------------------------------

c1, c2 = st.columns(2)

with c1:
    mode = st.radio(
        "Analysis Mode",
        ["Image Analysis", "Video Analysis"],
        horizontal=True
    )

with c2:
    type_ = st.radio(
        "Surveillance Type",
        ["Ground Surveillance", "Air Surveillance"],
        horizontal=True
    )

conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)

if mode == "Image Analysis":
    render_metrics()
    st.divider()

# --------------------------------------------------
# IMAGE MODE
# --------------------------------------------------

if mode == "Image Analysis":

    img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp"])

    if img_file:

        img    = Image.open(img_file).convert("RGB")
        img_np = np.array(img)

        img_col, result_col = st.columns(2)

        with img_col:
            st.image(img, caption="Original Image", use_container_width=True)

        # Zone inputs
        st.info("Set restricted zone coordinates (defaults to center 40% of image)")

        h, w = img_np.shape[:2]

        zi1, zi2, zi3, zi4 = st.columns(4)
        with zi1:
            x1 = st.number_input("Zone X1", 0, w, int(w * 0.3))
        with zi2:
            y1 = st.number_input("Zone Y1", 0, h, int(h * 0.3))
        with zi3:
            x2 = st.number_input("Zone X2", 0, w, int(w * 0.7))
        with zi4:
            y2 = st.number_input("Zone Y2", 0, h, int(h * 0.7))

        set_zone(x1, y1, x2, y2)

        if st.button("▶ Run Detection", use_container_width=True):

            with st.spinner("Running detection..."):
                result, det = process_image(img_np.copy(), mode=type_, conf=conf)

            st.session_state.det = det
            st.session_state.result_img = result

            st.session_state.objects = len(det)
            st.session_state.threats = sum(
                1 for d in det
                if d["threat"] == "HIGH RISK"
            )

            render_metrics()

        # Show result image if available
        if st.session_state.result_img is not None:
            with result_col:
                st.image(
                    st.session_state.result_img,
                    caption="Detection Output",
                    use_container_width=True
                )

        # Detection table
        if st.session_state.det:
            st.subheader("Detection Details")
            st.table(st.session_state.det)

# --------------------------------------------------
# VIDEO MODE
# --------------------------------------------------

else:

    vid = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if vid:

        input_path = "temp_input.mp4"

        with open(input_path, "wb") as f:
            f.write(vid.read())

        if st.button("▶ Run Video Detection", use_container_width=True):

            with st.spinner("Processing video — this may take a moment..."):
                processed_path, total_objects, total_threats = process_video(
                    input_path, mode=type_, conf=conf
                )

            if not os.path.exists(processed_path):
                st.error("Processing failed — output file not found.")
            elif os.path.getsize(processed_path) == 0:
                st.error("Processing failed — output file is empty.")
            else:
                st.session_state.objects = total_objects
                st.session_state.threats = total_threats
                render_metrics()

                video1, video2 = st.columns(2)

                with video1:
                    st.subheader("Original Video")
                    with open(input_path, "rb") as f:
                        st.video(f.read())

                with video2:
                    st.subheader("Detection Output")
                    with open(processed_path, "rb") as f:
                        video_bytes = f.read()
                    st.video(video_bytes)

                output_kb = os.path.getsize(processed_path) / 1024
                st.success(f"Done! Output size: {output_kb:.1f} KB")

        st.markdown("---")

st.markdown("""
<div style='text-align:center;color:gray'>
Developed as a DRDO Research Project <br>
AI-Based Air & Ground Surveillance System
</div>
""", unsafe_allow_html=True)
