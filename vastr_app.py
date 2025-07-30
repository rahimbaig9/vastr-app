import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile
import math

st.set_page_config(page_title="Vastr - Smart Size Estimator", layout="centered")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 📏 Distance Calculation
def calc_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# 📐 Get Measurements in cm
def get_body_measurements(landmarks, image_height, actual_height_cm):
    scaling_factor = actual_height_cm / image_height

    def scaled_distance(p1, p2):
        return calc_distance(p1, p2) * image_height * scaling_factor

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]

    shoulder_cm = scaled_distance(left_shoulder, right_shoulder)
    hip_cm = scaled_distance(left_hip, right_hip)

    # Estimate chest as slightly larger than shoulder (heuristic)
    chest_cm = shoulder_cm * 1.1
    waist_cm = hip_cm * 0.9

    return round(shoulder_cm, 2), round(chest_cm, 2), round(waist_cm, 2), round(hip_cm, 2)

# 👕 Size Mapping (simplified)
def map_size(chest_cm):
    if chest_cm < 85:
        return "XS"
    elif chest_cm < 95:
        return "S"
    elif chest_cm < 105:
        return "M"
    elif chest_cm < 115:
        return "L"
    elif chest_cm < 125:
        return "XL"
    else:
        return "XXL"

# 🧠 Pose Estimation
def estimate_pose(image, height_cm):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None, None
        landmarks = results.pose_landmarks.landmark
        h, w = image.shape[:2]
        shoulder_cm, chest_cm, waist_cm, hip_cm = get_body_measurements(landmarks, h, height_cm)
        size = map_size(chest_cm)
        return {
            "Shoulder": shoulder_cm,
            "Chest": chest_cm,
            "Waist": waist_cm,
            "Hip": hip_cm,
            "Size": size
        }, results

# 🎯 Streamlit UI
st.title("👕 Vastr: Smart Clothing Size Estimator")

gender = st.radio("Select Gender", ["Male", "Female"])
height_cm = st.number_input("Enter your height (in cm)", min_value=100, max_value=250, value=170)

img_file = st.file_uploader("Upload full-body image", type=["jpg", "png", "jpeg"])

if img_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(img_file.read())
    img = cv2.imread(tfile.name)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Estimate Size"):
        with st.spinner("Detecting keypoints and calculating size..."):
            result, landmarks_data = estimate_pose(img, height_cm)
            if result:
                st.success("✅ Size estimation successful!")
                st.write("### Body Measurements:")
                for k, v in result.items():
                    if k != "Size":
                        st.write(f"**{k}**: {v} cm")
                st.write(f"### 👕 Recommended Size: **{result['Size']}**")
                annotated = img.copy()
                mp_drawing.draw_landmarks(annotated, landmarks_data.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                st.image(annotated, caption="Detected Landmarks", use_column_width=True)
            else:
                st.error("❌ Could not detect full body. Try another image.")
