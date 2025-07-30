import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math

st.set_page_config(page_title="Vastr: Smart Clothing Size Estimator")
st.title("🧵 Vastr: Smart Clothing Size Estimator")

gender = st.selectbox("Select Gender", ["Male", "Female"])
uploaded_image = st.file_uploader("Upload Full Body Image", type=["jpg", "jpeg", "png"])
height_cm = st.number_input("Enter your height (in cm)", min_value=100, max_value=250, step=1)

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_clothing_size(gender, shoulder_cm, hip_cm):
    sizes = {
        "Male": [
            (40, 85, "S"),
            (44, 90, "M"),
            (48, 95, "L"),
            (52, 105, "XL"),
            (56, 115, "XXL"),
            (60, 125, "XXXL"),
        ],
        "Female": [
            (36, 80, "S"),
            (40, 85, "M"),
            (44, 90, "L"),
            (48, 100, "XL"),
            (52, 110, "XXL"),
            (56, 120, "XXXL"),
        ],
    }
    for s, h, size in sizes[gender]:
        if shoulder_cm <= s and hip_cm <= h:
            return size
    return "XXXL"

if uploaded_image and height_cm:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            top = landmarks[mp_pose.PoseLandmark.NOSE]
            bottom = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

            shoulder_px = calculate_distance(l_shoulder, r_shoulder)
            hip_px = calculate_distance(l_hip, r_hip)
            full_height_px = calculate_distance(top, bottom)
            px_to_cm = height_cm / full_height_px

            shoulder_cm = shoulder_px * px_to_cm
            hip_cm = hip_px * px_to_cm
            size = get_clothing_size(gender, shoulder_cm, hip_cm)

            st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
            st.subheader("🧍 Your Measurements:")
            st.write(f"**Shoulder Width:** {shoulder_cm:.2f} cm")
            st.write(f"**Hip Width:** {hip_cm:.2f} cm")
            st.subheader(f"👕 Recommended Size: **{size}**")
        else:
            st.error("No body landmarks found. Please upload a full-body image.")
