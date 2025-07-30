import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

mp_pose = mp.solutions.pose

def calculate_cm_distance(p1, p2, image_height, person_height_cm, landmarks):
    pixel_distance = np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))
    # Normalize by full body pixel height
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    pixel_height = np.linalg.norm(np.array([shoulder.y, shoulder.x]) - np.array([ankle.y, ankle.x]))
    if pixel_height == 0:
        return 0
    cm_per_pixel = person_height_cm / (pixel_height * image_height)
    return pixel_distance * image_height * cm_per_pixel

def calculate_measurements(landmarks, image_height, person_height_cm, gender):
    shoulder_width = calculate_cm_distance(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        image_height, person_height_cm, landmarks
    )
    hip_width = calculate_cm_distance(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        image_height, person_height_cm, landmarks
    )

    # Adjust multipliers for obese detection (realistic scaling)
    chest = shoulder_width * (3.1 if gender == 'Female' else 2.9)
    waist = hip_width * (2.4 if gender == 'Female' else 2.2)
    hip = hip_width * (2.8 if gender == 'Female' else 2.5)

    return {
        "chest": round(chest, 2),
        "waist": round(waist, 2),
        "hip": round(hip, 2)
    }

def predict_size(measurements, height_cm, gender):
    chest = measurements["chest"]
    waist = measurements["waist"]
    hip = measurements["hip"]

    # Adjusted realistic size thresholds
    if gender == 'Female':
        if waist < 72 and chest < 84 and hip < 88:
            return "S"
        elif waist < 80 and chest < 92 and hip < 96:
            return "M"
        elif waist < 88 and chest < 100 and hip < 104:
            return "L"
        elif waist < 96 and chest < 108 and hip < 112:
            return "XL"
        elif waist < 106 and chest < 116 and hip < 120:
            return "XXL"
        else:
            return "XXXL"
    else:
        if waist < 76 and chest < 90:
            return "S"
        elif waist < 84 and chest < 98:
            return "M"
        elif waist < 92 and chest < 106:
            return "L"
        elif waist < 100 and chest < 114:
            return "XL"
        elif waist < 110 and chest < 122:
            return "XXL"
        else:
            return "XXXL"

# --- Streamlit UI ---
st.title("🧵 Vastr: Smart Clothing Size Estimator")

gender = st.selectbox("Select Gender", ["Male", "Female"])
height_cm = st.number_input("Enter your height (in cm)", min_value=120, max_value=250, step=1)

uploaded_file = st.file_uploader("Upload Full Body Image", type=["jpg", "jpeg", "png"])
if uploaded_file and height_cm:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    image_height = img_np.shape[0]

    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            measurements = calculate_measurements(landmarks, image_height, height_cm, gender)
            size = predict_size(measurements, height_cm, gender)

            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.markdown("### 📏 Measurements (approx.):")
            st.write(f"Chest: {measurements['chest']} cm")
            st.write(f"Waist: {measurements['waist']} cm")
            st.write(f"Hip: {measurements['hip']} cm")
            st.markdown(f"### 👕 Predicted Size: **{size}**")
        else:
            st.error("No person detected in image.")
