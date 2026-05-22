import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Sign Sense", page_icon="👐", layout="wide")

st.title("👐 Sign Sense")
st.subheader("Real-time American Sign Language (ASL) Detector")

# Load model
@st.cache_resource
def load_asl_model():
    return load_model('sign_model.h5')

model = load_asl_model()

class_labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']

# Sidebar
st.sidebar.header("Instructions")
st.sidebar.info("""
1. Click **Start Camera**  
2. Show clear ASL sign inside the green box  
3. Watch real-time prediction
""")

st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ for accessibility")

# Camera Control
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("▶️ Start Camera", type="primary"):
        st.session_state.camera_running = True

    if st.button("⏹️ Stop Camera"):
        st.session_state.camera_running = False

with col2:
    st.write("")

# Main Camera Area
frame_placeholder = st.empty()

if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Region of Interest
        crop_size = 300
        x = w - crop_size - 30
        y = (h - crop_size) // 2

        hand_roi = frame[y:y+crop_size, x:x+crop_size]

        # Preprocess
        img = cv2.resize(hand_roi, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img, verbose=0)
        class_id = np.argmax(prediction[0])
        confidence = float(prediction[0][class_id])
        sign = class_labels[class_id]

        # Draw
        cv2.rectangle(frame, (x, y), (x + crop_size, y + crop_size), (0, 255, 0), 3)
        text = f"{sign} ({confidence:.1%})"
        cv2.putText(frame, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        # Display frame
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)

    cap.release()
    frame_placeholder.empty()

else:
    st.info("👆 Click **Start Camera** to begin real-time detection")
    st.image("https://via.placeholder.com/800x400/1e2937/60a5fa?text=Sign+Sense+Demo", use_column_width=True)

st.caption("Project: [GitHub Repository](https://github.com/shihabbk18/sign-sense)")
