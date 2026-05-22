import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os

st.set_page_config(page_title="Sign Sense", page_icon="👐", layout="wide")

st.title("👐 Sign Sense")
st.subheader("Real-time American Sign Language (ASL) Detector")

# Load model
@st.cache_resource
def load_asl_model():
    return load_model('sign_model.h5')

model = load_asl_model()

class_labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']

st.sidebar.header("How to Use")
st.sidebar.info("""
1. Allow camera access
2. Show clear ASL sign in the box
3. Wait for prediction
""")

# Camera input
run = st.checkbox("Start Camera", value=False)

if run:
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # ROI
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
        cv2.rectangle(frame, (x, y), (x+crop_size, y+crop_size), (0, 255, 0), 3)
        text = f"{sign} ({confidence:.1%})"
        cv2.putText(frame, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)
        
        if st.button("Stop Camera"):
            run = False
            break

    cap.release()

else:
    st.info("👆 Click 'Start Camera' to begin detection")

st.caption("Made with ❤️ for Sign Language Accessibility")
