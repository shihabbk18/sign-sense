import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

print("Loading model...")
model = load_model('sign_model.h5')
print("✅ Model loaded successfully!")

# ASL Alphabet - 29 classes (match your training)
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

print(f"✅ Ready with {len(class_labels)} classes")

cap = cv2.VideoCapture(0)

print("🎥 Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Region of Interest (right side)
    crop_size = min(300, h-20, w-20)
    x = w - crop_size - 20
    y = (h - crop_size) // 2

    hand_roi = frame[y:y+crop_size, x:x+crop_size]

    # Preprocess exactly like training
    img = cv2.resize(hand_roi, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)
    class_id = np.argmax(prediction[0])
    confidence = float(prediction[0][class_id])

    sign = class_labels[class_id]

    # Draw on screen
    cv2.rectangle(frame, (x, y), (x + crop_size, y + crop_size), (0, 255, 0), 3)
    text = f"{sign} ({confidence:.1%})"
    cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                1.8, (0, 255, 0), 4)

    cv2.imshow('Sign Sense - ASL Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()