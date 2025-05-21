from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Feature extractor (same used during training)
feature_extractor = MobileNetV2(include_top=False, pooling='avg', input_shape=(224, 224, 3))

app = Flask(__name__)

ann_model = load_model('D:\Research Paper Final Sem\React + Flask\model_inception_ANN.h5')
lstm_model = load_model('D:\Research Paper Final Sem\React + Flask\model_inception_LSTM.h5')
gru_model = load_model('D:\Research Paper Final Sem\React + Flask\model_inception_GRU.h5')

# Global variables
video_source = None  # No source initially
is_running = False  # Detection is off by default
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))  # resize to your model input
    frame = frame.astype('float32') / 255.0
    return frame
    
# Prediction function
# Global frame buffer to store recent features
frame_buffer = []

def preprocess_and_extract_features(frame):
    # Resize and preprocess the frame for MobileNetV2
    resized = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(resized, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array, verbose=0)  # Shape: (1, 1280)
    return features[0]  # Shape: (1280,)

def predict_violence(frame):
    global frame_buffer

    # Extract feature vector from current frame
    features = preprocess_and_extract_features(frame)
    frame_buffer.append(features)

    # Keep buffer size fixed (12 frames)
    if len(frame_buffer) > 12:
        frame_buffer.pop(0)

    # Only predict when we have 12 frames
    if len(frame_buffer) == 12:
        sequence = np.array(frame_buffer)  # Shape: (12, 1280)
        sequence = np.expand_dims(sequence, axis=0)  # Shape: (1, 12, 1280)

        # Predict using ANN, LSTM, and GRU models
        ann_pred = ann_model.predict(sequence, verbose=0)[0]
        lstm_pred = lstm_model.predict(sequence, verbose=0)[0]
        gru_pred = gru_model.predict(sequence, verbose=0)[0]

        return ann_pred, lstm_pred, gru_pred
    else:
        return None, None, None


    return ann_pred, lstm_pred, gru_pred
def generate_frames(source):
    global is_running
    cap = cv2.VideoCapture(source)
    try:
        while is_running:
            success, frame = cap.read()
            if not success:
                break
            else:
                # Resize or crop frame if needed here
                ann_pred, lstm_pred, gru_pred = predict_violence(frame)

                # Only add text if predictions are available
                if ann_pred is not None and lstm_pred is not None and gru_pred is not None:
                    cv2.putText(frame, f"ANN: V:{ann_pred[1]*100:.2f}%, NV:{ann_pred[0]*100:.2f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"LSTM: V:{lstm_pred[1]*100:.2f}%, NV:{lstm_pred[0]*100:.2f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"GRU: V:{gru_pred[1]*100:.2f}%, NV:{gru_pred[0]*100:.2f}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()


@app.route('/', methods=['GET', 'POST'])
def index2():
    global video_source, is_running
    if request.method == 'POST':
        if 'video' in request.files:
            # Save uploaded video
            video_file = request.files['video']
            video_path = f"./static/{video_file.filename}"
            os.makedirs('./static', exist_ok=True)
            video_file.save(video_path)
            video_source = video_path
            is_running = True
        elif 'real_time' in request.form:
            # Start real-time detection
            video_source = 0  # Webcam
            is_running = True
        elif 'stop' in request.form:
            # Stop detection
            is_running = False
    return render_template('index3.html')

@app.route('/video_feed')
def video_feed():
    global video_source
    if video_source is not None:
        return Response(generate_frames(video_source), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "No video source selected.", 400

if __name__ == "__main__":
    app.run(debug=True)
