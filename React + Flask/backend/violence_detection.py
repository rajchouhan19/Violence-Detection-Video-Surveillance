import numpy as np
import cv2
from collections import deque
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.nn import softmax

# Constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 12
NUM_FEATURES = 2048

# Load Models

ann_model = keras.models.load_model("D:\\Research Paper Final Sem\\React + Flask\\model_inception_ANN.h5")
lstm_model = keras.models.load_model("D:\\Research Paper Final Sem\\React + Flask\\model_inception_LSTM.h5")
gru_model = keras.models.load_model("D:\\Research Paper Final Sem\\React + Flask\\nmodel_inception_GRU.h5")


# Build Feature Extractor
def build_feature_extractor():
    base_model = keras.applications.InceptionV3(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = base_model(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

def load_video(path, max_frames=12, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, total_frames // max_frames)

        for i in range(max_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resize)
            frame = frame.astype("float32") / 255.0
            frames.append(frame)
    finally:
        cap.release()

    return np.array(frames)

def prepare_video(frames):
    frame_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    frame_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype="bool")

    length = min(MAX_SEQ_LENGTH, len(frames))
    for i in range(length):
        features = feature_extractor.predict(frames[i][None, ...], verbose=0)
        frame_features[0, i] = features
        frame_mask[0, i] = True

    return frame_features, frame_mask

def sequence_prediction(video_path):
    class_vocab = ['non violence', 'violence']
    frames = load_video(video_path)
    frame_features, frame_mask = prepare_video(frames)

    ann_pred = softmax(ann_model.predict(frame_features, verbose=0)[0]).numpy()
    lstm_pred = softmax(lstm_model.predict([frame_features, frame_mask], verbose=0)[0]).numpy()
    gru_pred = softmax(gru_model.predict([frame_features, frame_mask], verbose=0)[0]).numpy()

    ann_res = [f"{class_vocab[i]}: {ann_pred[i] * 100:.2f}%" for i in np.argsort(ann_pred)[::-1]]
    lstm_res = [f"{class_vocab[i]}: {lstm_pred[i] * 100:.2f}%" for i in np.argsort(lstm_pred)[::-1]]
    gru_res = [f"{class_vocab[i]}: {gru_pred[i] * 100:.2f}%" for i in np.argsort(gru_pred)[::-1]]

    return ann_res, lstm_res, gru_res

# Real-Time Prediction 
frame_buffer = deque(maxlen=MAX_SEQ_LENGTH)

def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame.astype("float32") / 255.0
    return frame

def predict_violence_realtime(frame):
    global frame_buffer
    frame = preprocess_frame(frame)
    frame_buffer.append(frame)

    if len(frame_buffer) < MAX_SEQ_LENGTH:
        return None  # Wait until enough frames

    frame_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    for i in range(MAX_SEQ_LENGTH):
        features = feature_extractor.predict(np.expand_dims(frame_buffer[i], axis=0), verbose=0)
        frame_features[0, i] = features

    frame_mask = np.ones((1, MAX_SEQ_LENGTH), dtype="bool")

    ann_pred = softmax(ann_model.predict(frame_features, verbose=0)[0]).numpy()
    lstm_pred = softmax(lstm_model.predict([frame_features, frame_mask], verbose=0)[0]).numpy()
    gru_pred = softmax(gru_model.predict([frame_features, frame_mask], verbose=0)[0]).numpy()

    return ann_pred, lstm_pred, gru_pred
