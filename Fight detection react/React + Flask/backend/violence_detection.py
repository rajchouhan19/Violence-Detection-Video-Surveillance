import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import os

IMG_SIZE = 224
MAX_SEQ_LENGTH = 12
NUM_FEATURES = 2048


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=12, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(frame_count/max_frames)
    frames = []
    try:
        for i in range(max_frames):
            ret, frame = cap.read()
            if ret:
                frameId= cap.get(1)
                if (frameId==(i*frame_rate)+1):    
                    frame = cv2.resize(frame, (224,224))
                    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = frame/255
                    frame = frame[:, :,:]
                    frames.append(frame)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, ((i+1)*frame_rate))

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    feature_extractor = build_feature_extractor()
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = ['non violence', 'violence']
    recon=keras.models.load_model("C:\\Users\\Lenovo\\Fight detection\\model_inception_ANN.h5")
    recon2=keras.models.load_model("C:\\Users\\Lenovo\\Fight detection\\model_inception_LSTM.h5")
    frames = load_video(os.path.join("", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = recon.predict(frame_features)[0]
    res=[]
    for i in np.argsort(probabilities)[::-1]:
        print(f"{class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
        x = f"{class_vocab[i]}: {probabilities[i] * 100:5.2f}%"
        res.append(x)
    probabilities1 = recon2.predict([frame_features, frame_mask])[0]
    res1=[]
    for i in np.argsort(probabilities1)[::-1]:
        print(f"{class_vocab[i]}: {probabilities1[i] * 100:5.2f}%")
        x = f"{class_vocab[i]}: {probabilities1[i] * 100:5.2f}%"
        res1.append(x)
    return res, res1