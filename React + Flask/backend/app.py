from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import cv2
import os
import re
from violence_detection import sequence_prediction, predict_violence_realtime

app = Flask(__name__)
CORS(app)

# Webcam control flags
video_capture = None
should_stop_stream = False

# === Fixed label-safe result parser ===
def parse_result(result_list):
    res_dict = {}
    for item in result_list:
        label, value = item.split(":")
        res_dict[label.strip()] = float(value.strip().replace("%", "")) / 100
    return res_dict

# Upload video and get prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = file.filename

    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    save_path = os.path.join(upload_folder, filename)
    file.save(save_path)

    result = sequence_prediction(save_path)
    print("🔥 Raw result from detect_violence:", result)

    ann_result, lstm_result, gru_result = result

    # Parse predictions safely
    ann_parsed = parse_result(ann_result)
    lstm_parsed = parse_result(lstm_result)
    gru_parsed = parse_result(gru_result)

    return jsonify({
        "ANN": {
            "non_violence": ann_parsed.get("non violence", 0),
            "violence": ann_parsed.get("violence", 0)
        },
        "LSTM": {
            "non_violence": lstm_parsed.get("non violence", 0),
            "violence": lstm_parsed.get("violence", 0)
        },
        "GRU": {
            "non_violence": gru_parsed.get("non violence", 0),
            "violence": gru_parsed.get("violence", 0)
        },
        "video_path": filename
    })

# Serve uploaded videos
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# Stream webcam with predictions
@app.route('/video_feed')
def video_feed():
    def generate():
        global video_capture, should_stop_stream
        video_capture = cv2.VideoCapture(0)
        should_stop_stream = False

        while True:
            if should_stop_stream:
                break

            success, frame = video_capture.read()
            if not success:
                break

            result = predict_violence_realtime(frame)
            if result:
                ann_pred, lstm_pred, gru_pred = result
                cv2.putText(frame, f"ANN V:{ann_pred[1]*100:.1f}% NV:{ann_pred[0]*100:.1f}%", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"LSTM V:{lstm_pred[1]*100:.1f}% NV:{lstm_pred[0]*100:.1f}%", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, f"GRU V:{gru_pred[1]*100:.1f}% NV:{gru_pred[0]*100:.1f}%", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        video_capture.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Stop webcam stream
@app.route('/stop', methods=['POST'])
def stop_feed():
    global should_stop_stream
    should_stop_stream = True
    return jsonify({"status": "stopped"})

@app.route('/')
def index():
    return "✅ Violence Detection Backend is running!"

if __name__ == '__main__':
    app.run(debug=True)
