from flask import Flask, request, jsonify
from flask_cors import CORS
import violence_detection
app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    
    file = request.files['file']
    file.save('static/' + file.filename)
    file.seek(0)
    file.save('../frontend/public/' + file.filename)
    path = "C:/Users/Lenovo/Desktop/Fight detection react/React + Flask/backend/static/{}".format(file.filename)
    path2 = file.filename
    # path1 = "./static/{}".format(file.filename)
    
    predict, predict1= violence_detection.sequence_prediction(path)
    print(path2)    



    myobject = {
        "property1": predict,
        "property2": predict1,
        "file_path": path2
    }
    return jsonify(myobject)

if __name__ == '__main__':
    app.run(debug=True)
