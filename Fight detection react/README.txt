==================================================================================
Minimum System Requirement:
==================================================================================

CPU: Intel i5 2.8GHz or 
RAM: 4 GB
GPU: 4GB Cuda compatible 

Installed 

Python 3.8 or above 
Tensorflow 2.10.0 or above
Keras 2.10.0 or above
openCV 4.6.0 or above
NumPy 1.23.3 or above
Flask 2.2.2 or above
VSCode
CUDNN library to enable GPU inference
Web Browser

=====================================================================================
Steps to run the software
=====================================================================================
1. Extract the zip file to a folder

2. Open Fight detection react\React + Flask\backend\violence_detection.py

3. reset path of recon variable to "..\Fight detection react\model_inception_ANN.h5" and recon2 to "..\Fight detection react\model_inception_LSTM.h5" in line number 76 and 77.

4. Open Fight detection react\React + Flask\backend\\app.py and change the path in line number 14 to the folder in which files are extracted.

5. Now save and close all the files.

6. Go to Fight detection react\React + Flask and right click and select open in VScode

7. Open two seprate terminals and run the following commands in sequence:

=> in first terminal
"cd backend"
"python app.py"

this will launch the flask server

=> in second terminal

"cd frontend"
"npm start"

this will launch React server on default browser.

8. GUI will open and a video need to be uploaded using the GUI and then click on submit. 

9. Software will reload the page and provide predictions.




 

