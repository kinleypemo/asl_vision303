from flask import Flask, render_template, Response, request
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow as tf
import json
from PIL import Image

app = Flask(__name__)
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
model = tf.keras.models.load_model('Model\keras_model.h5')
model.save("model")
classifier = Classifier("model", "Model\labels.txt")
img_height, img_width = 224, 224

labelInfo = {}
with open('Model/classes.json', 'r') as f:
    labelInfo = json.load(f)

offset = 20
imgSize = 224

labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"
]

def gen_frames():
    imgCrop = None
    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

        if imgCrop is not None and imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                if wCal > 0 and imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap : wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(prediction, index)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                if hCal > 0 and imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap : hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)


            cv2.rectangle(
                imgOutput,
                (x - offset, y - offset - 50),
                (x - offset + 90, y - offset - 50 + 50),
                (255, 0, 255),
                cv2.FILLED,
            )
            cv2.putText(
                imgOutput,
                labels[index],
                (x, y - 26),
                cv2.FONT_HERSHEY_COMPLEX,
                1.7,
                (255, 255, 255),
                2,
            )
            cv2.rectangle(
                imgOutput,
                (x - offset, y - offset),
                (x + w + offset, y + h + offset),
                (255, 0, 255),
                4,
            )

            # cv2.imshow("ImageCrop", imgCrop)
            # cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", imgOutput)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/upload', methods=['POST'])
def upload_predict_image():
    filePathName = request.form.get('uploadfilePath', '')
    uploaded_file = request.files['image']
    uploaded_file.save("imageToSave.jpeg")
    img = Image.open("imageToSave.jpeg")
    img = img.resize((img_height, img_width))
    x = np.array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)

    predi = model.predict(x)
    predictedLabel = labelInfo[str(np.argmax(predi[0]))][0]

    context = {'filePathName': filePathName, 'predictedLabel': predictedLabel}
    return render_template('index.html', **context)

@app.route('/')
def home():
    return render_template('home.html')
   
@app.route('/index')
def index():
    return render_template('index.html')
 
@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=False)

