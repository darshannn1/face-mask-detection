from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import base64


app = Flask(__name__)



face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            try:
                face = cv2.resize(face, (224, 224))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = face / 255.0
                face = np.reshape(face, (1, 224, 224, 3))

                pred = model.predict(face, verbose=0)

                if pred[0][0] > pred[0][1]:
                    label = "No Mask"
                    color = (0, 0, 255)
                else:
                    label = "Mask"
                    color = (0, 255, 0)

            except:
                label = "Error"
                color = (255, 255, 0)

            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')



# 👇 PASTE HERE
@app.route('/predict', methods=['POST'])
def predict():
   
    data = request.get_json()
    image_data = data['image']

    image_data = image_data.split(',')[1]

    img_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    
    result = "WORKING"


    return jsonify({'result': result})




@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=False)
