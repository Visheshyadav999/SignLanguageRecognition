from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('model/sign_model.h5')
labels = ['Hello', 'Okay', 'Love','stop','peace','fight','Namaste','Angry','call me','Help']
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image)
        predicted = ''
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data += [lm.x, lm.y, lm.z]
                if len(data) == 63:
                    prediction = model.predict(np.array([data]))[0]
                    predicted = labels[np.argmax(prediction)]
                    cv2.putText(frame, predicted, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)