import os
import numpy as np
import mediapipe as mp
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

labels = ['Hello', 'Okay', 'Love','stop','peace','fight','Namaste','Angry','call me','Help']
DATA_DIR = "data"

def collect_data():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    cap = cv2.VideoCapture(0)
    for label in labels:
        os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)
        print(f"Collecting for {label}")
        count = 0
        while count < 200:
            ret, frame = cap.read()
            if not ret: break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(image)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    data = []
                    for lm in hand_landmarks.landmark:
                        data += [lm.x, lm.y, lm.z]
                    data = np.array(data)
                    np.save(os.path.join(DATA_DIR, label, f"{count}.npy"), data)
                    count += 1
            cv2.imshow("Collecting", frame)
            if cv2.waitKey(10) == 27: break
    cap.release()
    cv2.destroyAllWindows()

def train():
    X, y = [], []
    for i, label in enumerate(labels):
        for file in os.listdir(os.path.join(DATA_DIR, label)):
            data = np.load(os.path.join(DATA_DIR, label, file))
            X.append(data)
            y.append(i)
    X = np.array(X)
    y = to_categorical(y)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(63,)),
        Dense(64, activation='relu'),
        Dense(len(labels), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10)
    model.save('model/sign_model.h5')

collect_data()
train()