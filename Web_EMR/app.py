from flask import Flask, render_template, Response
from keras.preprocessing.image import img_to_array
from keras.models import load_model # To import the model.h5 file
import cv2
import numpy as np
app = Flask(__name__)

face_classifier = cv2.CascadeClassifier(
        'D:\Sem 6\ML MP\EMR 3\Emotion_Recg_NB\haarcascade_frontalface_default.xml')
classifier = load_model('D:\Sem 6\ML MP\EMR 3\Emotion_Recg_NB\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# To get a video capture object for the camera.
cap = cv2.VideoCapture(0)

def gen_emr():
    while True:
        success, frame = cap.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converting frame to grayscale
        faces = face_classifier.detectMultiScale(gray)

        # Draw yellow rectangle around face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]  # roi -> region of intrest
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                # To standardize the roi
                roi = roi_gray.astype('float') / 255.0
                # Model is trained on getting array, so this too array
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                # Get max prediction value
                prediction = classifier.predict(roi)[0]
                # Find it in the labels
                label = emotion_labels[prediction.argmax()]
                # Display the emotion.
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('video2.html')

@app.route('/video')
def video():
    return Response(gen_emr(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)