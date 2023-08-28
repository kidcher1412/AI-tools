
import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def gen(video):
    while True:
        while True:
            success, image = video.read()
            if success:
                break
            else:
                # Đặt lại vị trí video về đầu
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return redirect(request.url)

    video_file = request.files['video']
    if video_file.filename == '':
        return redirect(request.url)

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    return redirect(url_for('video_feed', video_path=video_path))


@app.route('/video_feed')
def video_feed():
    # Nếu không có video_path, sử dụng webcam
    video_path = request.args.get('video_path', '0')
    video = cv2.VideoCapture(video_path)

    return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
