import base64
import os
import uuid

from flask import Flask, render_template, request

from routes.user_routes import user
from services.AuthenticationService import detect_faces, AuthenticationService

app = Flask(__name__, template_folder='templates', static_url_path='/static', static_folder='static')
app.register_blueprint(user, url_prefix='/user')
app.config['SECRET_KEY'] = 'secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FACES_FOLDER'] = 'faces'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.use_reloader = True


@app.before_request
def clear_jinja_cache():
    if 'localhost' in request.host_url or '0.0.0.0' in request.host_url:
        app.jinja_env.cache = {}


# Khởi tạo đối tượng AuthenticationService
# Trong constructor load file json ra để đỡ train lại
authenticationService = AuthenticationService()


# Train lại dữ liệu trong /faces
# authenticationService.register()


@app.route('/login', methods=['GET', 'POST'])
@app.route('/', methods=['GET'])
def login():
    if request.method == 'POST':
        # Lấy ảm ra dưới dạng Base64
        base64_str = request.form['image']
        base64_data = base64_str.split(',')[1]
        image_data = base64.b64decode(base64_data)

        # Đặt đường dẫn lưu trữ ảnh
        filename = str(uuid.uuid4()) + '.png'
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Lưu ảnh
        with open(save_path, 'wb') as f:
            f.write(image_data)

        # Đếm khuôn mặt bằng OpenCV
        faces = detect_faces(save_path)

        # Nhận diện khuôn mặt bằng Face Recognition
        result = authenticationService.recognize_faces(save_path)

        return render_template('result.html', faces=f"Face detected at: {faces} total: {len(faces)}",
                               result=f"Recognized faces of : {result} total: {len(result)} peoples")

    return render_template('login.html')


if __name__ == '__main__':
    app.run(extra_dirs=['uploads', 'faces'])
