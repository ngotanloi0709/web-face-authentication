import base64
import os
import uuid

from flask import Blueprint, render_template, request, current_app

from dto.FaceResultDTO import FaceResultDTO
from services.CNNFaceAuthenticationService import CNNFaceAuthenticationService
from services.EigenFaceAuthenticationService import EigenFaceAuthenticationService

from services.FaceDetectionService import FaceDetectionService

home = Blueprint('home', __name__)

# Khởi tạo service nhận diện khuôn mặt
# Trong constructor load file json ra để đỡ train lại
cnnFaceAuthenticationService = CNNFaceAuthenticationService("faces")
eigenFaceAuthenticationService = EigenFaceAuthenticationService('faces')


# Train lại dữ liệu trong /faces
# cnnFaceAuthenticationService.register()


@home.route('/', methods=['GET'])
@home.route('/home', methods=['GET'])
@home.route('/login', methods=['GET'])
def index():
	return render_template('login.html')


@home.route('/login', methods=['POST'])
def post_login():
	# Lấy ảm ra dưới dạng Base64
	base64_str = request.form['image']
	save_path = save_image(base64_str)

	# Đếm khuôn mặt bằng haar cascade OpenCV
	faces_detected_haar_cascade = FaceDetectionService.detect_faces_by_haar_cascade_open_cv(save_path)
	# Đếm khuôn mặt bằng HOG Face Recognition
	faces_detected_hog = FaceDetectionService.detect_faces_by_hog_face_recognition(save_path)

	# Nhận diện khuôn mặt bằng Face Recognition
	cnn_result = cnnFaceAuthenticationService.recognize_faces(save_path)
	# Nhận diện khuôn mặt bằng Eigenface
	eigen_result = eigenFaceAuthenticationService.recognize_faces(save_path)
	# Tạo FaceResultDTO để truyền vào template
	face_result_dto = FaceResultDTO(faces_detected_haar_cascade, faces_detected_hog, cnn_result, eigen_result)

	return render_template('result.html', face_result_dto=face_result_dto)


def save_image(base64_str):
	base64_data = base64_str.split(',')[1]
	image_data = base64.b64decode(base64_data)

	# Đặt đường dẫn lưu trữ ảnh
	filename = str(uuid.uuid4()) + '.png'
	save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
	os.makedirs(os.path.dirname(save_path), exist_ok=True)

	# Lưu ảnh
	with open(save_path, 'wb') as f:
		f.write(image_data)

	return save_path
