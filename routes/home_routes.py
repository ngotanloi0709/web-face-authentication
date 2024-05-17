import base64
import os
import uuid

from flask import Blueprint, render_template, request, current_app

from dto.FaceDetectionDTO import FaceDetectionDTO
from dto.FaceRecognitionDTO import FaceRecognitionDTO
from services.CNNFaceAuthenticationService import CNNFaceAuthenticationService
from services.DlibCNNFaceAuthenticationService import DlibCNNFaceAuthenticationService
from services.EigenFaceAuthenticationService import EigenFaceAuthenticationService
from services.FaceDetectionService import FaceDetectionService

home = Blueprint('home', __name__)

# Khởi tạo service nhận diện khuôn mặt
# Trong constructor load file json ra để đỡ train lại
cnnFaceAuthenticationService = CNNFaceAuthenticationService("faces")
dlibCNNFaceAuthenticationService = DlibCNNFaceAuthenticationService("faces")
eigenFaceAuthenticationService = EigenFaceAuthenticationService('faces')


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

	# Đếm khuôn mặt bằng Haar cascade (OpenCV)
	faces_detected_haar_cascade = FaceDetectionService.detect_faces_by_haar_cascade_open_cv(save_path)
	# Đếm khuôn mặt bằng HOG (Face Recognition)
	faces_detected_hog = FaceDetectionService.detect_faces_by_hog_face_recognition(save_path)

	# Nhận diện khuôn mặt bằng CNN (Face Recognition)
	cnn_result = cnnFaceAuthenticationService.recognize_faces(save_path)
	# Nhận diện khuôn mặt bằng CNN (Dlib)
	dlib_cnn_result = dlibCNNFaceAuthenticationService.recognize_faces(save_path)
	# Nhận diện khuôn mặt bằng Eigenface
	eigen_result = eigenFaceAuthenticationService.recognize_faces(save_path)
	# Tạo FaceDetectionDTO để truyền vào template
	face_detection_dto = FaceDetectionDTO(
		faces_detected_haar_cascade,
		faces_detected_hog
	)
	# Tạo FaceRecognitionDTO để truyền vào template
	face_recognition_dto = FaceRecognitionDTO(
		cnn_result,
		dlib_cnn_result,
		eigen_result
	)

	return render_template('result.html', face_detection_dto=face_detection_dto, face_recognition_dto=face_recognition_dto)


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
