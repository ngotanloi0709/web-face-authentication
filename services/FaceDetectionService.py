import cv2
import face_recognition
from unidecode import unidecode


class FaceDetectionService:
	@staticmethod
	def detect_faces_by_haar_cascade_open_cv(image_path):
		face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

		image = cv2.imread(image_path)

		if image is None:
			print(f"Failed to load image at {unidecode(image_path)}")
			return []

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		faces = face_cascade.detectMultiScale(gray, 1.1, 4)

		return faces

	@staticmethod
	def detect_faces_by_cnn_face_recognition(image_path):
		image = face_recognition.load_image_file(image_path)

		if image is None:
			print(f"Failed to load image at {image_path}")
			return []

		face_locations = face_recognition.face_locations(image, model="cnn")
		return [list(face_location) for face_location in face_locations]

	@staticmethod
	def detect_faces_by_hog_face_recognition(image_path):
		image = face_recognition.load_image_file(image_path)

		if image is None:
			print(f"Failed to load image at {image_path}")
			return []

		face_locations = face_recognition.face_locations(image, model="hog")
		return [list(face_location) for face_location in face_locations]
