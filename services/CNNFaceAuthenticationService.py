import json
import os

import face_recognition
import numpy as np

from utils.DataWriter import DataWriter


class CNNFaceAuthenticationService:
	def __init__(self, training_data_path, model_path="known_faces_cnn.json"):
		print("CNNFaceAuthenticationService init")
		self.known_faces = {}
		self.face_dir = training_data_path
		self.model_file = model_path

		# Kiểm tra xem file json đã tồn tại và có kích thước lớn hơn 0 không
		if not os.path.exists(self.model_file) or not os.path.getsize(self.model_file) > 0:
			# Nếu không, chúng ta gọi phương thức train để tạo file json và train lại dữ liệu
			self.train()

		# Sau khi đã có file json, chúng ta gọi phương thức load_known_faces để load dữ liệu đã train
		self.load_known_faces()

	def train(self):
		print("CNNFaceAuthenticationService start training")

		# Vòng for đầu tiên lặp qua các thư mục trong thư mục faces
		for person in os.listdir(self.face_dir):
			# Trong mỗi vòng lặp,  ta xây dựng đường dẫn đầy đủ đến thư mục của người đó
			person_dir = os.path.join(self.face_dir, person)
			# Sau đó, chúng ta kiểm tra xem mỗi thư mục có phải là thư mục không
			if os.path.isdir(person_dir):
				# Nếu đúng, chúng ta khởi tạo một danh sách rỗng để lưu trữ mã hóa khuôn mặt của người đó
				self.known_faces[person] = []
				# Vòng for thứ hai lặp qua tất cả các tệp hình ảnh trong thư mục của người đó
				for image_file in os.listdir(person_dir):
					image_path = os.path.join(person_dir, image_file)
					image = face_recognition.load_image_file(image_path)
					face_encodings = face_recognition.face_encodings(image)
					# Nếu trong hình ảnh có khuôn mặt, chúng ta thêm nó vào danh sách mã hóa khuôn mặt của người đó
					if face_encodings:
						self.known_faces[person].append(face_encodings[0])

		#  In ra kết quả cuoi cùng để kiểm tra
		for person, face_encodings in self.known_faces.items():
			print(f"CNN Registered {len(face_encodings)} images for {person}")

		# Ghi dữ liệu đã train vào file json
		DataWriter.write_known_faces_cnn_to_file(self.known_faces, self.model_file)

	def recognize_faces(self, image_path, min_distance=0.45):
		# Đọc ảnh
		unknown_image = face_recognition.load_image_file(image_path)
		# Mã hóa khuôn mặt từ ảnh đó
		unknown_face_encodings = face_recognition.face_encodings(unknown_image)

		recognized_faces = []
		# Vòng lặp này sẽ lặp qua tất cả các mã hóa khuôn mặt của người lạ
		for unknown_face_encoding in unknown_face_encodings:
			# và so sánh chúng với mã hóa khuôn mặt đã biết
			for person, known_face_encodings in self.known_faces.items():
				# matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
				# if True in matches:
				#     recognized_faces.append(person)
				face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encoding)
				if min(face_distances) < min_distance:
					recognized_faces.append(person)
					continue

		print(f"CNN Recognized faces: {recognized_faces}")

		return recognized_faces

	def load_known_faces(self):
		with open(self.model_file, 'r') as f:
			self.known_faces = {k: [np.array(vi) for vi in v] for k, v in json.load(f).items()}
