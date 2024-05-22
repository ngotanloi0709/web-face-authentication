import json
import os

import dlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.DataWriter import DataWriter


class DlibCNNFaceAuthenticationService:
	def __init__(self, training_data_path, model_path="known_faces_cnn_dlib.json"):
		print("DlibCNNFaceAuthenticationService init")

		self.known_faces = {}
		self.face_dir = training_data_path
		self.model_file = model_path

		#  Mô hình phát hiện khuôn mặt có sẵn của dlib
		self.detector = dlib.get_frontal_face_detector()
		# Mô hình dự đoán các điểm đặc trưng trên khuôn mặt
		self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
		# Mô hình nhận dạng khuôn mặt
		self.face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

		# Kiểm tra xem file json đã tồn tại và có kích thước lớn hơn 0 không
		if not os.path.exists(self.model_file) or not os.path.getsize(self.model_file) > 0:
			# Nếu không, chúng ta gọi phương thức train để tạo file json và train lại dữ liệu
			self.train()

		# Sau khi đã có file json, chúng ta gọi phương thức load_known_faces để load dữ liệu đã train
		self.load_known_faces()

	def train(self):
		print("FaceNetDlibService start training")

		# Lặp qua tất cả các thư mục trong thư mục faces
		for person in os.listdir(self.face_dir):
			# Xây dựng đường dẫn đầy đủ đến thư mục của người đó
			person_dir = os.path.join(self.face_dir, person)

			# Kiểm tra xem mỗi thư mục có phải là thư mục không
			if os.path.isdir(person_dir):
				# Nếu đúng, chúng ta khởi tạo một danh sách rỗng để lưu trữ mã hóa khuôn mặt của người đó
				self.known_faces[person] = []

				# Lặp qua tất cả các tệp hình ảnh trong thư mục của người đó
				for image_file in os.listdir(person_dir):
					# Đọc ảnh và chuyển đổi sang RGB
					image_path = os.path.join(person_dir, image_file)
					image = dlib.load_rgb_image(image_path)

					# Phát hiện khuôn mặt
					dets = self.detector(image, 1)

					for k, d in enumerate(dets):
						# Xác định vị trí các điểm đặc trưng trên khuôn mặt
						shape = self.shape_predictor(image, d)
						# Tính toán vector biểu diễn 128 chiều của khuôn mặt (FaceNet embedding)
						face_descriptor = self.face_rec_model.compute_face_descriptor(image, shape)
						# Thêm vector biểu diễn vào danh sách mã hóa khuôn mặt của người đó
						self.known_faces[person].append(np.array(face_descriptor))

		for person, face_encodings in self.known_faces.items():
			print(f"Dlib Registered {len(face_encodings)} images for {person}")

		# Ghi dữ liệu đã train vào file json
		DataWriter.write_known_faces_cnn_to_file(self.known_faces, self.model_file)

	def recognize_faces(self, image_path):
		try:
			# Đọc ảnh và chuyển đổi sang RGB
			image = dlib.load_rgb_image(image_path)
			# Phát hiện khuôn mặt trong ảnh
			detected_faces = self.detector(image, 1)

			# Tính toán vector biểu diễn 128 chiều của khuôn mặt (FaceNet embedding) cho tất cả các khuôn mặt phát hiện được
			unknown_face_descriptors = [
				self.face_rec_model.compute_face_descriptor(image, self.shape_predictor(image, face))
				for face in detected_faces
			]

			recognized_faces = []

			# Lặp qua tất cả các vector biểu diễn của khuôn mặt không xác định
			for unknown_face_descriptor in unknown_face_descriptors:
				# So sánh vector biểu diễn của khuôn mặt không xác định với các vector biểu diễn đã biết
				for person, known_face_descriptors in self.known_faces.items():
					# Tính toán độ tương đồng cosine giữa vector biểu diễn của khuôn mặt không xác định và các vector biểu diễn đã biết
					similarities = cosine_similarity([unknown_face_descriptor], known_face_descriptors)
					# Lấy ra chỉ số của vector biểu diễn đã biết có độ tương đồng cao nhất
					best_match_index = np.argmax(similarities)
					# Thêm tên người và độ tương đồng vào danh sách nhận dạng
					recognized_faces.append((person, similarities[0][best_match_index]))

			# Sắp xếp danh sách nhận dạng theo độ tương đồng giảm dần
			recognized_faces.sort(key=lambda x: x[1], reverse=True)
			recognized_faces = [f"{name} ({score})" for name, score in recognized_faces]

			print(f"Dlib CNN Recognized faces: {recognized_faces}")
			return recognized_faces
		except:
			return ["Đã xảy ra lỗi khi nhận dạng khuôn mặt bằng Dlib CNN. Vui lòng thử lại sau."]

	def load_known_faces(self):
		with open(self.model_file, 'r') as f:
			self.known_faces = {k: [np.array(v) for v in v_list] for k, v_list in json.load(f).items()}
