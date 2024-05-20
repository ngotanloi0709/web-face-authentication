import os
import pickle
import random

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from utils.DataWriter import DataWriter


class EigenFaceAuthenticationService:
	def __init__(self, training_data_path="faces", model_path="known_faces_eigen.json", num_components=50):
		print("EigenFaceAuthenticationService init")

		#  Dictionary để lưu tên người và khuôn mặt đã biết
		self.known_faces = {}
		self.face_dir = training_data_path
		self.modal_file = model_path
		# Số thành phần chính (Eigenfaces) sử dụng trong PCA
		self.num_components = num_components
		# PCA (Principal Component Analysis)
		# là một kỹ thuật giảm chiều dữ liệu, giúp giảm số lượng biến (features)
		# trong khi vẫn giữ được càng nhiều thông tin quan trọng càng tốt.
		self.pca = PCA(n_components=num_components, whiten=True)
		# KNN (K-Nearest Neighbors) là một thuật toán phân loại (classification) đơn giản nhưng hiệu quả.
		# Nó dựa trên ý tưởng rằng các điểm dữ liệu gần nhau có xu hướng thuộc về cùng một lớp
		self.knn = KNeighborsClassifier(n_neighbors=7)
		# Mô hình phát hiện khuôn mặt có sẵn của OpenCV
		self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
		#  Trung bình của tất cả các ảnh
		self.mean_face = None
		# Kiểm tra xem file json đã tồn tại và có kích thước lớn hơn 0 không
		if not os.path.exists(self.modal_file) or os.path.getsize(self.modal_file) == 0:
			# Nếu không, chúng ta gọi phương thức train để tạo file json và train lại dữ liệu
			self.train()

		# Sau khi đã có file json, chúng ta gọi phương thức load_known_faces để load dữ liệu đã train
		self.load_known_faces()

	def train(self):
		print("EigenFaceAuthenticationService start training")
		images = []
		labels = []

		# Lặp qua tất cả các thư mục trong thư mục faces
		for person_name in os.listdir(self.face_dir):
			# Xây dựng đường dẫn đầy đủ đến thư mục của người đó
			person_dir = os.path.join(self.face_dir, person_name)

			# Kiểm tra xem mỗi thư mục có phải là thư mục không
			if os.path.isdir(person_dir):
				# Lặp qua tất cả các tệp hình ảnh trong thư mục của người đó
				for image_file in os.listdir(person_dir):
					if image_file.endswith(".png") or image_file.endswith(".jpg"):
						image_path = os.path.join(person_dir, image_file)
						# Chuyển ảnh về ảnh xám
						img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

						if img is not None:
							# Phát hiện khuôn mặt trong ảnh
							faces = self.face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5,
							                                           minSize=(30, 30))

							for (x, y, w, h) in faces:
								face_img = img[y:y + h, x:x + w]
								# Thay đổi kích thước ảnh về 100x100  và làm phẳng nó về 1D
								face_img_resized = cv2.resize(face_img, (100, 100)).flatten()
								# Thêm ảnh và nhãn (tên người) vào danh sách
								images.append(face_img_resized)
								labels.append(person_name)

		# Xáo trộn dữ liệu ảnh và nhãn (tên người).
		combined = list(zip(images, labels))
		random.shuffle(combined)
		images[:], labels[:] = zip(*combined)

		# Tinh trung bình của tập dữ liệu
		self.mean_face = np.mean(images, axis=0)

		# Lặp qua và Trừ đi giá trị trung bình của tập dữ liệu
		images = [image - self.mean_face for image in images]

		# Sử dụng PCA để trích xuất các thành phần chính từ ảnh
		self.pca.fit(images)
		eigen_images = self.pca.transform(images)

		# Sử dụng KNN để huấn luyện mô hình phân loại dựa trên các thành phần chính đã trích xuất
		self.knn.fit(eigen_images, labels)

		for person_name in set(labels):
			print(f"Eigen Registered {labels.count(person_name)} images for {person_name}")

		# Lưu mô hình PCA và KNN vào file
		DataWriter.write__known_faces_eigen_face_to_file(self.modal_file, self.pca, self.knn, self.mean_face)

	def recognize_faces(self, image_path):
		try:
			result = []
			# Đọc ảnh và chuyển về ảnh xám, thay đổi kích thước.
			img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
			if img is None:
				return []

			# Phát hiện khuôn mặt trong ảnh
			faces = self.face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
			# Lặp qua tất cả các khuôn mặt phát hiện được
			for (x, y, w, h) in faces:
				face_img = img[y:y + h, x:x + w]
				# Thay đổi kích thước ảnh về 100x100  và làm phẳng nó về 1D
				face_img_resized = cv2.resize(face_img, (100, 100)).flatten()
				# Trừ đi giá trị trung bình của tập dữ liệu
				face_img_resized = face_img_resized - np.mean(face_img_resized)
				unknown_eigen_image = self.pca.transform([face_img_resized])

				# Dùng KNN để dự đoán tên người từ ảnh đã chuyển đổi.
				personal_names = self.knn.classes_
				probabilities = self.knn.predict_proba(unknown_eigen_image)[0]

				for name, probability in zip(personal_names, probabilities):
					result.append((name, probability))

			result.sort(key=lambda x: x[1], reverse=True)

			string_result = [f"{name} ({score})" for name, score in result]

			print(f"Eigen Recognized faces: {string_result} ")

			return string_result
		except:
			return ["Đã xảy ra lỗi khi nhận dạng khuôn mặt bằng EigenFace. Vui lòng thử lại sau."]

	def load_known_faces(self):
		with open(self.modal_file, 'rb') as f:
			data = pickle.load(f)

			self.pca = data['pca']
			self.knn = data['knn']
			self.mean_face = data['mean_face']
