import os
import pickle
import random

import cv2
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
							# Thay đổi kích thước về 100x100 pixel và làm phẳng thành mảng 1 chiều
							img_resized = cv2.resize(img, (100, 100)).flatten()
							images.append(img_resized)
							labels.append(person_name)

		# Xáo trộn dữ liệu ảnh và nhãn (tên người).
		combined = list(zip(images, labels))
		random.shuffle(combined)
		images[:], labels[:] = zip(*combined)

		# Sử dụng PCA để trích xuất các thành phần chính từ ảnh
		self.pca.fit(images)
		eigen_images = self.pca.transform(images)

		# Sử dụng KNN để huấn luyện mô hình phân loại dựa trên các thành phần chính đã trích xuất
		self.knn.fit(eigen_images, labels)

		for person_name in set(labels):
			print(f"Eigen Registered {labels.count(person_name)} images for {person_name}")

		# Lưu mô hình PCA và KNN vào file
		DataWriter.write__known_faces_eigen_face_to_file(self.modal_file, self.pca, self.knn)

	def recognize_faces(self, image_path):
		# Đọc ảnh và chuyển về ảnh xám, thay đổi kích thước.
		img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		if img is None:
			return None

		img_resized = cv2.resize(img, (100, 100)).flatten()
		# Sử dụng PCA đã được huấn luyện để chuyển đổi ảnh về không gian Eigenfaces.
		unknown_eigen_image = self.pca.transform([img_resized])

		# Dùng KNN để dự đoán tên người từ ảnh đã chuyển đổi.
		personal_names = self.knn.classes_
		probabilities = self.knn.predict_proba(unknown_eigen_image)[0]

		result = []
		for name, probability in zip(personal_names, probabilities):
			if probability > 0.0:
				result.append(name)

		print(f"Eigen Recognized faces: {result}")

		return result

	def load_known_faces(self):
		with open(self.modal_file, 'rb') as f:
			data = pickle.load(f)

			self.pca = data['pca']
			self.knn = data['knn']
