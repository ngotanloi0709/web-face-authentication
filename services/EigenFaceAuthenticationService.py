import os
import random

import cv2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from utils.DataWriter import DataWriter


class EigenFaceAuthenticationService:
	def __init__(self, training_data_path="faces", model_path="known_faces_eigen.json", num_components=50):
		print("EigenFaceAuthenticationService init")
		self.known_faces = {}
		self.face_dir = training_data_path
		self.modal_file = model_path
		self.num_components = num_components
		self.pca = PCA(n_components=num_components, whiten=True)
		self.knn = KNeighborsClassifier(n_neighbors=7)

		# Only train if the model file doesn't exist or is empty
		if not os.path.exists(self.modal_file) or os.path.getsize(self.modal_file) == 0:
			self.train()
		# else:
		self.load_known_faces()

	def train(self):
		print("EigenFaceAuthenticationService start training")
		images = []
		labels = []

		for person_name in os.listdir(self.face_dir):  # Lặp qua từng người trong thư mục
			person_dir = os.path.join(self.face_dir, person_name)

			if os.path.isdir(person_dir):
				for image_file in os.listdir(person_dir):  # Lặp qua từng ảnh của người đó
					if image_file.endswith(".png") or image_file.endswith(".jpg"):  # Chỉ xử lý file ảnh
						image_path = os.path.join(person_dir, image_file)
						img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
						if img is not None:  # Kiểm tra ảnh đã được đọc thành công
							img_resized = cv2.resize(img, (100, 100)).flatten()
							images.append(img_resized)
							labels.append(person_name)

		combined = list(zip(images, labels))
		random.shuffle(combined)
		images[:], labels[:] = zip(*combined)

		self.pca.fit(images)
		eigen_images = self.pca.transform(images)
		self.knn.fit(eigen_images, labels)

		for person_name in set(labels):
			print(f"Eigen Registered {labels.count(person_name)} images for {person_name}")

		# Assuming DataWriter handles the saving of the PCA model and KNN model
		DataWriter.write_eigenface_model(self.modal_file, self.pca, self.knn)

	def recognize_faces(self, image_path):
		img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		if img is None:
			return None

		img_resized = cv2.resize(img, (100, 100)).flatten()
		unknown_eigen_image = self.pca.transform([img_resized])

		# prediction = self.knn.predict(unknown_eigen_image)
		# prediction = self.knn.predict_proba(unknown_eigen_image)
		# return prediction
		personal_names = self.knn.classes_

		probabilities = self.knn.predict_proba(unknown_eigen_image)[0]  # Lấy mảng xác suất

		result = []

		for name, probability in zip(personal_names, probabilities):
			if probability > 0.0:
				result.append(name)

		return result

	def load_known_faces(self):
		# Assuming DataWriter handles the loading of PCA and KNN models
		self.pca, self.knn = DataWriter.load_eigenface_model(self.modal_file)
