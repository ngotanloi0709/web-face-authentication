import json

import dlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class DlibCNNFaceAuthenticationService:
	def __init__(self, training_data_path, model_path="known_faces_cnn.json"):
		print("DlibCNNFaceAuthenticationService init")
		self.known_faces = {}
		self.face_dir = training_data_path
		self.model_file = model_path

		self.detector = dlib.get_frontal_face_detector()
		self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
		self.face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

		self.load_known_faces()

	def recognize_faces(self, image_path):
		image = dlib.load_rgb_image(image_path)
		detected_faces = self.detector(image, 1)

		unknown_face_descriptors = [
			self.face_rec_model.compute_face_descriptor(image, self.shape_predictor(image, face))
			for face in detected_faces
		]

		recognized_faces = []
		for unknown_face_descriptor in unknown_face_descriptors:
			for person, known_face_descriptors in self.known_faces.items():
				similarities = cosine_similarity([unknown_face_descriptor], known_face_descriptors)
				best_match_index = np.argmax(similarities)

				recognized_faces.append((person, similarities[0][best_match_index]))

		recognized_faces.sort(key=lambda x: x[1], reverse=True)
		recognized_faces = [f"{name} ({score})" for name, score in recognized_faces]

		print(f"Dlib CNN Recognized faces: {recognized_faces}")
		return recognized_faces

	def load_known_faces(self):
		with open(self.model_file, 'r') as f:
			self.known_faces = {k: [np.array(v) for v in v_list] for k, v_list in json.load(f).items()}
