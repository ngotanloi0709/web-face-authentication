import json
import os

import face_recognition
import numpy as np

from utils.DataWriter import DataWriter


class HOGFaceAuthenticationService:
	def __init__(self):
		print("HOGFaceAuthenticationService init")
		self.known_faces = {}
		self.face_dir = 'faces'
		self.json_file = 'known_faces_hog.json'

		if os.path.exists(self.json_file) and os.path.getsize(self.json_file) > 0:
			with open(self.json_file, 'r') as f:
				self.known_faces = {k: [np.array(vi) for vi in v] for k, v in json.load(f).items()}
		else:
			self.register()

	def register(self):
		for person in os.listdir(self.face_dir):
			person_dir = os.path.join(self.face_dir, person)
			if os.path.isdir(person_dir):
				self.known_faces[person] = []
				for image_file in os.listdir(person_dir):
					image_path = os.path.join(person_dir, image_file)
					image = face_recognition.load_image_file(image_path)
					face_encodings = face_recognition.face_encodings(image, model="hog")
					if face_encodings:
						self.known_faces[person].append(face_encodings[0])

		for person, face_encodings in self.known_faces.items():
			print(f"HOG Registered {len(face_encodings)} images for {person}")

		DataWriter.write_known_faces_to_file(self.known_faces, self.json_file)

	def recognize_faces(self, image_path, min_distance=0.5):
		unknown_image = face_recognition.load_image_file(image_path)
		unknown_face_encodings = face_recognition.face_encodings(unknown_image, model="hog")

		recognized_faces = []
		for unknown_face_encoding in unknown_face_encodings:
			for person, known_face_encodings in self.known_faces.items():
				face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encoding)
				if min(face_distances) < min_distance:
					recognized_faces.append(person)
					break

		print(f"HOG Recognized faces: {recognized_faces}")

		return recognized_faces
