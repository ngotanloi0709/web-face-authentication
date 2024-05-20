import json
import pickle


class DataWriter:
	@staticmethod
	def write_known_faces_cnn_to_file(known_faces, json_file):
		with open(json_file, 'w') as f:
			json.dump({k: [vi.tolist() for vi in v] for k, v in known_faces.items()}, f)
		print(f"Saved cnn-faces model to {json_file}")

	@staticmethod
	def write__known_faces_eigen_face_to_file(json_file, pca, knn, mean_face):
		with open(json_file, 'wb') as f:
			pickle.dump({
				'pca': pca,
				'knn': knn,
				'mean_face': mean_face
			}, f)
		print(f"Saved Eigen-face model to {json_file}")
