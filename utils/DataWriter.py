import json
import pickle


class DataWriter:
	@staticmethod
	def write_known_faces_cnn_to_file(known_faces, json_file):
		with open(json_file, 'w') as f:
			json.dump({k: [vi.tolist() for vi in v] for k, v in known_faces.items()}, f)
		print(f"Saved known faces to {json_file}")

	@staticmethod
	def write_known_faces_eigen_to_file(known_faces, components, mean, explained_variance, json_file):
		with open(json_file, 'w') as f:
			json.dump({
				'known_faces': {k: [vi.tolist() for vi in v] for k, v in known_faces.items()},
				'components': components.tolist(),
				'mean': mean.tolist(),
				'explained_variance': explained_variance.tolist()
			}, f)
		print(f"Saved known faces to {json_file}")

	@staticmethod
	def write_eigenface_model(model_path, pca, knn):
		with open(model_path, 'wb') as f:
			pickle.dump({
				'pca': pca,
				'knn': knn
			}, f)
		print(f"Saved Eigenface model to {model_path}")

	@staticmethod
	def load_eigenface_model(model_path):
		with open(model_path, 'rb') as f:  # Sử dụng 'rb' để đọc dưới dạng nhị phân
			data = pickle.load(f)
			pca = data['pca']
			knn = data['knn']

			return pca, knn
