import json


class DataWriter:
	@staticmethod
	def write_known_faces_to_file(known_faces, json_file):
		with open(json_file, 'w') as f:
			json.dump({k: [vi.tolist() for vi in v] for k, v in known_faces.items()}, f)
		print(f"Saved known faces to {json_file}")
