class FaceResultDTO:
	def __init__(self, faces_detected_haar_cascade, faces_detected_hog, cnn_result, hog_result):
		self.faces_detected_haar_cascade = faces_detected_haar_cascade
		self.faces_detected_hog = faces_detected_hog
		self.cnn_result = cnn_result
		self.hog_result = hog_result
