class FaceDetectionDTO:
	def __init__(
			self,
			faces_detected_haar_cascade,
			faces_detected_hog,
	):
		self.faces_detected_haar_cascade = faces_detected_haar_cascade
		self.faces_detected_hog = faces_detected_hog
