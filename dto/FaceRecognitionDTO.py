class FaceRecognitionDTO:
	def __init__(
			self,
			cnn_result,
			dlib_cnn_result,
			eigen_result
	):
		self.cnn_result = cnn_result
		self.dlib_cnn_result = dlib_cnn_result
		self.eigen_result = eigen_result
