import json
import os

import cv2
import face_recognition
import numpy as np
from unidecode import unidecode


def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image at {unidecode(image_path)}")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    return faces


class AuthenticationService:
    def __init__(self):
        self.known_faces = {}
        self.face_dir = 'faces'
        self.json_file = 'known_faces.json'

        if os.path.exists(self.json_file) and os.path.getsize(self.json_file) > 0:
            with open(self.json_file, 'r') as f:
                self.known_faces = {k: [np.array(vi) for vi in v] for k, v in json.load(f).items()}
        else:
            self.register()

    def register(self):
        # Vòng for đầu tiên lặp qua các thư mục trong thư mục faces
        for person in os.listdir(self.face_dir):
            # Trong mỗi vòng lặp,  ta xây dựng đường dẫn đầy đủ đến thư mục của người đó
            person_dir = os.path.join(self.face_dir, person)
            # Sau đó, chúng ta kiểm tra xem mỗi thư mục có phải là thư mục không
            if os.path.isdir(person_dir):
                # Nếu đúng, chúng ta khởi tạo một danh sách rỗng để lưu trữ mã hóa khuôn mặt của người đó
                self.known_faces[person] = []
                # Vòng for thứ hai lặp qua tất cả các tệp hình ảnh trong thư mục của người đó
                for image_file in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_file)
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    # Nếu trong hình ảnh có khuôn mặt, chúng ta thêm nó vào danh sách mã hóa khuôn mặt của người đó
                    if face_encodings:
                        self.known_faces[person].append(face_encodings[0])

        #  In ra kết quả cuoi cùng để kiểm tra
        for person, face_encodings in self.known_faces.items():
            print(f"Registered {len(face_encodings)} images for {person}")

        with open(self.json_file, 'w') as f:
            json.dump({k: [vi.tolist() for vi in v] for k, v in self.known_faces.items()}, f)

        print(f"Saved known faces to {self.json_file}")

    def recognize_faces(self, image_path, min_distance=0.5):
        unknown_image = face_recognition.load_image_file(image_path)
        unknown_face_encodings = face_recognition.face_encodings(unknown_image)

        recognized_faces = []
        for unknown_face_encoding in unknown_face_encodings:
            for person, known_face_encodings in self.known_faces.items():
                # matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
                # if True in matches:
                #     recognized_faces.append(person)
                face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encoding)
                if min(face_distances) < min_distance:
                    recognized_faces.append(person)
                    break

        print(f"Recognized faces: {recognized_faces}")

        return recognized_faces
