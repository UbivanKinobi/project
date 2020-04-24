from src.mtcnn.src import detect_faces
import cv2
import dlib
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((5, 2), dtype=dtype)

    coords[0] = (shape.part(1).x, shape.part(1).y)
    coords[1] = (shape.part(2).x, shape.part(2).y)
    coords[2] = (shape.part(0).x, shape.part(0).y)
    coords[3] = (shape.part(3).x, shape.part(3).y)
    coords[4] = (shape.part(4).x, shape.part(4).y)

    # return the list of (x, y)-coordinates
    return coords


class ClassicalDetector:
    def __init__(self, face_detector: str = 'src/haarcascade_frontalface_default.xml',
                 shape_predictor: str = 'src/5_landmarks_predictor.dat'):
        self.face_detector = cv2.CascadeClassifier(face_detector)
        self.shape_predictor = dlib.shape_predictor(shape_predictor)

    def detect_face_haarcascad(self, gray_image):
        bboxes = self.face_detector.detectMultiScale(gray_image, 1.3, 5)
    
        if len(bboxes) == 0:
            return None
    
        max_index = max(range(len(bboxes)), key=lambda x: bboxes[x][2] * bboxes[x][3])
        face_rectangle = bboxes[max_index]
        x, y, w, h = face_rectangle[0], face_rectangle[1], face_rectangle[2], face_rectangle[3]
        face_rectangle = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
        shape = self.shape_predictor(gray_image, face_rectangle)
        coordinates = shape_to_np(shape)
        return coordinates


def detect_face_mtcnn(image):
    #try:
    bboxes, landmarks_arr = detect_faces(image)
    #except:
    #    return None, None

    # if no faces return None
    if len(bboxes) == 0 or len(landmarks_arr) == 0:
        return None, None

    max_index = max(range(len(bboxes)), key=lambda i: (bboxes[i][2] - bboxes[i][0]) *
                                                      (bboxes[i][3] - bboxes[i][1]))
    bounding_box = bboxes[max_index]
    landmarks = landmarks_arr[max_index]

    return bounding_box, landmarks
