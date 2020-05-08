from skimage import transform as trans
from scipy import spatial
from PIL import Image
from src.detectors import detect_face_mtcnn
import numpy as np
import cv2
import tensorflow as tf
import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def transforms(image, landmarks, mtcnn=False):
    src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]], dtype=np.float32)

    dst = landmarks.astype(np.float32)
    if mtcnn:
        dst = landmarks.reshape((2, 5)).T.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    tform = tform.params[0:2, :]
    warped = (cv2.warpAffine(image, tform, (112, 112), borderValue=0.0) - 127.5) / 128.0
    tensor = warped.reshape((1, 112, 112, 3))
    return tensor


def load_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph


class NN:
    def __init__(self, load: bool = True, threshold: float = 0.9):
        path_to_graph = 'src/frozen_graph.pb'
        self.graph = load_graph(path_to_graph)
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.input = self.graph.get_tensor_by_name('img_inputs:0')
        self.output = self.graph.get_tensor_by_name('embeddings:0')
        feed_dict = {self.input: np.zeros((1, 112, 112, 3))}
        self.sess.run(self.output, feed_dict=feed_dict)
        self.threshold = threshold
        if load:
            self.insiders = np.load('src/insiders_data/embeddings_matrix.npy')
            self.classes = load_obj('src/insiders_data/classes.pkl')
        else:
            self.insiders = None
            self.classes = {}

    def predict(self, image, landmarks):
        if len(self.insiders) == 0:
            print('Insiders are not indicated')
            return -1

        #data = [transforms(image, landmarks) for image, landmarks in zip(images, landmarks_arr)]
        data = transforms(image, landmarks)

        feed_dict = {self.input: data}
        embeddings = self.sess.run(self.output, feed_dict=feed_dict)

        distances = spatial.distance.cdist(self.insiders[:, :-1], embeddings)
        min_arg = np.argmin(distances, axis=0)
        class_ = self.insiders[min_arg, -1].astype('int')
        min_distance = np.array([distances[arg, i] for i, arg in enumerate(min_arg)])

        print(np.mean(min_distance))

        if min_distance < self.threshold:
            return class_[0]
        else:
            return -1

    def get_embedding(self, path_to_image):
        image = Image.open(path_to_image)
        _, landmarks = detect_face_mtcnn(image)
        if landmarks is None:
            print('No faces in the image: ' + path_to_image)
            return -1

        data = transforms(np.array(image), landmarks, mtcnn=True)

        feed_dict = {self.input: data}
        embeddings = self.sess.run(self.output, feed_dict=feed_dict)
        return embeddings

    def save_data(self):
        np.save('src/insiders_data/embeddings_matrix', self.insiders)
        save_obj('src/insiders_data/classes', self.classes)

    def close(self):
        self.sess.close()


def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
