from skimage import transform as trans
from scipy import spatial
from PIL import Image
from src.detectors import detect_face_mtcnn
import numpy as np
import cv2
import tensorflow as tf
import logging
import sqlite3 as sql

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
module_logger = logging.getLogger('main_loop.model')


def transforms(image, landmarks, mtcnn=False):
    # make transformations of image to feed it to nn
    logger = logging.getLogger('main_loop.model.transforms')
    try:
        # positions of landmarks points
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]], dtype=np.float32)

        dst = landmarks.astype(np.float32)
        # mtcnn return landmarks in different format than HaarCascade
        if mtcnn:
            dst = landmarks.reshape((2, 5)).T.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        tform = tform.params[0:2, :]
        warped = (cv2.warpAffine(image, tform, (112, 112), borderValue=0.0) - 127.5) / 128.0
        return warped
    except Exception as err:
        logger.error('Failed normalizing image: ' + str(err))
        return -1


def load_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph


class NN:
    def __init__(self, load: bool = True, threshold: float = 0.8):
        path_to_graph = 'src/frozen_graph.pb'
        self.graph = load_graph(path_to_graph)
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.input = self.graph.get_tensor_by_name('img_inputs:0')
        self.output = self.graph.get_tensor_by_name('embeddings:0')
        feed_dict = {self.input: np.zeros((1, 112, 112, 3))}
        self.sess.run(self.output, feed_dict=feed_dict)
        self.threshold = threshold
        self.last_photo = 0
        self.min_dist = 0
        if load:
            self.ds = load_dataset()
            self.emp = load_employers()

    def predict(self, image, landmarks):
        logger = logging.getLogger('main_loop.model.nn.predict')
        logger.info('Starting prediction')

        self.last_photo = transforms(image, landmarks)
        if self.last_photo is -1:
            return

        feed_dict = {self.input: self.last_photo.reshape((1, 112, 112, 3))}
        embeddings = self.sess.run(self.output, feed_dict=feed_dict)

        distances = spatial.distance.cdist(self.ds['embeddings'], embeddings)
        min_arg = int(np.argmin(distances, axis=0))
        name = self.ds['names'][min_arg]
        self.min_dist = float(distances[min_arg])

        if self.min_dist < self.threshold:
            logger.info('Done prediction')
            return name
        else:
            logger.info('Done prediction')
            return 'alien'

    def set_threshold(self):
        aliens = np.load('src/data_tf.npy')
        distances = spatial.distance.cdist(self.ds['embedding'], aliens)
        min_distances = np.sort(np.min(distances, axis=0))
        eps = 1e-3
        length = min_distances.size
        index = int(length*eps)
        self.threshold = min_distances[index]
        return

    def get_embedding(self, path_to_image):
        image = Image.open(path_to_image)
        _, landmarks = detect_face_mtcnn(image)
        if landmarks is None:
            print('No faces in the image: ' + path_to_image)
            return -1

        data = transforms(np.array(image), landmarks, mtcnn=True).reshape((1, 112, 112, 3))

        feed_dict = {self.input: data}
        embedding = self.sess.run(self.output, feed_dict=feed_dict).astype('float64')
        return embedding

    def close(self):
        self.sess.close()


def load_dataset():
    logger = logging.getLogger('main_loop.model.load_dataset')
    logger.info('Starting loading dataset')
    con = sql.connect('dataset.db')
    query = """
    SELECT * FROM embeddings;
    """
    ds = {}
    try:
        data = con.execute(query)
        embeddings = []
        names = []
        for row in data:
            embeddings.append(np.frombuffer(row[2]))
            names.append(row[1])
        ds['embeddings'] = np.vstack(embeddings)
        ds['names'] = names
        logger.info('Done loading dataset')
        return ds
    except Exception as err:
        logger.error('Failed loading dataset: ' + str(err))


def load_employers():
    logger = logging.getLogger('main_loop.model.load_employers')
    logger.info('Starting loading employers')
    con = sql.connect('dataset.db')
    query = """
    SELECT * FROM employers;
    """
    employers = {}
    try:
        data = con.execute(query)
        for row in data:
            employers[row[0]] = row[1]
        logger.info('Done loading employers')
        return employers
    except Exception as err:
        logger.error('Failed loading employers: ' + str(err))
