from skimage import transform as trans
from scipy import spatial
from PIL import Image
from detectors import detect_face_mtcnn
import numpy as np
import cv2
import tensorflow as tf
import os
import pickle


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
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph


class NN:
    def __init__(self, path_to_insiders_folder, path_to_graph: str = 'frozen_graph.pb', threshold: float = 0.9):
        self.graph = load_graph(path_to_graph)
        self.sess = tf.Session(graph=self.graph)
        self.input = self.graph.get_tensor_by_name('img_inputs:0')
        self.output = self.graph.get_tensor_by_name('embeddings:0')
        if path_to_insiders_folder is not None:
            self.insiders = np.load(os.path.join(path_to_insiders_folder, 'insiders.npy'))
            self.classes = load_obj(os.path.join(path_to_insiders_folder, 'classes.pkl'))
        self.threshold = threshold
        feed_dict = {self.input: np.zeros((1, 112, 112, 3))}
        self.sess.run(self.output, feed_dict=feed_dict)

    def predict(self, images, landmarks_arr):
        if len(self.insiders) == 0:
            print('Не указаны лица имеющие доступ')
            return

        data = [transforms(image, landmarks) for image, landmarks in zip(images, landmarks_arr)]
        data = np.concatenate(data, axis=0)

        feed_dict = {self.input: data}
        embeddings = self.sess.run(self.output, feed_dict=feed_dict)

        distances = spatial.distance.cdist(self.insiders[:, :-1], embeddings)
        min_arg = np.argmin(distances, axis=0)
        classes = self.insiders[min_arg, -1].astype('int')
        min_distances = np.array([distances[arg, i] for i, arg in enumerate(min_arg)])
        print(min_distances)
        ans = sum((min_distances < self.threshold).astype('int'))/min_distances.size
        if ans > 0.5:
            classes = list(classes)
            return max(classes, key=classes.count)
        else:
            return -1

    def save(self, dataset_folder: str, output_folder: str):
        self.insiders = None
        self.classes = {}

        folders = os.listdir(dataset_folder)
        for i, folder in enumerate(folders):
            self.classes[i] = folder
            fol_path = os.path.join(dataset_folder, folder)
            image_names = os.listdir(fol_path)

            for name in image_names:
                im_path = os.path.join(fol_path, name)
                image = Image.open(im_path)
                _, landmarks = detect_face_mtcnn(image)

                data = transforms(np.array(image), landmarks, mtcnn=True)

                feed_dict = {self.input: data}
                embeddings = self.sess.run(self.output, feed_dict=feed_dict)
                embeddings_plus_class = np.append(embeddings, i)

                if self.insiders is None:
                    self.insiders = np.expand_dims(embeddings_plus_class, 0)
                else:
                    self.insiders = np.vstack([self.insiders, np.expand_dims(embeddings_plus_class, 0)])

        np.save(os.path.join(output_folder, 'insiders'), self.insiders)
        save_obj(self.classes, os.path.join(output_folder, 'classes'))

    def close(self):
        self.sess.close()


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
