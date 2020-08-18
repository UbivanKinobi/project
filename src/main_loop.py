from src.videoutils import WebcamVideoStream
from src.detectors import ClassicalDetector
from src.model import NN
import time
import cv2
import sqlite3 as sql
from datetime import datetime
import logging
import collections

import warnings
warnings.filterwarnings("ignore")


class TimeContainer:
    def __init__(self):
        self.timer = time.time()
        self.names = []
        self.photos = []
        self.dists = []

    def add(self, name, dist):
        if time.time() - self.timer > 5:
            self.clear()
        self.names.append(name)
        self.dists.append(dist)
        self.timer = time.time()

    def clear(self):
        self.names.clear()
        self.dists.clear()

    def create_log_data(self, name, was_passed):
        data = [name, was_passed, self.photos, datetime.now()] + self.dists
        return data


def load_nn():
    logger = logging.getLogger('main_loop.load_nn')
    logger.info('Loading network')
    try:
        nn = NN()
        return nn
    except Exception as err:
        logger.error(err)
        return -1


def load_detector():
    logger = logging.getLogger('main_loop.load_detector')
    logger.info('Loading detector')
    try:
        detector = ClassicalDetector()
        return detector
    except Exception as err:
        logger.error(err)
        return -1


def create_con():
    # connect to log.db and create table if not exist
    logger = logging.getLogger('main_loop.create_con')
    logger.info('Creating connection to database')
    try:
        con = sql.connect('log.db')
        with con:
            query = """
                CREATE TABLE IF NOT EXISTS access_control_log (
                id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                name TINYTEXT,
                was_passed TINYINT, 
                photo BLOB,
                datetime DATETIME,
                dist_1 DOUBLE,
                dist_2 DOUBLE,
                dist_3 DOUBLE,
                dist_4 DOUBLE,
                dist_5 DOUBLE );
                """
            con.execute(query)
        return con
    except Exception as err:
        logger.error(err)
        return -1


def load_webcam_stream():
    logger = logging.getLogger('main_loop.load_webcam_stream')
    logger.info('Starting webcam stream')
    stream = WebcamVideoStream(src=0).start()
    if not stream.is_opened():
        logger.error('Failed to connect to camera')
        return -1
    time.sleep(2.0)
    logger.info('Done')
    return stream


def create_logger():
    logger = logging.getLogger('main_loop')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('log.txt')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info('System started')
    return logger


def loop():
    # main preparations before loop
    logger = create_logger()
    nn = load_nn()
    if nn == -1:
        logger.info('End of program')
        return
    # nn.set_threshold()
    detector = load_detector()
    if detector == -1:
        logger.info('End of program')
        return
    con = create_con()
    if con == -1:
        logger.info('End of program')
        return
    stream = load_webcam_stream()
    if stream == -1:
        logger.info('End of program')
        return

    # camera shape for cool drownings on images
    camera_shape = (int(stream.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(stream.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    3)  # (480, 640, 3)

    # data in this container exists only 5 seconds
    # it was done for situation, when person goes away before 5 photos of him were made
    container = TimeContainer()

    # frame loop
    logger.info('Starting frame loop')
    while True:
        # grabs the frame from the stream
        # and makes 2 copies of it for detector (gray) and for drownings (frame_for_print)
        frame = stream.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_for_print = frame.copy()

        # detect face and return face landmarks
        landmarks = detector.detect_face_haarcascad(gray)

        if landmarks is not None:
            # show landmarks
            for (i, (x, y)) in enumerate(landmarks):
                cv2.circle(frame_for_print, (x, y), 1, (0, 0, 255), -1)
            cv2.imshow('Camera', frame_for_print)

            # face identification
            start_time = time.time()
            name = nn.predict(frame, landmarks)
            print('Calculation time: ' + str(round((time.time() - start_time) * 1000, 3)) + 'ms')

            container.add(name, nn.min_dist)

            # final verdict + logging
            if len(container.names) >= 5:
                name = list(collections.Counter(container.names).keys())[0]
                container.photos = nn.last_photo
                if name == 'alien':
                    write_log(con, container.create_log_data('alien', 0))
                    access_denied(stream, camera_shape, 'alien')
                elif nn.emp[name] == 0:
                    write_log(con, container.create_log_data(name, 0))
                    access_denied(stream, camera_shape, name)
                else:
                    write_log(con, container.create_log_data(name, 1))
                    access_granted(stream, camera_shape, name)

                # clear container to be ready for new faces
                container.clear()

        else:
            # if no faces then just show frame
            cv2.imshow('Camera', frame_for_print)

        # key to stop loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.info('Frame loop end\n')
            nn.close()
            break

    # do a bit of cleanup
    stream.stop()
    con.close()
    cv2.destroyAllWindows()


def write_log(con, data):
    logger = logging.getLogger('main_loop.write_log')
    try:
        query = """
        INSERT INTO access_control_log (name, was_passed, photo, datetime, dist_1, dist_2, dist_3, dist_4, dist_5 ) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with con:
            con.execute(query, data)
            logger.info('Log was written to log.db')
    except Exception as err:
        logger.error(err)


def access_granted(stream, camera_shape, insider):
    # grant access for n seconds
    logger = logging.getLogger('main_loop.access_granted')
    logger.info('Access was granted')
    n = 2  # change it to 5
    start_time = time.time()
    while time.time() - start_time < n:
        frame = stream.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'ACCESS GRANTED TO', (15, camera_shape[0] - 50),
                    font, 1, (0, 255, 0), 2)
        cv2.putText(frame, insider.upper(), (15, camera_shape[0] - 15),
                    font, 1, (0, 255, 0), 2)
        cv2.imshow('Camera', frame)
        cv2.waitKey(1)


def access_denied(stream, camera_shape, insider):
    # denied access and block loop for n seconds
    logger = logging.getLogger('main_loop.access_denied')
    logger.info('Access was denied')
    n = 2
    start_time = time.time()
    while time.time() - start_time < n:
        frame = stream.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        if insider == 'alien':
            cv2.putText(frame, 'ACCESS DENIED', (15, camera_shape[0] - 15),
                        font, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'ACCESS DENIED TO', (15, camera_shape[0] - 50),
                        font, 1, (0, 0, 255), 2)
            cv2.putText(frame, insider.upper(), (15, camera_shape[0] - 15),
                        font, 1, (0, 0, 255), 2)
        cv2.imshow('Camera', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    loop()
