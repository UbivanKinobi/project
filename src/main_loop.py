from src.videoutils import WebcamVideoStream
from src.videoutils import FPS
from src.detectors import ClassicalDetector
from src.model import NN
import time
import cv2

import warnings
warnings.filterwarnings("ignore")


class TimeContainer:
    def __init__(self):
        self.timer = time.time()
        self.classes_ = []

    def add(self, class_):
        if time.time() - self.timer > 5:
            self.clear()
        self.classes_.append(class_)
        self.timer = time.time()

    def clear(self):
        self.classes_.clear()


def loop():
    print('[INFO] Loading network...')
    nn = NN()
    print('[INFO] Done')

    #print('[INFO] Setting threshold...')
    #nn.set_threshold()
    #print('[INFO] Done')

    print('[INFO] Loading detectors...')
    detector = ClassicalDetector()
    print('[INFO] Done')

    print('[INFO] Starting webcam stream...')
    stream = WebcamVideoStream(src=0).start()
    if not stream.is_opened():
        print('[ERROR] failed to connect to camera')
        return
    time.sleep(2.0)
    fps = FPS().start()
    print('[INFO] Done')

    camera_shape = (int(stream.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(stream.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    3)  # (480, 640, 3)

    container = TimeContainer()
    # frame loop
    while True:
        # grab the frame from the stream
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
            class_ = nn.predict(frame, landmarks)
            print('Calculation time: ' + str(round((time.time() - start_time) * 1000, 3)) + 'ms')

            container.add(class_)

            # final verdict
            if len(container.classes_) >= 5:
                insider = max(set(container.classes_), key=container.classes_.count)
                if insider != -1:
                    access_granted(stream, camera_shape, nn.classes[insider])
                else:
                    access_denied(stream, camera_shape)
                # clearing container to be ready for new faces
                container.clear()

        else:
            cv2.imshow('Camera', frame_for_print)

        # key to stop loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            nn.close()
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    stream.stop()
    cv2.destroyAllWindows()


def access_granted(stream, camera_shape, insider):
    start_time = time.time()
    while time.time() - start_time < 2:
        frame = stream.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'ACCESS GRANTED TO ' + insider.upper(), (15, camera_shape[0] - 15),
                    font, 1, (0, 255, 0), 2)
        cv2.imshow('Camera', frame)
        cv2.waitKey(1)


def access_denied(stream, camera_shape):
    start_time = time.time()
    while time.time() - start_time < 2:
        frame = stream.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'ACCESS DENIED', (15, camera_shape[0] - 15),
                    font, 1, (0, 0, 255), 2)
        cv2.imshow('Camera', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    loop()
