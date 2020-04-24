from src.videoutils import WebcamVideoStream
import cv2


def main():
    stream = WebcamVideoStream(src=0).start()
    i = 0
    while True:
        frame = stream.read()
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("w"):
            cv2.imwrite('picture' + str(i) + '.jpg', frame)
            i += 1
        elif key == ord("q"):
            break

    stream.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
