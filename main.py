import cv2
from ultralytics import YOLO


def select_roi_with_mouse(video):
    _, frame = video.read()
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    return roi


def detect_people_in_roa(results):
    for result in results:
        if 0 in result.boxes.cls:
            return True
        return False


def open_video(path: str):
    model = YOLO("yolo26n.pt")
    video = cv2.VideoCapture(path)

    if not video.isOpened():
        print('Video not found')

    x, y, w, h = select_roi_with_mouse(video)

    current_state: bool = False
    pervious_state: bool = False
    stable_state: bool = False
    counter = 0
    current_color = (0, 0, 255)

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            print("Can't recieve frame")
            break

        roi_frame = frame[y:y+h, x:x+w]

        results = model.predict(roi_frame)
        pervious_state = current_state

        if detect_people_in_roa(results):
            current_state = True
        else:
            current_state = False

        if current_state == pervious_state:
            counter += 1
        else:
            counter = 0

        if counter >= 10:
            stable_state = current_state

        if stable_state:
            current_color = (0, 255, 0)
        elif not stable_state:
            current_color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), current_color, 2)
        cv2.imshow("Orig", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


open_video("1.mp4")
