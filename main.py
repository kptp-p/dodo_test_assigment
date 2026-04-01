import cv2
from ultralytics import YOLO

RED = (0, 0, 255)
GREEN = (0, 255, 0)
LEFT = 'Свободный стол'
APPROACH = 'Стол занят'


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


def main(path: str):
    model = YOLO("yolo26n.pt")
    video = cv2.VideoCapture(path)

    if not video.isOpened():
        print('Video not found')

    x, y, w, h = select_roi_with_mouse(video)

    current_state: bool = False
    pervious_state: bool = False
    stable_state: bool = False
    pervious_stable_state: bool = False
    counter = 0
    current_color = RED

    events: list[tuple] = []

    _, frame = video.read()
    if detect_people_in_roa(model.predict(frame[y:y+h, x:x+w])):
        stable_state = True
        current_state = True
        pervious_state = True

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            print("Can't recieve frame")
            break

        roi_frame = frame[y:y+h, x:x+w]

        results = model.predict(roi_frame)
        pervious_state = current_state

        current_state = detect_people_in_roa(results)

        if current_state == pervious_state:
            counter += 1
        else:
            counter = 1

        if counter >= 30:
            stable_state = current_state

        if stable_state:
            current_color = GREEN
        else:
            current_color = RED

        if stable_state != pervious_stable_state:
            pervious_stable_state = stable_state
            timemark = video.get(cv2.CAP_PROP_POS_MSEC)
            if stable_state:
                events.append((APPROACH, timemark))
            else:
                events.append((LEFT, timemark))

        cv2.rectangle(frame, (x, y), (x+w, y+h), current_color, 2)
        cv2.imshow("Orig", frame)
        print(events)
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("1.mp4")
