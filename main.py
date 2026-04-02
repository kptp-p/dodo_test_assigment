import argparse

import cv2
import pandas
from ultralytics import YOLO

RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)

EMPTY = 'EMPTY'  # GREEN
APPROACH = 'APPROACH'  # YELLOW
OCCUPIED = 'OCCUPIED'  # RED


def select_roi_with_mouse(video):
    _, frame = video.read()
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    return roi


def detect_people_in_roi(results):
    for result in results:
        if 0 in result.boxes.cls:
            return True
    return False


def convert_eventlist_to_dataframe(events):
    if not events:
        return pandas.DataFrame(columns=['status', 'time'])
    return pandas.DataFrame(events, columns=['status', 'time'])


def calculating_delay(df):
    df["next_status"] = df["status"].shift(-1)
    df["next_time"] = df["time"].shift(-1)

    mask = (df["status"] == EMPTY) & (df["next_status"] == APPROACH)
    delays = df[mask]["next_time"] - df[mask]["time"]

    if len(delays) == 0:
        return None

    return delays.mean()


def main(path: str):
    model = YOLO("yolo26n.pt")
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)

    if not video.isOpened():
        print('Video not found')

    x, y, w, h = select_roi_with_mouse(video)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, fps,
                          (int(video.get(3)), int(video.get(4))))

    current_state: bool = False
    pervious_state: bool = False

    stable_state: bool = False
    pervious_stable_state: bool = False

    counter = 0
    sit_counter = 0

    current_color: tuple = GREEN

    events: list[tuple] = []

    _, frame = video.read()
    if detect_people_in_roi(model.predict(frame[y:y+h, x:x+w])):
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
        current_state = detect_people_in_roi(results)

        if current_state == pervious_state:
            counter += 1
        else:
            counter = 1

        if counter >= 30:
            stable_state = current_state

        if not stable_state:
            current_color = GREEN
            sit_counter = 0
            current_label = EMPTY
        else:
            sit_counter += 1

            if sit_counter >= fps * 10:
                current_color = RED
                current_label = OCCUPIED
            else:
                current_color = YELLOW
                current_label = APPROACH

        if stable_state != pervious_stable_state:
            pervious_stable_state = stable_state
            timemark = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if stable_state:
                events.append((APPROACH, timemark))
            else:
                events.append((EMPTY, timemark))

        if stable_state and sit_counter == int(fps * 20):
            timemark = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
            events.append((OCCUPIED, timemark))

        cv2.rectangle(frame, (x, y), (x+w, y+h), current_color, 2)
        cv2.putText(frame, current_label, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)
        out.write(frame)
        cv2.imshow("Orig", frame)
        print(events)
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()
    df = convert_eventlist_to_dataframe(events)
    print(df)
    print(calculating_delay(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Путь к видео")
    args = parser.parse_args()

    main(args.video)
