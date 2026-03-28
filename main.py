import cv2
from ultralytics import YOLO


def select_roi_with_mouse(video):
    _, frame = video.read()
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    return roi


def open_video(path: str):
    model = YOLO("yolo26n.pt")
    video = cv2.VideoCapture(path)

    if not video.isOpened():
        print('Video not found')

    x, y, h, w = select_roi_with_mouse(video) 

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            print("Can't recieve frame")
            break

        roi_frame = frame[y:y+w, x:x+h]

        results = model(roi_frame)
        for result in results:
            if 0 in result.boxes.cls:
                print('chel')
                cv2.rectangle(frame, (x, y), (x+h, y+w), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+h, y+w), (0, 0, 255), 2)
        cv2.imshow("Orig", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


open_video("2.mp4")