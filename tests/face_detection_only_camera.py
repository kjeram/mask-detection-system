import cv2
import os
import numpy as np
import time

def center_box(box):
    startX, startY, endX, endY = box.astype("int")
    h = endY - startY
    w = endX - startX
    aspect = h/w
    if aspect > 1:
        offset = int(np.round((h / 2) - (w / 2)))
        return startX - offset, startY, endX + offset, endY
    else:
        offset = int(np.round((w / 2) - (h / 2)))
        return startX, startY - offset, endX, endY + offset

# Set working dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Face detect model
prototxt_path = '../deploy.prototxt'
weights_path = '../res10_300x300_ssd_iter_140000_fp16.caffemodel'
face_detect = cv2.dnn.readNet(prototxt_path, weights_path)


# Image properties
image_size = 224

cap = cv2.VideoCapture(0)

time1 = 0
time2 = 0


while True:

    _, frame = cap.read()

    h, w = frame.shape[:2]

    # Flip camera frame
    frame = cv2.flip(frame, -1)

    # OpenCV DNN pre-processing
    blob = cv2.dnn.blobFromImage(frame, 1.0, (image_size, image_size))
    # Process the pre-processed frame & find faces
    face_detect.setInput(blob)
    faces = face_detect.forward()

    # For all faces detected in frame
    for i in range(0, faces.shape[2]):
        # Get the confidence that it is a face 
        confidence = faces[0, 0, i, 2]
        if confidence > 0.7:

            # Get coordinates for face
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = center_box(box)

            # ensure the bounding boxes fall within the dimensions of the frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            # Pre-process frame
            frame_crop = frame[startY:endY, startX:endX]
            frame_crop_resize = cv2.resize(frame_crop, (image_size, image_size), cv2.INTER_AREA)
            frame_crop_resize_reshape = np.array(frame_crop_resize, dtype=np.float32).reshape(-1, image_size, image_size, 3)

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)

    time1 = time.time()
    fps = 1 / (time1 - time2)
    time2 = time1

    cv2.putText(frame, str(int(fps)) + 'fps', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow('window', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
