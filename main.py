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

def classify_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = image
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Set to True if running on Raspberry Pi
arm = False

# Set working dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Face detect model
prototxt_path = 'deploy.prototxt'
weights_path = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
face_detect = cv2.dnn.readNet(prototxt_path, weights_path)

# CNN
if arm:
    from tflite_runtime.interpreter import Interpreter
    model = Interpreter('model.tflite')
    model.allocate_tensors()
else: 
    from tensorflow.keras.models import load_model
    model_folder = 'model'
    model = load_model(model_folder)

# Image properties
image_size = 96

# Camera init
video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()

# Main loop
while True:
    # Start timer
    time1 = time.time()

    # Read frame from camera
    _, frame = video_capture.read()
    h, w = frame.shape[:2]

    # OpenCV DNN pre-processing
    blob = cv2.dnn.blobFromImage(frame, 1.0, (image_size * 2, image_size * 2))
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

            # Make a prediction
            if arm:
                pred = classify_image(model, frame_crop_resize_reshape)
            else:
                pred = model.predict(frame_crop_resize_reshape)

            # Convert prediction to lables
            if pred[0].argmax(axis=0) == 0:
                label = 'Mask'
                color = (0,255,0)
                acc = pred[0][0]
            else:
                label = 'No Mask'
                color = (0,0,255)
                acc = pred[0][1]

            # Draw a rectangle around the faces
            text = label + ' (' +  str(round(acc * 100)) + '%)'
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, text, (startX, startY - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

    # Stop timer & calculate execution time
    time2 = time.time()
    classification_time = np.round(time2-time1, 3)
    cv2.putText(frame, str(classification_time), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('window', frame)
    
    # Key code for ESC
    if cv2.waitKey(1) & 0xFF == 27:  
        break

video_capture.release()
cv2.destroyAllWindows()