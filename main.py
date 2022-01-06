from time import sleep
import cv2
import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model


#-- Data Processer --#

def center_crop_image(img):
    h, w = img.shape[:2]
    aspect = h/w
    if aspect > 1:
        offset = int(np.round((h / 2) - (w / 2)))
        return img[offset:w + offset, 0:w]
    else:
        offset = int(np.round((w / 2) - (h / 2)))
        return img[0:h, offset:h + offset]

def process_images(images, image_size):
    cropped_images = [center_crop_image(img) for img in images]
    resized_images = [cv2.resize(img, (image_size, image_size), cv2.INTER_AREA) for img in cropped_images]
    return resized_images


#-- Image processing methods --#

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    #just making a copy of image passed, so that passed image is not changed
    img_copy = colored_img.copy()
    
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
    
    #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)
    
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
    return img_copy

def crop_face(f_cascade, img, scaleFactor = 1.1):
    # convert the image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)
    try:
        x, y, w, h = faces[0]
    except IndexError:
        return None
    return img[y:y+h, x:x+w]


# Main starts here

print('[INFO] Starting')

# Set working dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Face detect model
face_detect = 'data\haarcascades\haarcascade_frontalface_alt2.xml'
assert os.path.exists(face_detect)
haar_face_cascade = cv2.CascadeClassifier(face_detect)

# CNN
model_folder = 'model'
assert os.path.exists(model_folder)
model = load_model(model_folder)

# Image properties
image_size = 96
extensions = ['jpg', 'jpeg', 'png']

# Folders
source_folder = 'source/'
dest_folder = 'images/'
copy_folder = 'back/'
if not os.path.exists(source_folder):
    os.makedirs(source_folder)
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)
if not os.path.exists(copy_folder):
    os.makedirs(copy_folder)

# Main loop
print('[INFO] Main loop starts')

try:
    while True:
        valid_files = [x for x in os.listdir(source_folder)]
        # If files in source folder
        if os.listdir(source_folder) != 0:
            # Iterate through all files
            for i in os.listdir(source_folder):

                # Try to move files to "copy folder"
                while True:
                    try:
                        shutil.move(source_folder + i, copy_folder + i)
                        break
                    except PermissionError:
                        sleep(2)
                

                # Get the image data
                img_raw = crop_face(haar_face_cascade, cv2.imread(copy_folder + i))

                # Crop & resize image
                img_list = process_images([img_raw], image_size)

                # Reshape image structure
                img = np.array(img_list).reshape(-1, image_size, image_size, 3)

                # Make a prediction
                pred = model.predict(img)

                # Convert prediction to lables
                if pred[0].argmax(axis=0) == 0:
                    label = 'Mask'
                else:
                    label = 'No_Mask'
                
                # Write image to folder
                cv2.imwrite(dest_folder + label + '_' + i, img_list[0])
                print(f'[INFO] Predicted {i}')
        # If no file are found
        else:
            sleep(5)
except KeyboardInterrupt:
    print('[INFO] Exiting')
    exit()