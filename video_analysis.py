from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
def classify_video(video_path, sample_rate):
    face_detection  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_classifier = load_model('trained-model.h5', compile=False)
    EMOTIONS = ['aggressive', 'rude', 'hessitant', 'friendly', 'dissapointed', 'suprised', 'neutral']
    count = 0
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    success = True
    predictions=[]
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*sample_rate))    # added this line 
        success,frame = vidcap.read()
        count+=1
        if frame is None:
            continue
        frame = imutils.resize(frame,width=400) # Resizing the frame to have a width of 450 pixels.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We define a new variable, gray, as the frame, converted to gray.
        faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
        # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW] # region of images
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
        
        
            preds = emotion_classifier.predict(roi)[0]
            
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            if(label!='neutral'):
                predictions.append(preds)
                # print(label)
    avg_pred = np.average(predictions, axis=0)
    video_emotions_dict={}
    for i in range(len(avg_pred)):
        video_emotions_dict[EMOTIONS[i]]=avg_pred[i]

    # print("Emotions probabilty:")
    # print(video_emotions_dict)

    # print("Dominant Emotion")
    # emotion_probability = np.max(avg_pred)
    # emotion = emotions_labels[avg_pred.argmax()]
    # print(emotion)
    # print(emotion_probability)
    return video_emotions_dict
        