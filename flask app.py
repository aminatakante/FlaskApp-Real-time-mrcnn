import numpy as np
import sys, os
from fastapi import FastAPI, UploadFile, File
from flask import Flask, render_template, Response
from starlette.requests import Request
import io
import cv2
from pydantic import BaseModel
#app = FastAPI()
import uvicorn
import urllib.request

# Mask R-CNN 
import time
import imutils
import numpy as np
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import utils, visualize
from imutils.video import WebcamVideoStream
import random

from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

import tensorflow as tf
global graph
graph = tf.get_default_graph()

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

print("test1")
camera = cv2.VideoCapture("http://192.168.1.45:8080/video") 
print("test2")
camera2 = cv2.VideoCapture("http://192.168.1.45/video")
print("test3")
camera2 = cv2.VideoCapture("http://192.168.1.45/shot.jpg") 

class ImageType(BaseModel):
    url: str
        
print("EVERYTHING IS WELL IMPORTED")

class face_maskConfig(Config):
    """Configuration for training "face mask recogniton" model.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "face_mask"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    
    # Number of GPUs in the machine (should be increased respectively for AWS instance)
    GPU_COUNT = 1 #2,4

    # The requirement set by R-CNN model (to ensure the quality of classifications) for the image input dimensions size
    IMAGE_MIN_DIM = 128

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2 # 2 classes
    
    # Number of epochs = Number of training images
    EPOCHS = 500
    
    # Number of training steps per epoch = Number of train images/batch size
    STEPS_PER_EPOCH = 100
    
    # Number of validation steps = Number of validation images
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 45
    
    LEARNING_RATE = 0.001

    # Skip detections with < XX% confidence
    DETECTION_MIN_CONFIDENCE = 0.6
    
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution image
    USE_MINI_MASK = False
    #MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    


class InferenceConfig(face_maskConfig):
 # Set batch size to 1 since we’ll be running inference on
 # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

MODEL_DIR = "/home/ubuntu/"

# Create model object in inference mode.

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights('/home/ubuntu/mask_rcnn_face_mask_iteration_2.h5', by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Define COCO Class names
class_names = ['BG','Face with mask', 'Face without mask']
colors = visualize.random_colors(40)

print("app starting")

def gen_frames():
    while True:
        print("we are in face mask detection")
        success, frame = camera.read()        
        print("we loaded the image1")
        ret, buffer = cv2.imencode('.jpg', frame)
        print("we loaded the image2")
        frame = buffer.tobytes()
        #frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        print("we loaded the image3")
        with graph.as_default():
            results = model.detect([frame])
            
        r = results[0]
        print("we loaded the image6")
        print("ok1")      
        output_image = visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], colors)
        print("ok2")
        frame = output_image
        print("about to yield")
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
        

@app.route("/predict")
def video_stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

            
@app.route("/")
def index():
    return render_template("index.html")



      

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)



 
