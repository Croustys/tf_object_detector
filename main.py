from Detector import *

MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
IMG_PATH = "test/4.jpg"
VIDEO_PATH = "test/2.mp4"
CLASS_FILE = "coco.names"

threshold = 0.5

detector = Detector()
detector.read_classes(CLASS_FILE)
detector.download_module(MODEL_URL)
detector.load_model()
detector.predict_image(IMG_PATH, threshold)
detector.predict_video(VIDEO_PATH, threshold)