from asyncio import current_task
import cv2, time, os, tensorflow as tf

import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)

class Detector:
  def __init__(self):
    pass

  def read_classes(self, classes_file_path):
    with open(classes_file_path, 'r') as f:
      self.classes_list = f.read().splitlines()

    self.color_list = np.random.uniform(low=0, high=255, size=(len(self.classes_list), 3))

  def download_module(self, url):
    file_name = os.path.basename(url)
    self.model_name = file_name[:file_name.index('.')]
    self.cache_dir = "./pretrained_models"

    os.makedirs(self.cache_dir, exist_ok=True)

    get_file(fname=file_name, origin=url, cache_dir=self.cache_dir, cache_subdir="checkpoints", extract=True)
  
  def load_model(self):
    tf.keras.backend.clear_session()
    self.model = tf.saved_model.load(os.path.join(self.cache_dir, "checkpoints", self.model_name, "saved_model"))

  def create_bounding_box(self, image, threshold):
    input_tensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = self.model(input_tensor)
    bboxs = detections['detection_boxes'][0].numpy()
    class_indexes = detections['detection_classes'][0].numpy().astype(np.int32)
    class_scores = detections['detection_scores'][0].numpy()

    imH, imW, imC = image.shape

    bbox_idx = tf.image.non_max_suppression(bboxs, class_scores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)

    if len(bboxs) != 0:
      for i in bbox_idx:
        bbox = tuple(bboxs[i].tolist())
        class_confidence = round(100*class_scores[i])
        class_index = class_indexes[i]

        class_label_text = self.classes_list[class_index]
        class_color = self.color_list[class_index]

        display_text = '{}: {}%'.format(class_label_text, class_confidence)

        ymin, xmin, ymax, xmax = bbox

        xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
        xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=class_color, thickness=1)
        cv2.putText(image, display_text, (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN, 1, class_color, 2)

    return image


  def predict_image(self, img_path, threshold = 0.5):
    image = cv2.imread(img_path)

    bbox_image = self.create_bounding_box(image, threshold)

    cv2.imwrite(self.model_name + ".jpg", bbox_image)
    cv2.imshow("Result", bbox_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
  def predict_video(self, video_path, threshold = 0.5):
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if cap.isOpened() == False:
      print("Error opening video...")
      return
    
    (success, image) = cap.read()

    start_time = 0
    
    while success:
      current_time = time.time()

      fps = 1 / (current_time - start_time)
      start_time = current_time

      bbox_image = self.create_bounding_box(image, threshold)

      cv2.putText(bbox_image, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
      cv2.imshow("Result", bbox_image)

      key = cv2.waitKey(1) & 0xFF
      if key == ord("q"):
        break
      
      (success, image) = cap.read()
    cv2.destroyAllWindows()