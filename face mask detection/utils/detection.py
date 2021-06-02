import tensorflow as tf
import cv2
import numpy as np
import warnings
import utils.visualization_utils as viz_utils
from utils.load_model import Load_model
from logger.app_logger import App_Logger
from os import path

warnings.filterwarnings('ignore')


class Detection:
    """
        ClassName: Detection
        Description: This class contain methods for real time object detection.
        Parameter: model_name - str, saved model directory name
        Written By: Tejas Dadhaniya
        Version: 1.0

    """

    def __init__(self, model_name):
        self.load = Load_model()
        self.model_name = model_name
        self.logger = App_Logger()

    def stream(self):
        """
            ClassName: Detection
            Method: stream
            Description: This is method for real time object detection through integrated webcam.
            Parameter: None
            Written By: Tejas Dadhaniya
            Version: 1.0
            Return: None
            Exception: Raise Exception

        """

        file = open(path.join("logs", "Logs.txt"), 'a+')
        try:
            self.logger.log(file_object=file, log_message='Streaming started..!!')
            vid = cv2.VideoCapture(0)
            detect_fn, category_index = self.load.loader(self.model_name)
            while True:
                ret, frame = vid.read()
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_expanded = np.expand_dims(frame, axis=0)
                input_tensor = tf.convert_to_tensor(frame)
                input_tensor = input_tensor[tf.newaxis, ...]
                # input_tensor = np.expand_dims(image_np, 0)
                detections = detect_fn(input_tensor)
                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy()
                              for key, value in detections.items()}
                detections['num_detections'] = num_detections
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
                image_with_detections = frame.copy()
                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes'],
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=200,
                    min_score_thresh=0.5,
                    agnostic_mode=False)
                print('Done')
                cv2.imshow(self.model_name, image_with_detections)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            vid.release()
            cv2.destroyAllWindows()
            self.logger.log(file_object=file, log_message='Streaming done..!!')

        except Exception as e:
            self.logger.log(file_object=file, log_message='Exception occurred')
            self.logger.log(file_object=file, log_message='Exception massage:: {}'.format(e))
            self.logger.log(file_object=file, log_message='Exiting from stream method of Detection class')
            raise e
