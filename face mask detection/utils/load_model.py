import tensorflow as tf
import time
import utils.label_map_util as label_map_util
from os import path
from logger.app_logger import App_Logger


class Load_model:
    """
        ClassName: Load_model
        Description: This class contain methods for load saved tensorflow object detection model.
        Written By: Tejas Dadhaniya
        Version: 1.0

    """

    def __init__(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        self.min_conf_threshold = float(0.60)
        self.logger = App_Logger()

    def loader(self, model_name):

        """
            ClassName: Load_model
            Method: loader
            Description: This is method for load saved tensorflow object detection model.
            Parameter: model_name - str, saved model directory name present inside models directory.
            Written By: Tejas Dadhaniya
            Version: 1.0
            Return: load model object, category index
            Exception: Raise Exception

        """
        file = open(path.join("logs", "Logs.txt"), 'a+')
        try:
            self.logger.log(file_object=file, log_message='Loading frozen inference graph')
            PATH_TO_MODEL_DIR = path.join('models', model_name)
            PATH_TO_LABELS = path.join('config', 'label_map.pbtxt')
            PATH_TO_SAVED_MODEL = path.join(PATH_TO_MODEL_DIR, "saved_model")
            print('Loading model...', end='')
            start_time = time.time()
            detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print('Done! Took {} seconds'.format(elapsed_time))
            category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
            self.logger.log(file_object=file, log_message='Successfully loading frozen inference graph')
            return detect_fn, category_index

        except Exception as e:
            self.logger.log(file_object=file, log_message='Exception occurred')
            self.logger.log(file_object=file, log_message='Exception massage:: {}'.format(e))
            self.logger.log(file_object=file, log_message='Exiting from loader method of Load_model class')
            raise e
