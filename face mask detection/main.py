"""
    ModuleName: main
    Version: 1.0
    Date: 31 may, 2021
    Written by: Tejas Dadhaniya
    Python version: 3.6

"""

from utils.detection import Detection

detection = Detection(model_name='centernet_mobilenet_320_v2')  # centernet_mobilenet_320_v2
detection.stream()
