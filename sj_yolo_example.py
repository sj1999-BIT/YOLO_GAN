import numpy as np

from ultralytics import YOLO
from sj_yolo_labels_function import *
from sj_data_visualisation import *
from sj_yolo_testing_pipeline import *
from sj_yolo_training_pipeline import *

if __name__ == '__main__':
    # given the folder containing the original data: images and labels 2 subfolders
    original_datapath = "data/sample_labelled_data"
    # create the folder for training data
    trainable_datapath = "data/trainable_data"
    # define dict for current labels
    define_label_dict = {0: 'tiles', 1: 'cracks', 2: 'empty', 3: 'chipped_off'}

    # train a yolo model for 1 epoch(example) using pre-trained weights (yolov8n.pt)
    # and export the model to ONNX format
    trained_weights_file_path = yolo_training_pipeline(define_label_dict, original_datapath, trainable_datapath,
                                                       epochs=1, initial_model_weight='yolov8n.pt',
                                                       output_format="onnx")

    # run testing on the trainined yolo model to obtain full results
    yolo_testing_pipeline(define_label_dict, trained_weights_file_path, trainable_datapath,
                          threshold=0.4, save_labelled_img=True)





