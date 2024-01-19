from sj_utils import *
from sj_yolo_labels_function import *
from ultralytics import YOLO


def full_yolo_data_prep_pipeline(label_dict, original_data_path, trainable_data_path):
    prepare_yolo_data_folder(trainable_data_path, is_log_printing=False)
    # separate the data
    separate_data(original_data_path, trainable_data_path, val_percent=10.0, test_percent=10.0, is_log_printing=True)
    # create the required txt_files for yolo training
    prep_txt_file_for_yolo(trainable_data_path, is_log_printing=False)
    # Get the absolute path
    absolute_train_data_path = os.path.abspath(trainable_data_path)
    # create yaml file for training, return the yaml file path as its needed for training
    yaml_file_path = create_yaml_file(absolute_train_data_path, label_dict, is_log_printing=True)
    return yaml_file_path


def yolo_training_pipeline(define_label_dict, original_data_path, epochs=300,
                           img_size=640, initial_model_weight='yolov8n.pt', output_format="onnx"):
    """
    function to train a yolo based on a simple data folder
    :param define_label_dict: a dict mapping label index to labels.
                              E.g. {0: 'tiles', 1: 'cracks', 2: 'empty', 3: 'chipped_off'}
    :param original_data_path: data path to folder containing original data arranged as such.
                                            └── images: folder contain all images for training
                                            └── labels: ground truth labels, not all images need a corresponding label.
    :param epochs: determine number of epoch for training (optional).
    :param img_size: determine input size nXn of images. All Images will be transformed to this size (optional).
    :param initial_model_weight: string for initialised weight of model (optional).
    :param output_format: format of output weight (optional).
    :return: path to dir where the best trained weights are stored.
    """
    # Generate the new folder name
    new_folder_name = f'train_{os.path.basename(original_data_path)}'

    # Combine the new folder name with the parent directory to get the full path
    trainable_data_path = os.path.join(os.path.dirname(original_data_path), new_folder_name)

    # prepare the data for model training, get the yaml file path as its needed for training
    yaml_file_path = full_yolo_data_prep_pipeline(define_label_dict, original_data_path, trainable_data_path)

    model = YOLO(initial_model_weight)  # load a pretrained model (recommended for training)
    # Train the model
    # normally training is done within 300epoch
    results = model.train(data=yaml_file_path, epochs=epochs, imgsz=img_size)
    # export the model to ONNX format
    trained_weight_path = model.export(format=output_format)

    return trained_weight_path

if __name__ == '__main__':
    # given the folder containing the original data: images and labels 2 subfolders
    original_datapath = "C:/Users/Shuij/OneDrive/Documents/GitHub/YOLO_GAN\data/CS4243_labelled_weap_data"

    # define the label dictionary for this dataset
    define_label_dict = {0: 'weapon'}

    # # export the model to ONNX format
    trained_yolo_weights_filepath = yolo_training_pipeline(define_label_dict, original_datapath, img_size=640,
                                                           initial_model_weight='yolov4_tiny.pt')

    print(f'path is {trained_yolo_weights_filepath}')





