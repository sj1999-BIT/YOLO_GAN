import numpy as np

from ultralytics import YOLO
from sj_yolo_labels_function import *
from sj_data_visualisation import *
from sj_yolo_testing_pipeline import *


def yolo_application_pipeline(cv2_read_img, model_weight_filepath, save_label=False,
                              save_labelled_img=False, show_labelled_img=False, out_path=None, img_filename=None):
    """
    API for Pathwatcher to apply the trained model weights into detection given an cv2 read image.
    :param cv2_read_img: target image read using cv2.
    :param model_weight_filepath: path to the trained yolo weights files.
    :param save_label: determine if txt file containing the label should be saved (optional).
    :param save_labelled_img: determine if the resulting labelled img to be saved (optional).
    :param show_labelled_img: determine labelled img result to be shown (optional).
    :param out_path: data path for saving the labels, will save to default folder if not defined (optional).
    :param img_filename: filename for the saved label image.
    :return: arr of labels, each label is a size 6 1d arr.
             label values arranged in (label_index, x, y, w, h, confidence)
             x, y : normalized centre coordinates of the label box
             w, h: normalised width and height of the label box
             label_index: can be used to find the correct label using pre-defined label_dict.
    """

    create_folder(out_path, is_log_printing=True)
    # Load a pretrained YOLOv8n model
    model = YOLO(model_weight_filepath)

    # Run inference on the source
    results = model(cv2_read_img, save_txt=True, save_conf=True)  # list of Results objects

    label_saved_dir = results[0].save_dir

    label_file = get_all_label_filename(label_saved_dir)[0]  # should only be one file

    label_data_list = read_labels_from_file(os.path.join(label_saved_dir, "labels", label_file))

    if not save_label:
        remove_folder(label_saved_dir)
    elif out_path is not None:
        write_labels_to_file(os.path.join(out_path, label_file), label_data_list)

    if not save_labelled_img and not show_labelled_img:
        return label_data_list

    label_dict = results[0].names

    print(f'detected labels are {label_data_list}, with label_dict {label_dict}')

    labelled_img = label_img(label_dict, label_data_list, cv2_read_img)

    if show_labelled_img:
        cv2.imshow('labelled Image', labelled_img)
        # Wait for a key event and close the window when a key is pressed
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_labelled_img:
        if img_filename is None:
            img_filename = "labelled_image.jpg"
        cv2.imwrite(os.path.join(out_path, img_filename), labelled_img)

    return label_data_list


# for testing purpose, should be able to output results of each label for target model given weights.
if __name__ == '__main__':
    # first find the path to the file
    trained_weights_file_path = "trained_usable_weights/tactile_tiles_yolo_detector_109epochs_weights.onnx"

    # read the image from cv2
    cv2_read_image = cv2.imread("data/sample_data/images/1 (1).jpg")

    # apply the image to the trained model for prediction
    yolo_application_pipeline(cv2_read_image,
                              trained_weights_file_path,
                              save_labelled_img=True,
                              show_labelled_img=True,
                              save_label=True,
                              out_path="testing_new_folder",
                              img_filename="testing.jpg")




