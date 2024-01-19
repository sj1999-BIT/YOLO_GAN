import numpy as np

from ultralytics import YOLO
from sj_yolo_labels_function import *
from sj_data_visualisation import *


def yolo_model_testing(trained_weights_file_path, testing_img_folder_path, imgSize=640, conf=0.15):
    """
    Load the weights and run it on the target dataset
    :param trained_weights_file_path: path to the weights
    :param testing_img_folder_path: path to the folder containing images to be run on
    :return:
    """
    # load the weights into the yolo model
    trained_model = YOLO(trained_weights_file_path)
    # use API to save all labels, confidence of the model detection
    result = trained_model(testing_img_folder_path, save_txt=True,
                           show_boxes=True, save=True, save_conf=True, conf=conf, imgsz=imgSize)

    return result[0].save_dir, result[0].names


def label_stats(label_dict, data_folder_path, have_confident=True):
    """
    Provide a summary of how many labels present in total in the given files
    :param label_dict: a dictionary mapping index to label
    :param data_folder_path: folder should contain a subfolder named labels
    :param have_confident: detected labels have confident, while ground truth labels dont
    :return: None
    """

    src = "TRAINING"
    if have_confident:
        src = "TESTING"

    label_count = [0 for i in label_dict]
    background = 0
    label_files = get_all_label_filename(data_folder_path)
    for label_file in label_files:
        if label_file == "labels.txt":
            # there is a labels.txt file which do not contain labels
            continue
        label_file_path = os.path.join(data_folder_path, "labels", label_file)
        label_list = read_labels_from_file(label_file_path, have_confident)

        if len(label_list) == 0:
            background += 1

        for label in label_list:
            cur_label_index = get_label_index(label)
            label_count[cur_label_index] += 1


    print(f"SUMMARY FOR EACH {src} LABEL")
    for index in range(len(label_count)):
        cur_label = label_dict[index]
        cur_count = label_count[index]
        print(f"there is a total of {cur_count} instance of {cur_label}")
    print(f"there is a total of {background} background images")
    print()


def is_label_overlapped(cur_label, label_list, threshold=0.4):
    """
    Check if the cur_label overlap with any of the labels in the label_list that shares the same label index.
    :param cur_label: single label
    :param label_list: a list of labels
    :param alt_mode:
    :return:
    """
    cur_label_index = get_label_index(cur_label)
    for label in label_list:
        if get_label_index(label) != cur_label_index:
            # only check IOU if they have the same label index
            continue

        # get the dimension data for each label
        cur_label_box = get_label_box(cur_label)
        compare_label_box = get_label_box(label)

        if calculate_iou(cur_label_box, compare_label_box) > threshold:
            # if IOU > threshold, its overlapped
            return True

    # none of the labels in the label_list overlaps sufficiently with the cur_label
    return False


def label_to_label_data_analysis(filtered_original_label_list, filtered_detect_label_list, threshold=0.4):
    """
    Function compares a single label data to another single label data to obtain statistic results
    :param filtered_original_label_list: label_list containing ground truth labels with same label index
    :param filtered_detect_label_list: label_list containing model predicted labels with same label index
    :param threshold: To decide if IOU is significant to be considered overlap between 2 labels
    :return: tp, tn, fp, fn for this 2 label_data
    """
    # keep record of the instances
    tp = 0  # true positive, if detected label IOU with original label pass threshold
    tn = 0  # true negative, if no original label and no detected label
    fp = 0  # false positive, if no original label IOU with detected label pass threshold
    fn = 0  # false negative, if have original label, but no detected label

    for original_label in filtered_original_label_list:
        # # we only care about labels under we target to find
        # if get_label_index(original_label) != target_label_index:
        #     continue
        if is_label_overlapped(original_label, filtered_detect_label_list, threshold):
            # there is an overlap in the detected_label, its a true positive
            tp += 1
        else:
            # there is no detection made
            fn += 1

    for detected_label in filtered_detect_label_list:
        if not is_label_overlapped(detected_label, filtered_original_label_list, threshold):
            # any detection not overlapped is false positive
            fp += 1

    if len(filtered_detect_label_list) == 0 and fn == 0:
        # no detection and no original label under what we wanted, so we give it a true negative
        tn += 1



    return tp, fp, tn, fn


def target_label_index_data_analysis(label_dict, target_label_index, original_data_path,  original_label_files,
                                     detection_data_path, detected_label_files, threshold=0.4, save_labelled_img=False):
    """
    Given a target label index, provide the overall statistic analysis for all the label_datas
    regarding the target label.
    :param target_label_index: int for the
    :param original_label_files:
    :param detected_label_files:
    :param threshold:
    :return:
    """

    if target_label_index not in label_dict.keys():
        print(f"invalid target index {target_label_index}, please choose index from {label_dict.keys()}")
        return

    if save_labelled_img:
        # create folder to contain all the invalid images
        prepare_error_labelled_img_folder(detection_data_path, label_dict[target_label_index], is_log_printing=True)
        # create the colour scheme for both original and detected
        original_color_scheme = [get_random_color() for i in range(len(label_dict.keys()))]
        detected_color_scheme = [get_random_color() for i in range(len(label_dict.keys()))]

    # # path to folder containing images for testing
    # img_folder_path = os.path.join(define_original_data_path, "images")


    # keep record of the instances
    tp = 0  # true positive, if detected label IOU with original label pass threshold
    tn = 0  # true negative, if no original label and no detected label
    fp = 0  # false positive, if no original label IOU with detected label pass threshold
    fn = 0  # false negative, if have original label, but no detected label

    for original_label_file in original_label_files:

        if original_label_file == "labels.txt":
            continue

        # get the original labels
        original_label_file_path = os.path.join(original_data_path, "labels", original_label_file)
        full_original_label_list = read_labels_from_file(original_label_file_path, have_confident=False)
        # we only want labels that have the target_label
        original_label_list = [label for label in full_original_label_list
                               if get_label_index(label) == target_label_index]

        if original_label_file in detected_label_files:
            # get the detected labels
            detect_label_file_path = os.path.join(detection_data_path, "labels", original_label_file)
            full_detect_label_list = read_labels_from_file(detect_label_file_path)
            # we only want labels that matter to us
            detect_label_list = [label for label in full_detect_label_list
                                 if get_label_index(label) == target_label_index]
        else:
            full_detect_label_list = []
            # empty list indicates no detection
            detect_label_list = []

        cur_tp, cur_fp, cur_tn, cur_fn = label_to_label_data_analysis(original_label_list, detect_label_list, threshold)

        tp += cur_tp
        fp += cur_fp
        tn += cur_tn
        fn += cur_fn

        if not save_labelled_img:
            # no need to do following if save_img flag not on
            continue

        original_img_filename = get_img_from_label(original_label_file, os.path.join(original_data_path, "images"),
                                               is_log_printing=False)
        target_input_filepath = os.path.join(original_data_path, "images", original_img_filename)

        if cur_fp > 0:

            target_output_filepath = os.path.join(detection_data_path, "invalid_labelled_images",
                                                  label_dict[target_label_index],
                                                  "false_positives", original_img_filename)

            create_combined_labelled_img(target_input_filepath, label_dict, original_color_scheme,
                                         detected_color_scheme, full_original_label_list, full_detect_label_list,
                                         target_output_filepath)
            # copy data over
            # shutil.copy(target_input_filepath, target_output_filepath)

        if cur_fn > 0:
            target_output_filepath = os.path.join(detection_data_path, "invalid_labelled_images",
                                                  label_dict[target_label_index],
                                                  "false_negatives", original_img_filename)

            create_combined_labelled_img(target_input_filepath, label_dict, original_color_scheme,
                                         detected_color_scheme, full_original_label_list, full_detect_label_list,
                                         target_output_filepath)
            # # copy data over
            # shutil.copy(target_input_filepath, target_output_filepath)

    print(f'label: {label_dict[target_label_index]} has {tp} tp, {fp} fp, {tn} tn, {fn} fn')
    plot_confusion_matrix(tp, fp, tn, fn, title=label_dict[target_label_index] + "_confusion matrix",
                          save_dir=os.path.join(detection_data_path, "invalid_labelled_images",
                                                  label_dict[target_label_index]))
    return tp, fp, tn, fn


def has_defects(labels, non_defects_list=[]):
    """
    return True if crack/gap labels present in the label
    :param labels: a list of labels read from a single file
    :return:
    """
    for label in labels:
        if get_label_index(label) not in non_defects_list:
            return True
    return False


def defects_data_analysis(label_dict, original_data_path, original_label_files, detection_data_path,
                          detected_label_files, non_defects_list=[]):
    """
    Function specifically made for lighthaus usage.
    Only determine if the model detected any form of defects (label index > 0) and if original label has any defects.
    No need for IOU.
    :param original_label_files:
    :param detected_label_files:
    :return:
    """

    # create folder to contain all the invalid images
    prepare_error_labelled_img_folder(detection_data_path, "defects", is_log_printing=True)
    # create the colour scheme for both original and detected
    original_color_scheme = generate_random_colour_scheme_for_labels(label_dict)
    detected_color_scheme = generate_random_colour_scheme_for_labels(label_dict)

    # keep record of the instances
    tp = 0  # true positive, if detected label IOU with original label pass threshold
    tn = 0  # true negative, if no original label and no detected label
    fp = 0  # false positive, if no original label IOU with detected label pass threshold
    fn = 0  # false negative, if have original label, but no detected label
    for original_label_file in original_label_files:
        original_img_filename = get_img_from_label(original_label_file, os.path.join(original_data_path, "images"),
                                                   is_log_printing=False)
        target_input_filepath = os.path.join(original_data_path, "images", original_img_filename)
        # get the original labels
        original_label_file_path = os.path.join(original_data_path, "labels", original_label_file)
        original_label_list = read_labels_from_file(original_label_file_path, have_confident=False)

        # get the detected labels
        if original_label_file in detected_label_files:
            detect_label_file_path = os.path.join(detection_data_path, "labels", original_label_file)
            detect_label_list = read_labels_from_file(detect_label_file_path)
        else:
            # empty list indicates no detection
            detect_label_list = []

        # check if there are any defects in both label_list regarding the same image
        is_image_defected = has_defects(original_label_list, non_defects_list=non_defects_list)
        is_defect_predicted = has_defects(detect_label_list, non_defects_list=non_defects_list)

        if is_image_defected and is_defect_predicted:
            tp += 1
        if is_image_defected and not is_defect_predicted:
            fn += 1
            target_output_filepath = os.path.join(detection_data_path, "invalid_labelled_images",
                                                  "defects",
                                                  "false_negatives", original_img_filename)

            create_combined_labelled_img(target_input_filepath, label_dict, original_color_scheme,
                                         detected_color_scheme, original_label_list, detect_label_list,
                                         target_output_filepath)
        if not is_image_defected and is_defect_predicted:
            fp += 1
            target_output_filepath = os.path.join(detection_data_path, "invalid_labelled_images",
                                                  "defects",
                                                  "false_positives", original_img_filename)

            create_combined_labelled_img(target_input_filepath, label_dict, original_color_scheme,
                                         detected_color_scheme, original_label_list, detect_label_list,
                                         target_output_filepath)
        if not is_image_defected and not is_defect_predicted:
            tn += 1
    print(f'overall defect detection has {tp} tp, {fp} fp, {tn} tn, {fn} fn')
    plot_confusion_matrix(tp, fp, tn, fn, title="defects_confusion matrix",
                          save_dir=os.path.join(detection_data_path, "invalid_labelled_images",
                                                  "defects"))
    return tp, fp, tn, fn


def yolo_testing_pipeline(trained_weights_filepath, ground_truth_test_data_path,
                          threshold=0.4, save_labelled_img=True, imgSize=640, non_defects_list=[], conf=0.15):

    """
    The yolo_testing_pipeline function performs testing and analysis on a YOLO model.
    It takes as input the necessary parameters such as label dictionary, trained weights file path,
    and the path to the test data. It provides various analysis results, including label statistics,
    target-specific data analysis, and overall defect analysis.
    :param trained_weights_filepath:  Path to the trained weights file of the YOLO model.
    :param ground_truth_test_data_path: Path to the folder containing the original testing data file.
                                        It should have subdirectories images and labels.
    :param threshold: Confidence threshold for detections.
    :param save_labelled_img: Flag to save images with labeled predictions.
    :param imgSize: Size of the input images. Must be equal to the img_size used for weight training.
    :param non_defects_list: Specific to PathWatcher. Certain labels not considered defects should add their index here.
    :return:
    """

    # testing folder may not have equal labels files and images files as background images may not have label file.
    generate_empty_label_files(ground_truth_test_data_path)

    # path to folder containing images for testing
    testing_img_folder_path = os.path.join(ground_truth_test_data_path, "images")

    # get the labels and file path of the output
    detection_data_path, label_dict = yolo_model_testing(trained_weights_filepath, testing_img_folder_path, imgSize=imgSize, conf=conf)

    # get all the labels from both original and detection
    original_label_files = get_all_label_filename(ground_truth_test_data_path)

    # sometimes such files exists as label websites such as Makesense.ai requires it to import labels.
    if "labels.txt" in original_label_files:
        original_label_files.remove("labels.txt")

    detected_label_files = get_all_label_filename(detection_data_path)

    # # provide overall information of the data
    label_stats(label_dict, ground_truth_test_data_path, have_confident=False)
    label_stats(label_dict, detection_data_path)

    for target_label_index in label_dict.keys():
        tp, fp, tn, fn = target_label_index_data_analysis(label_dict, target_label_index,
                                                          ground_truth_test_data_path, original_label_files,
                                                          detection_data_path, detected_label_files,
                                                          threshold=threshold, save_labelled_img=save_labelled_img)

    # test overall defect analysis
    tp, fp, tn, fn = defects_data_analysis(label_dict, ground_truth_test_data_path, original_label_files,
                                           detection_data_path, detected_label_files, non_defects_list=non_defects_list)

    return detection_data_path

# for testing purpose, should be able to output results of each label for target model given weights.
if __name__ == '__main__':

    # first find the path to the file
    trained_weights_file_path = "trained_usable_weights/tactile_tiles_yolo_detector_109epochs_weights.onnx"

    # use the allocated testing data
    define_original_data_path = "data/train_sample_data/test"

    # run the tresting pipeline
    yolo_testing_pipeline(trained_weights_file_path, define_original_data_path,
                          threshold=0.4, save_labelled_img=True, imgSize=640, non_defects_list=[0])

