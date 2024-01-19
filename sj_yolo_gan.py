import subprocess

from sj_yolo_training_pipeline import *
from sj_yolo_testing_pipeline import *


def get_recall(trained_weight_path, test_data_folder_path, conf=0.1, imgSize=640):
    # load the weights into the yolo model
    trained_model = YOLO(trained_weight_path)

    # path to folder containing images for testing
    # testing_img_folder_path = os.path.join(test_data_folder_path, "images")
    testing_img_folder_path = "data/norm_images"
    # use API to save all labels, confidence of the model detection
    result = trained_model(testing_img_folder_path, save_txt=True,
                           show_boxes=True, save=True, save_conf=True, conf=conf, imgsz=imgSize)
    return result[0].save_dir, result[0].names


def generate_false_positive_labels(gt_label_data, detect_label_data, threshold=0.2):
    fp_label_data = []
    fp_label_index = 0  # false positive data only has one label

    for cur_detected_label in detect_label_data:
        is_false_positive = True
        for cur_gt_label in gt_label_data:
            # no need to compare if not same label index
            if get_label_index(cur_gt_label) != get_label_index(cur_detected_label):
                continue

            # if exist ground turht label overlap with detection, its not a false positive
            if calculate_iou(get_label_box(cur_detected_label), get_label_box(cur_gt_label)) >= threshold:
                is_false_positive = False
                break

        # collect the false positive data
        if is_false_positive:
            fp_label_data.append(set_label_index(fp_label_index, cur_detected_label))

    return fp_label_data


def generate_supervised_data(trained_weight_path, ground_truth_data_path, output_data_path="supervised_data",
                             conf=0.1, imgSize=640):

    # create the folder if its not existed, folder will contain images and labels subfolder
    prepare_data_folder(output_data_path)

    # load the weights into the yolo model
    trained_model = YOLO(trained_weight_path)

    # path to folder containing images for testing
    # testing_img_folder_path = os.path.join(test_data_folder_path, "images")
    testing_img_folder_path = os.path.join(ground_truth_data_path, "images")

    # use API to save all labels, confidence of the model detection
    result = trained_model(testing_img_folder_path, save_txt=True,
                           show_boxes=True, save=True, save_conf=True, conf=conf, imgsz=imgSize)

    # get the detected results
    detection_data_path = result[0].save_dir

    # we need to go through every image to compare results.
    img_files = os.listdir(testing_img_folder_path)

    for cur_img_filename in img_files:
        target_label_filename = get_labels_file_name(cur_img_filename)

        # read labels from ground truth, returns zero
        cur_gt_labels_path = os.path.join(ground_truth_data_path, "labels", target_label_filename)
        cur_ground_truth_label_data = read_labels_from_file(cur_gt_labels_path)

        # read labels from detection
        cur_detect_labels_path = os.path.join(detection_data_path, "labels", target_label_filename)
        cur_detect_label_data = read_labels_from_file(cur_detect_labels_path)

        # detect positive data
        cur_fp_label_data = generate_false_positive_labels(cur_ground_truth_label_data, cur_detect_label_data)

        # no need to save if no false positive is detected
        if len(cur_fp_label_data) == 0:
            continue


        # path to store the new data
        fp_img_path = os.path.join(output_data_path, "images", cur_img_filename)
        fp_label_path = os.path.join(output_data_path, "labels", target_label_filename)

        # copy image over
        cur_img_path = os.path.join(testing_img_folder_path, cur_img_filename)
        shutil.copy(cur_img_path, fp_img_path)

        # generate labels for training, no need confidence
        write_labels_to_file(fp_label_path, cur_fp_label_data, write_confidence=False)


        print(f'false positives {cur_fp_label_data}')

    return output_data_path


def yolo_gan_analysis_labels(positive_label_data, negative_label_data, thres_overlap=0.2, thres_conf=3):
    final_label_data = []
    for positive_label in positive_label_data:
        is_label_removed=False
        for negative_label in negative_label_data:
            cur_iou = calculate_iou(get_label_box(positive_label), get_label_box(negative_label))
            cur_conf_multi = get_label_confid(positive_label) / get_label_confid(negative_label)

            if cur_iou > thres_overlap and cur_conf_multi < thres_conf:
                is_label_removed = True
                break

        if not is_label_removed:
            final_label_data.append(positive_label)

    return final_label_data



def yolo_supervise_model_training_pipeline(supervised_data_path, epoch=300):
    define_label_dict = {0: "false_positive"}
    return yolo_training_pipeline(define_label_dict, supervised_data_path, epochs=epoch)


def yolo_gan_prediction(trained_yolo_positive_weight_path, trained_yolo_supervise_weight_path,
                        target_img_folder_path, output_path=None, print_log=False, confidence=0.2,
                        imgSize=640):

    if output_path is None:
        # Generate the new folder name
        new_folder_name = f'prediction_{os.path.basename(target_img_folder_path)}'

        # Combine the new folder name with the parent directory to get the full path
        output_path = os.path.join(os.path.dirname(target_img_folder_path), new_folder_name)

    # prepare a data folder to contain all images and labels with overall detection
    prepare_data_folder(output_path, is_log_printing=print_log)

    # copy all images over
    copy_img_folder(target_img_folder_path, os.path.join(output_path, "images"))

    # load the models
    yolo_positive_model = YOLO(trained_yolo_positive_weight_path)
    yolo_supervise_model = YOLO(trained_yolo_supervise_weight_path)

    # # let the positive model make predictions on the target img and get the detections
    # positive_results = yolo_positive_model(target_img_folder_path, save_txt=True, show_boxes=True,
    #                                        save=True, save_conf=True, conf=confidence, imgsz=imgSize)
    # positive_detection_data_path = os.path.join(positive_results[0].save_dir, "labels")
    #
    # # let the supervised model make predictions on the target img and get the detections
    # negative_results = yolo_supervise_model(target_img_folder_path, save_txt=True, show_boxes=True,
    #                                        save=True, save_conf=True, conf=confidence, imgsz=imgSize)
    # negative_detection_data_path = os.path.join(negative_results[0].save_dir, "labels")

    positive_detection_data_path = os.path.join("runs\detect\predict20", "labels")
    negative_detection_data_path = os.path.join("runs\detect\predict21", "labels")

    # we only care if positive detections are valid, so we only go through all the
    for positive_detection_label_file in os.listdir(positive_detection_data_path):

        # get all positive labels
        positive_label_filepath = os.path.join(positive_detection_data_path, positive_detection_label_file)
        positive_label_data = read_labels_from_file(positive_label_filepath)

        # get all negative labels
        negative_label_filepath = os.path.join(negative_detection_data_path, positive_detection_label_file)
        negative_label_data = read_labels_from_file(negative_label_filepath)

        # compare 2 model results to determine the final label
        final_label_data = \
            yolo_gan_analysis_labels(positive_label_data, negative_label_data, thres_overlap=0.2,
                                     thres_conf=3)

        new_label_path = os.path.join(output_path, "labels", positive_detection_label_file)

        if len(final_label_data) > 0:
            write_labels_to_file(new_label_path, final_label_data)


def yolo_gan_training_pipeline(define_label_dict, original_data_path, supervised_data_path=None, epochs=300):

    # first train a yolo model based on the original data
    trained_yolo_positive_weight_path = \
        yolo_training_pipeline(define_label_dict, original_data_path, epochs=epochs,
                               img_size=640, initial_model_weight='yolov8n.pt')

    # if no images for supervise leanring, test model on original training images to get false positives
    if supervised_data_path is None:
        supervised_data_path = os.path.join(original_data_path, "train", "images")

    # generate supervised data
    supervised_data_path = generate_supervised_data(trained_yolo_positive_weight_path,
                                                    supervised_data_path)

    # train the supervised model
    trained_yolo_supervise_weight_path = \
        yolo_supervise_model_training_pipeline(supervised_data_path, epoch=epochs)

    return trained_yolo_positive_weight_path, trained_yolo_supervise_weight_path



if __name__ == '__main__':
    # given the folder containing the original data: images and labels 2 subfolders
    original_datapath = "data/weap_training_data"

    # define the label dictionary for this dataset
    define_label_dict = {0: 'weapon'}

    # train models
    positive_yolo_path, negative_yolo_path = \
        yolo_gan_training_pipeline(define_label_dict, original_datapath,
                                   supervised_data_path="data/norm_training_data")

    # folder contains image for testing
    test_data_folder = "data/train_weap_training_data/test/images"

    # test the overall capability
    yolo_gan_prediction(positive_yolo_path, negative_yolo_path,
                        test_data_folder, output_path=None, print_log=True, confidence=0.2,
                        imgSize=640)


    # trained_yolo_positive_weight_path = "trained_weights/train_weap_training_data/yolo_weap_90.pt"
    # trained_yolo_supervise_weight_path = "trained_weights/train_weap_not_model_data/yolo_not_weap_90.pt"
    # test_data_folder = "data/weap_training_data/images"

    # get_recall(trained_weights_path, test_data_folder)

    # supervised_data_path = generate_supervised_data(trained_weights_path, test_data_folder)
    #
    # supervised_trained_weight_path = \
    #     yolo_supervise_model_training_pipeline(supervised_data_path, epoch=100)

    # yolo_gan_prediction(trained_yolo_positive_weight_path, trained_yolo_supervise_weight_path,
    #                     test_data_folder, output_path=None, print_log=True, confidence=0.2,
    #                     imgSize=640)


    # original_data_path = "data/weap_training_data"
    #
    # # define the label dictionary for this dataset
    # define_label_dict = {0: 'weapon'}
    #
    # epoch_num = 0
    #
    # epoch_interval = 100
    #
    # trained_yolo_weights_filepath = 'yolov8n.pt'
    #
    # # Generate the new folder name
    # new_folder_name = f'train_{os.path.basename(original_data_path)}'
    #
    # # Combine the new folder name with the parent directory to get the full path
    # trainable_data_path = os.path.join(os.path.dirname(original_data_path), new_folder_name)
    #
    # # prepare the data for model training, get the yaml file path as its needed for training
    # yaml_file_path = full_yolo_data_prep_pipeline(define_label_dict, original_data_path, trainable_data_path)
    # # yaml_file_path = "data/train_CS4243_labelled_weap_data/lighthaus_data.yaml"

      # load a pretrained model (recommended for training)
    # Train the model
    # normally training is done within 300epoch

    # save_weight_name = "yolo_positive_"
    #
    # save_weights_folder = os.path.join("trained_weights", new_folder_name)
    #
    # epoch_num += epoch_interval
    #
    # create_folder(save_weights_folder)
    #
    # model = YOLO(trained_yolo_weights_filepath)
    #
    # results = model.train(data=yaml_file_path, epochs=epoch_interval, batch=16)
    #
    # saved_weights_path = os.path.join(results.save_dir, "weights", "last.pt")
    #
    # saved_weights_name = f"yolo_positive_{epoch_num}.pt"
    #
    # trained_yolo_weights_filepath = os.path.join(save_weights_folder, saved_weights_name)
    #
    # shutil.copy(saved_weights_path, trained_yolo_weights_filepath)
    #
    #
    # #
    # # print(f'path is {trained_yolo_weights_filepath}')
    #
    # testing_datapath = "data/train_CS4243_labelled_weap_data/test"
    #
    # current_weights_path = "trained_weights/actual_model/last.pt"
    #
    # yolo_testing_pipeline(current_weights_path, testing_datapath,
    #                       threshold=0.2, save_labelled_img=True, imgSize=640, non_defects_list=[0])
    #

