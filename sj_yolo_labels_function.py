import shutil
import cv2
from sj_utils import *
# from PIL import Image


def generate_empty_label_files(folder_path):
    label_folder = os.path.join(folder_path, "labels")
    img_folder = os.path.join(folder_path, "images")

    # Iterate through all files in the folder
    for filename in os.listdir(img_folder):
        # Create an empty text file with the same name
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_filepath = os.path.join(label_folder, txt_filename)
        if not os.path.exists(txt_filepath):
            with open(txt_filepath, 'w') as txt_file:
                pass  # Writing nothing to create an empty file


def get_all_label_filename(folder_path):
    """
    Give the main data folder, return a list of label filenames
    :param folder_path: folder should contain a subfolder named labels
    :return: a list of strings
    """
    label_folder = os.path.join(folder_path, "labels")

    # get all the labels from both files
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    return label_files


def read_labels_from_file(file_path, have_confident=True):
    labels = []
    try:
        with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 6:
                        class_id, x, y, w, h, confid = map(float, parts)
                    elif len(parts) == 5:
                        class_id, x, y, w, h = map(float, parts)
                        confid = 1
                    labels.append((int(class_id), x, y, w, h, confid))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return labels


# Calculate size (product of width and height) for a YOLO label
def label_size(label):
    return label[4] * label[5]


# get the value for the label
def get_label_index(label):
    return int(label[0])

# change the label inedx, used for false positive data
def set_label_index(new_label_index, label):
    """

    :param new_label_index: must be an integer
    :param label:
    :return:
    """
    list(label)[0] = new_label_index
    return label

def get_label_confid(label, alt_mode=True):
    if alt_mode:
        return float(label[-1])
    else:
        return float(label[0])


def get_label_name(label_index, label_list=None):
    if label_list is None or int(label_index) < 0 or  int(label_index) > len(label_list)-1:
        return str(label_index)
    return label_list[int(label_index)]


# get the x,y,w,h value from the label
def get_label_box(label):
    return label[1:-1]


def generate_random_label_color():
    # Generate random values for red, green, and blue components
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)

    return red, green, blue


def create_save_labelled_img(src_img_folder, cur_folder, labels_list, label_color_list=None,
                             output_labelled_image_folder=None, have_confident=True, alt_mode=True):
    """
    function to create labelled image given the labels and src_imgs
    :param src_img_folder:
    :param label_folder:
    :param output_labelled_image_folder:
    :return:
    """
    if output_labelled_image_folder is None:
        output_labelled_image_folder = os.path.join(cur_folder, "images")
    label_folder = os.path.join(cur_folder, "labels")
    # prepare the folder
    create_folder(output_labelled_image_folder)
    clear_folder(output_labelled_image_folder)

    # generate random color for each label if not provided
    if label_color_list is None:
        label_color_list = [generate_random_label_color() for i in labels_list]

    # List all YOLO label files
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    for label_file in label_files:
        image_name = os.path.splitext(label_file)[0] + ".png"
        image_path = os.path.join(src_img_folder, image_name)
        if not os.path.exists(image_path):
            image_name = os.path.splitext(label_file)[0] + ".jpg"
            image_path = os.path.join(src_img_folder, image_name)
        if not os.path.exists(image_path):
            print(f"error: no images for label files {label_file}")
            continue
        print(image_path)

        output_path = os.path.join(output_labelled_image_folder, image_name)

        if os.path.exists(image_path):
            # Load the image
            img = cv2.imread(image_path)
            img_height, img_width, _ = img.shape

            thickness = max(2, int(min(img_height, img_width) / 1000)) # adaptive label

            labels = read_labels_from_file(os.path.join(label_folder, label_file), have_confident=have_confident)

            for label in labels:
                label_index = get_label_index(label)
                label_name = get_label_name(label_index, labels_list)
                label_confidence = get_label_confid(label, alt_mode=alt_mode)
                color = label_color_list[label_index]

                x_center, y_center, box_width, box_height = get_label_box(label)
                # y_center = 1 - y_center
                print(x_center, y_center, box_width, box_height)
                label_text = f"Class: {label_name} ({label_confidence})"

                # Convert YOLO format to OpenCV format
                x1 = int((x_center - box_width / 2) * img_width)
                y1 = int((y_center - box_height / 2) * img_height)
                x2 = int((x_center + box_width / 2) * img_width)
                y2 = int((y_center + box_height / 2) * img_height)

                print(f"class {label_name}, confidence {label_confidence}, bottom-left {(x1, y1)}, upper-right {(x2, y2)}")


                img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, max(2, int(thickness/2)))

            # Save the image with bounding boxes
            cv2.imwrite(output_path, img)
            print(f'image saved at {output_path}')
    print("Images with bounding boxes have been saved in the output folder.")
    print("Labeled images have been copied to the output folder.")


def calculate_iou(box1, box2, is_only_extension=False, is_original=False):
    # Calculate the intersection over union (IOU) of two bounding boxes
    # print(box1)
    # print(box2)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xA = min(x1 + w1/2, x2 + w2/2)
    xB = max(x1 - w1/2, x2 - w2/2)

    yA = min(y1 + h1/2, y2 + h2/2)
    yB = max(y1 - h1/2, y2 - h2/2)

    inter_area = (xB - xA) * (yB - yA)

    box1_area = w1 * h1
    box2_area = w2 * h2

    if is_original:
      return inter_area / (float(box2_area) + float(box1_area) - inter_area)

    if is_only_extension:
        return inter_area / float(box2_area)

    return max(inter_area / float(box2_area), inter_area / float(box1_area))


def merge_overlapping_labels(labels, threshold=0.8):
    new_labels = []
    removed_labels_index = []

    for i, label in enumerate(labels):
        if i in removed_labels_index:
            continue
        merged = False
        cur_label = label
        for j, other_label in enumerate(labels):
            if j > i and j not in removed_labels_index and get_label_index(cur_label) == get_label_index(other_label):
                iou = calculate_iou(get_label_box(cur_label), get_label_box(other_label), is_only_extension=True)
                # print(f'cur_label: {cur_label} at pos {i} compared to label: {other_label} at pos {j}, overlapped {iou}')
                if iou >= threshold:
                    removed_labels_index.append(j)
                    print(f'label {other_label} at pos {j} is merged')
                    cur_c = get_label_confid(cur_label)
                    cur_x, cur_y, cur_w, cur_h = get_label_box(cur_label)
                    other_c = get_label_confid(other_label)
                    other_x, other_y, other_w, other_h = get_label_box(other_label)

                    left_most_edge = min((cur_x - cur_w / 2), (other_x - other_w / 2))
                    right_most_edge = max((cur_x + cur_w / 2), (other_x + other_w / 2))
                    upper_most_edge = max((cur_y + cur_h / 2), (other_y + other_h / 2))
                    bottom_most_edge = min((cur_y - cur_h / 2), (other_y - other_h / 2))

                    c = max(cur_c, other_c)
                    x = (left_most_edge + right_most_edge) / 2
                    y = (bottom_most_edge + upper_most_edge) / 2
                    w = right_most_edge - left_most_edge
                    h = upper_most_edge - bottom_most_edge
                    cur_label = (label[0], x, y, w, h, c)
        new_labels.append(cur_label)

    return new_labels


def write_labels_to_file(file_path, labels, write_confidence=True):
    with open(file_path, 'w') as file:
        for label in labels:
            if len(label) == 6 and write_confidence:
                file.write(f"{int(label[0])} {label[1]} {label[2]} {label[3]} {label[4]} {label[5]}\n")
            else:
                file.write(f"{int(label[0])} {label[1]} {label[2]} {label[3]} {label[4]}\n")


def create_finalised_detected_data(target_folder, label_list):
    """
    1. perform merging on results
    2. generate the appropriate images
    :param target_folder: located in yolo_detected_data folder, expected to follow the arranagement
    :return:
    """
    original_img_folder = os.path.join(target_folder, "original/images")
    detected_label_folder = os.path.join(target_folder, "detected/labels")
    detected_label_files = [f for f in os.listdir(detected_label_folder) if f.endswith('.txt')]
    for detected_label_file in detected_label_files:
        detected_label_file_path = os.path.join(detected_label_folder, detected_label_file)
        # get the labels from the file
        detected_labels = read_labels_from_file(detected_label_file_path)
        merged_detected_labels = merge_overlapping_labels(detected_labels)
        write_labels_to_file(detected_label_file_path, merged_detected_labels)

    # generate the coprresponding labelled images
    create_save_labelled_img(original_img_folder, os.path.join(target_folder, "detected"), label_list)


def generate_false_positive_training_data(detection_data_path, original_data_path, threshold=0.2,
                                          output_path="false_positive_dataset"):
    """
    Aim is to generate training data based on all the false positives
    :param threshold: a value to determine if the labels overlaps
    :param detection_data_path: labels which were detected by the model
    :param original_data_path: ground truth labels of the same dataset, assume background images have no label file
    :param output_path: folder containing the images and labels for false positives. All labels have same label index: 0.
    :return:
    """

    # in case original data has no empty label file for background iamge
    generate_empty_label_files(original_data_path)


    return output_path


# if __name__ == '__main__':
    # label_list = ["tiles", "cracks", "gap"]
    #
    # # create_finalised_detected_data("../../self_prepared_data/yolo_detected_data/yolo_tiles_crack_gap_model_on_test_images/", label_list)
    # ori_label_list = ["original_tiles", "original_cracks", "original_gap"]
    # target_folder = "../../self_prepared_data/yolo_detected_data/yolo_tiles_crack_gap_model_on_test_images/"
    # # generate a combined labelled image for better visualisation
    # # create_save_labelled_img(os.path.join(target_folder, "detected/images"), os.path.join(target_folder, "original"),
    # #                          ori_label_list,
    # #                          output_labelled_image_folder=os.path.join(target_folder, "final_combined_label_images"),
    # #                          have_confident=False,
    # #                          alt_mode=False)
    # """
    # Aim to find out effectiveness of the detection
    # 1. given specific label index and the target folder
    # """
    # result_folder = "../../self_prepared_data/yolo_detected_data/yolo_tiles_crack_gap_model_on_test_images/"
    # original_label_folder = os.path.join(result_folder, "original/labels")
    # original_label_files = [f for f in os.listdir(original_label_folder) if f.endswith('.txt')]
    # detected_label_folder = os.path.join(result_folder, "detected/labels")
    # detected_label_files = [f for f in os.listdir(detected_label_folder) if f.endswith('.txt')]
    #
    # # for static
    # num_of_original_labels = 0
    # num_of_original_detected = 0
    #
    # tp, tn, fp, fn = 0, 0, 0, 0
    # total_labels = 0
    #
    # for original_label_file in original_label_files:
    #     # if original_label_file == "1 (25).txt":
    #     #     print("testing")
    #     # else:
    #     #     continue
    #     # get the labels from the file
    #     detected_labels = read_labels_from_file(os.path.join(detected_label_folder, original_label_file))
    #     original_labels = read_labels_from_file(os.path.join(original_label_folder, original_label_file),have_confident=False)
    #     threshold = 0
    #     # if cur_fp > 0:
    #     #     print(f"false positive: detection made when there is no original label at img {original_label_file}")
    #     label_index_compared = 1
    #     cur_tp, cur_tn, cur_fp, cur_fn, cur_total_labels = compare_label(original_labels, detected_labels, label_index_compared,
    #                                                            threshold)
    #     if cur_fp > 0:
    #         print(f' false prediction at {original_label_file}')
    #     tp += cur_tp
    #     tn += cur_tn
    #     fp += cur_fp
    #     fn += cur_fn
    #     total_labels += cur_total_labels
    #
    # print(f'tp={tp}, tn={tn}, fp={fp}, fn={fn}, total_labels={total_labels}')
    #     # pipeline for overall defect detection
    #
    #     # check for cracks
    #     if has_defects(original_labels):
    #         total_labels += 1
    #
    #     if has_defects(original_labels) and has_defects(detected_labels): # has defects, detected defects
    #         tp += 1
    #     if not has_defects(original_labels) and has_defects(detected_labels): # no defects, has detects
    #         fp += 1
    #     if has_defects(original_labels) and not has_defects(detected_labels): # has defects, no detects
    #         fn += 1
    #         print(f'overall false detection at {original_label_file}')
    #     if not has_defects(original_labels) and not has_defects(detected_labels): # no defects, no detects
    #         tn += 1
    #
    #     print(f'tp={tp}, tn={tn}, fp={fp}, fn={fn}, total_labels={total_labels}')
    #
    # print(f'total labels detected {num_of_original_detected} / {num_of_original_labels}')
