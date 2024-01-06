import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sj_utils import *

def plot_confusion_matrix(tp=0, fp=0, tn=0, fn=0, title='Confusion Matrix', save_dir=None):
    """
    Plot the confusion matrix given results.

    :param tp: True Positive
    :param fp: False Positive
    :param tn: True Negative
    :param fn: False Negative
    :param title: Plot title
    :param save_dir: Directory to save the plot (if None, the plot will be displayed but not saved)
    """
    # Create confusion matrix
    conf_matrix = np.array([[fp, tn],
                            [tp, fn]])
    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)

    # Save or show the plot
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'confusion_matrix_plot.png')
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    else:
        plt.show()


# simple function find the label index
def find_label(input_index_string):
    # List of strings
    my_list = ["tactile_tile", "pavement", "grating"]

    label_index = int(input_index_string)

    if 0 <= label_index < len(my_list):
        return my_list[label_index]
    else:
        return "invalid label index"


def get_label_color(index):
    index = int(index)
    # Check if the index is within the valid range (0 to 2 for RGB)
    if 0 <= index <= 2:
        # Return the RGB tuple with 255 at the specified index
        rgb_color = [0, 0, 0]
        rgb_color[index] = 255
        return tuple(rgb_color)
    else:
        # Return a default color if the index is out of range
        return (0, 0, 0)  # Default: black


def get_random_color():
    # Generate a random RGB
    return [int(random.random() * 255) for i in range(3)]


def label_img(color_list, label_dict, label_data, img, is_data_from_detection=False):
    img_height, img_width, _ = img.shape
    for label in label_data:
        # if its detected label, added in detection and confidence
        label_index, x_center, y_center, box_width, box_height, confidence = label
        label_index = int(label_index)
        if is_data_from_detection:
            label_text = f"Detected_{label_dict[label_index]} ({confidence})"
        else:
            label_text = f"{label_dict[label_index]}"

        # Convert YOLO format to OpenCV format
        x1 = int((x_center - box_width / 2) * img_width)
        y1 = int((y_center - box_height / 2) * img_height)
        x2 = int((x_center + box_width / 2) * img_width)
        y2 = int((y_center + box_height / 2) * img_height)

        # Draw bounding box with specific color
        color = color_list[label_index]
        thickness = 2
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return img


def create_combined_labelled_img(image_filepath, label_dict, original_color_list, detected_color_list,
                                 original_label_data, detected_label_data, output_path):
    # image_name = os.path.basename(image_filepath)
    #
    # output_path = os.path.join(output_folder, image_name)

    if os.path.exists(image_filepath):
        # Load the image
        img = cv2.imread(image_filepath)

        img = label_img(original_color_list, label_dict, original_label_data, img)
        img = label_img(detected_color_list, label_dict, detected_label_data, img, is_data_from_detection=True)

        # Save the image with bounding boxes
        cv2.imwrite(output_path, img)
        print(f'image saved at {output_path}')
    print("finished")


# def create_save_labelled_img(image_folder, label_folder, output_labelled_image_folder):
#     list_cur_dir()
#     # prepare the folder
#     prepare_folder(output_labelled_image_folder)
#
#     # List all YOLO label files
#     label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
#
#     for label_file in label_files:
#         image_name = os.path.splitext(label_file)[0] + ".jpg"
#         image_path = os.path.join(image_folder, image_name)
#         if not os.path.exists(image_path):
#             print(f"wrong path, no image {image_name} existed in folder {image_folder}")
#         output_path = os.path.join(output_labelled_image_folder, image_name)
#
#         if os.path.exists(image_path):
#             # Load the image
#             img = cv2.imread(image_path)
#             img_height, img_width, _ = img.shape
#
#             # Read YOLO label file
#             with open(os.path.join(label_folder, label_file), 'r') as label:
#                 for line in label:
#                     parts = line.strip().split()
#                     label_index, x_center, y_center, box_width, box_height = map(float, parts)
#
#                     # Convert YOLO format to OpenCV format
#                     x1 = int((x_center - box_width / 2) * img_width)
#                     y1 = int((y_center - box_height / 2) * img_height)
#                     x2 = int((x_center + box_width / 2) * img_width)
#                     y2 = int((y_center + box_height / 2) * img_height)
#
#                     # Draw bounding box
#                     # Add class label
#                     label_text = f"Class: {find_label(label_index)}"
#                     color = get_label_color(label_index)
#                     thickness = 2
#                     img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
#                     cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
#
#             # Save the image with bounding boxes
#             cv2.imwrite(output_path, img)
#             print(f'image saved at {output_path}')
#     print("finished")


def map_over_graph(csv_file_path):
    df = pd.read_csv(csv_file_path)
    # Print all column names
    print("Column names:")
    index = 0
    for column in df.keys():
        print(f'{index}: #{column}#')
        index += 1
    # Extract relevant columns for plotting

    mAP_values = df["           val/box_loss"]
    epochs = df['                  epoch']

    # Plot the mAP values over epochs
    plt.plot(epochs, mAP_values, marker='o', linestyle='-', color='b')
    plt.title('val/box_loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # map_over_graph("runs/detect/training_109_epoch/results.csv")

    # image_folder = "yolo_data/train/images"
    # label_folder = "yolo_data/train/labels"
    # output_labelled_image_folder = "labelled_images/train/"
    # target_label = "tactile_tile"
    # create_save_labelled_img(image_folder, label_folder, output_labelled_image_folder)
    plot_confusion_matrix(45, 7, 1813, 0, title="trial test tile")