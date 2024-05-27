import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np


def retrieve_image_paths(root_dir: str):
    """
    This function recursively searches a directory for image files (JPG, JPEG, PNG) and their corresponding label files (TXT). 
    It returns a list of tuples, where each tuple contains the file path of an image and its corresponding label file.

    Args:
        root_dir (str): The path to the directory to search.

    Returns:
        list of tuples: A list of tuples, where each tuple contains the file path of an image and its corresponding label file.
    """
    image_label_pairs = []

    image_files = []
    label_files = []

    for dirname, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(dirname, filename))
            elif filename.lower().endswith('.txt'):
                label_files.append(os.path.join(dirname, filename))

    label_mapping = {os.path.splitext(os.path.basename(label_file))[0]: label_file for label_file in label_files}

    for image_file in image_files:
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        label_file = label_mapping.get(image_name)
        if label_file:
            image_label_pairs.append((image_file, label_file))
    return image_label_pairs


def parse_annotations(label_file_path: str):
    """
    This function reads a label file and parses the annotations contained within it. It returns a list of dictionaries, where each
    dictionary represents an annotation and contains the class ID, center coordinates, width, and height of the annotated object.

    Args:
        label_file_path (str): The file path of the label file to parse.

    Returns:
        list of dicts: A list of dictionaries, where each dictionary represents an annotation.
    """
    annotations = []
    with open(label_file_path, 'r') as file:
        for line in file:
            components = line.strip().split()
            if components:
                annotation = {
                    'class_id': int(components[0]),
                    'x_center': float(components[1]),
                    'y_center': float(components[2]),
                    'width': float(components[3]),
                    'height': float(components[4])
                }
                annotations.append(annotation)
    return annotations

def read_images_and_annotations(image_file_path: str, label_file_path: str):
    """
    This function reads an image file and its corresponding label file, parses the annotations from the label file, and returns the image and the list of annotations.

    Args:
        image_file_path (str): The file path of the image file.
        label_file_path (str): The file path of the label file.

    Returns:
        tuple: A tuple containing the image (numpy array) and the list of annotations (list of dictionaries).
    """
    image = cv2.imread(image_file_path)
    
    if os.path.getsize(label_file_path) == 0:
        return image, []
    
    annotations = parse_annotations(label_file_path)
    
    return image, annotations



def display_annotated_images(image_label_pairs: list[tuple[str, str]], 
                             n_cols: int = 5, n_rows: int = 5,
                             num_images: int = 25, figsize_: tuple[int, int] = (20, 20)):
    """
    This function displays a set of randomly selected images along with their corresponding annotations. The annotations are displayed as bounding boxes on the images, and the class names (if provided) are shown inside the boxes.

    Args:
        image_label_pairs (list of tuples): A list of tuples, where each tuple contains the file path of an image and its corresponding label file.
        n_cols (int, optional): The number of columns to display. Defaults to 5.
        n_rows (int, optional): The number of rows to display. Defaults to 5.
        num_images (int, optional): The number of images to display. Defaults to 25.
        figsize (tuple, optional): The size of the figure. Defaults to (20, 20).
        class_names (dict, optional): A dictionary mapping class IDs to class names. If not provided, the class IDs will be displayed instead.

    Returns:
        None
    """
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown']
    selected_pairs = random.sample(image_label_pairs, min(num_images, len(image_label_pairs)))

    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=figsize_)
    axes = axes.ravel()

    for idx, (image_path, label_path) in enumerate(selected_pairs):
        try:
            image, annotations = read_images_and_annotations(image_path, label_path)
            axes[idx].imshow(image)
            for ann in annotations:
                class_id = ann['class_id']
                x_center, y_center, width, height = ann['x_center'], ann['y_center'], ann['width'], ann['height']
                x1 = int((x_center - width / 2) * image.shape[1])
                y1 = int((y_center - height / 2) * image.shape[0])
                x2 = int((x_center + width / 2) * image.shape[1])
                y2 = int((y_center + height / 2) * image.shape[0])
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=colors[class_id], facecolor='none')
                axes[idx].add_patch(rect)
                text = f'Class: {class_id}'
                axes[idx].text(x1, y1, text, color='white', fontsize=8, verticalalignment='top', bbox={'color': colors[class_id], 'pad': 0})
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()
