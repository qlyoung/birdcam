#!/usr/bin/env python3
"""
Transforms Cornell NABird dataset into YOLOv5-compatible dataset
"""

import os
import tempfile
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

DATASET_DIR = "nabirds"
OUTPUT_DIR = "nabirds_yolov5"


###
# The following was ripped from the nabirds.py provided, just to avoid
# complications with python packaging and Cornell using python 2 back in 2015
###
def load_bounding_box_annotations(dataset_path=""):
    """
    Load bounding box annotations from dataset
    """

    bboxes = {}

    with open(os.path.join(dataset_path, "bounding_boxes.txt")) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            bbox = map(int, pieces[1:])
            bboxes[image_id] = bbox

    return bboxes


def load_part_annotations(dataset_path=""):
    """
    Load bird part annotations from dataset
    """

    parts = {}

    with open(os.path.join(dataset_path, "parts/part_locs.txt")) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            parts.setdefault(image_id, [0] * 11)
            part_id = int(pieces[1])
            parts[image_id][part_id] = map(int, pieces[2:])

    return parts


def load_part_names(dataset_path=""):
    """
    Load bird part names from dataset
    """

    names = {}

    with open(os.path.join(dataset_path, "parts/parts.txt")) as f:
        for line in f:
            pieces = line.strip().split()
            part_id = int(pieces[0])
            names[part_id] = " ".join(pieces[1:])

    return names


def load_class_names(dataset_path=""):
    """
    Load mapping from names to classes from dataset

    Returns dict like {"0": "Birds", ...}
    """

    names = {}

    with open(os.path.join(dataset_path, "classes.txt")) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = " ".join(pieces[1:])

    return names


def load_image_labels(dataset_path=""):
    """
    Load labels for dataset images

    Returns dict like {"<image id>": "0", ...}
    """
    labels = {}

    with open(os.path.join(dataset_path, "image_class_labels.txt")) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            class_id = pieces[1]
            labels[image_id] = class_id

    return labels


def load_image_paths(dataset_path="", path_prefix=""):
    """
    Load image paths for dataset images

    Paths are relative to 'image' directory within dataset

    Returns dict like {"<image id>": "102/<image.jpg>", ...}
    """
    paths = {}

    with open(os.path.join(dataset_path, "images.txt")) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            path = os.path.join(path_prefix, pieces[1])
            paths[image_id] = path

    return paths


def load_image_sizes(dataset_path=""):
    """
    Load image sizes from dataset

    Returns dict like {"<image id>": [<x>, <y>], ...}
    """
    sizes = {}

    with open(os.path.join(dataset_path, "sizes.txt")) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            width, height = map(int, pieces[1:])
            sizes[image_id] = [width, height]

    return sizes


def load_hierarchy(dataset_path=""):

    parents = {}

    with open(os.path.join(dataset_path, "hierarchy.txt")) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            parents[child_id] = parent_id

    return parents


def load_photographers(dataset_path=""):

    photographers = {}
    with open(os.path.join(dataset_path, "photographers.txt")) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            photographers[image_id] = " ".join(pieces[1:])

    return photographers


def load_train_test_split(dataset_path=""):
    """
    Load recommended train/test split

    Returns two lists; the first list is the recommended list of training
    images, the latter the recommended test images
    """
    train_images = []
    test_images = []

    with open(os.path.join(dataset_path, "train_test_split.txt")) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            is_train = int(pieces[1])
            if is_train:
                train_images.append(image_id)
            else:
                test_images.append(image_id)

    return train_images, test_images


###
# End Cornell rip
###


def annotate(outputdir, datasetdir=DATASET_DIR):
    """
    For all images in imagedir, annotate them using the metadata located in
    datasetdir. The annotation files are placed next to the image.

    The images must be named according to their corresponding image IDs.
    """
    bboxes = load_bounding_box_annotations(datasetdir)
    labels = load_image_labels(datasetdir)
    sizes = load_image_sizes(datasetdir)
    classnames = load_class_names(datasetdir)

    for name, _ in bboxes.items():
        bbox = list(bboxes[name])
        clas = labels[name]
        size = sizes[name]
        class_name = classnames[clas]

        print(
            "{}: bbox = {}, class = {} ({}), size = {}".format(
                name, bbox, clas, class_name, size
            )
        )

        # Need to transform from the dataset's native bounding box format,
        # given as:
        #    x, y, width, height
        # where (x,y) designates the top left of the bounding box to YOLOv5
        # bounding box format, given as:
        #    x_center, y_center, width, height
        # where (x_center,y_center) designates the center of the bounding box
        # and width and height remain the same (denote the complete width and
        # height of the bounding box, not half)

        bbox = [
            bbox[0] + bbox[2] / 2.0,
            bbox[1] + bbox[3] / 2.0,
            bbox[2],
            bbox[3],
        ]

        # Also, all coordinates are normalized to [0,1]
        bbox = [
            bbox[0] / size[0],
            bbox[1] / size[1],
            bbox[2] / size[0],
            bbox[3] / size[1],
        ]
        fields = [clas, *bbox]

        with open(os.path.join(outputdir, name + ".txt"), "w") as annotation_file:
            annotation_file.write("{} {} {} {} {}\n".format(*fields))


def restructure_for_yolov5(
    stagingdir, outputdir, datasetdir=DATASET_DIR, respect_suggestions=False
):
    """
    Given a staging directory, split into training, validation and test sets.

    :param respect_suggestions: Whether to use the suggested training/test
    split given as part of the dataset.
    """
    ids = list(load_image_paths(datasetdir).keys())

    # This next code is a bit confusing because we operate purely in id-space
    # until the end; since images and their annotation files have the same
    # prefix which is unique to that pair, we dont really need to operate on
    # image files and annotation files as if they're discrete; just compute the
    # splits and append the necessary extensions at the end

    if not respect_suggestions:
        images = ids
        annotations = ids
        train_images, val_images, train_annotations, val_annotations = train_test_split(
            images, annotations, test_size=0.2, random_state=1
        )
        val_images, test_images, val_annotations, test_annotations = train_test_split(
            val_images, val_annotations, test_size=0.5, random_state=1
        )
    else:
        train_ids, test_ids = load_train_test_split(datasetdir)
        # Take the training set at face value
        train_images = train_ids
        train_annotations = train_ids

        # Take the test set and split it into a test and val set
        val_images, test_images, val_annotations, test_annotations = train_test_split(
            test_ids, test_ids, test_size=0.5, random_state=1
        )

    train_images = list(
        map(lambda x: os.path.join(stagingdir, x) + ".jpg", train_images)
    )
    test_images = list(map(lambda x: os.path.join(stagingdir, x) + ".jpg", test_images))
    val_images = list(map(lambda x: os.path.join(stagingdir, x) + ".jpg", val_images))
    train_annotations = list(
        map(lambda x: os.path.join(stagingdir, x) + ".txt", train_annotations)
    )
    test_annotations = list(
        map(lambda x: os.path.join(stagingdir, x) + ".txt", test_annotations)
    )
    val_annotations = list(
        map(lambda x: os.path.join(stagingdir, x) + ".txt", val_annotations)
    )

    def move_files_to_folder(list_of_files, destination_folder):
        print("Moving to {}".format(destination_folder))
        for f in list_of_files:
            try:
                shutil.copy(f, destination_folder)
            except:
                print(f)
                assert False

    # Move the splits into their folders
    tomove = [
        (os.path.join(outputdir, "images", "train"), train_images),
        (os.path.join(outputdir, "images", "val"), val_images),
        (os.path.join(outputdir, "images", "test"), test_images),
        (os.path.join(outputdir, "labels", "train"), train_annotations),
        (os.path.join(outputdir, "labels", "val"), val_annotations),
        (os.path.join(outputdir, "labels", "test"), test_annotations),
    ]

    for path, images in tomove:
        Path(path).mkdir(parents=True, exist_ok=True)
        move_files_to_folder(images, path)


def flatten(stagedir, datasetdir=DATASET_DIR):
    # Load mapping of image IDs to paths
    image_paths = load_image_paths(datasetdir)

    # Copy all images to <id>.jpg in the staging directory
    counter = 0
    for name, path in image_paths.items():
        shutil.copyfile(
            os.path.join(datasetdir, "images", path),
            os.path.join(stagedir, name + ".jpg"),
        )
        counter += 1
        if counter % 1000 == 0:
            print("... Copied {}".format(counter))

    print("Copied {} total".format(counter))
    return stagedir


def dump_class_names_as_yaml_list(datasetdir=DATASET_DIR):
    """
    Dump the class names as a YAML list where the items are ordered by their
    class ID.
    """
    classnames = load_class_names(datasetdir)

    thelist = []
    for i in range(0, len(classnames)):
        thelist.append(classnames[str(i)])

    # Conveniently, YAML inline list syntax matches Python's for this particular list
    print(thelist)
    print("\n{} classes".format(len(thelist)))


if __name__ == "__main__":
    stagedir = tempfile.mkdtemp(dir=".")
    print("Created stage directory: {}".format(stagedir))
    print("Copying all images to stage directory...")
    flatten(stagedir)
    print("Staged to {}".format(stagedir))
    print("Annotating...")
    annotate(stagedir)
    print("Restructuring {} into {}".format(stagedir, OUTPUT_DIR))
    restructure_for_yolov5(stagedir, OUTPUT_DIR, respect_suggestions=True)
    dump_class_names_as_yaml_list()
