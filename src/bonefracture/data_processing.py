"""
Data loading, validation, and integrity-checking utilities for the
bone fracture detection project.
"""

import os
from PIL import Image
import imagehash


def get_split_paths(dataset_dir, split_name):
    """
    Given the dataset root and a split name ('train', 'valid', 'test'),
    return the paths to that split's images and labels folders.

    Centralizing this in one place means if the folder layout ever
    changes, we only need to update it here — not in every function
    that needs to build these paths.
    """
    images_dir = os.path.join(dataset_dir, split_name, 'images')
    labels_dir = os.path.join(dataset_dir, split_name, 'labels')
    return images_dir, labels_dir


def check_image_integrity(dataset_dir, split_name):
    """
    Try to open every image in a split. Returns a list of
    (filename, error_message) tuples for any file that fails to open —
    an empty list means every image is valid.
    """
    images_dir, _ = get_split_paths(dataset_dir, split_name)
    all_images = os.listdir(images_dir)

    broken_files = []
    for filename in all_images:
        filepath = os.path.join(images_dir, filename)
        try:
            img = Image.open(filepath)
            img.verify()
        except Exception as e:
            broken_files.append((filename, str(e)))

    return broken_files


def check_label_integrity(dataset_dir, split_name, class_names):
    """
    Validate every label file in a split. Checks that each line has
    exactly 5 values, a valid class_id, and normalized (0-1) coordinates.
    Returns a list of human-readable problem descriptions — an empty
    list means every label file is valid.
    """
    _, labels_dir = get_split_paths(dataset_dir, split_name)
    all_labels = os.listdir(labels_dir)

    problems = []
    for filename in all_labels:
        filepath = os.path.join(labels_dir, filename)
        with open(filepath) as f:
            for line_num, line in enumerate(f, start=1):
                parts = line.strip().split()

                if len(parts) != 5:
                    problems.append(f"{filename} line {line_num}: expected 5 values, got {len(parts)}")
                    continue

                class_id_str, x, y, w, h = parts

                try:
                    class_id = int(class_id_str)
                    if not (0 <= class_id < len(class_names)):
                        problems.append(f"{filename} line {line_num}: class_id {class_id} out of range")
                except ValueError:
                    problems.append(f"{filename} line {line_num}: class_id '{class_id_str}' is not a valid integer")

                try:
                    coords = [float(x), float(y), float(w), float(h)]
                    if not all(0.0 <= c <= 1.0 for c in coords):
                        problems.append(f"{filename} line {line_num}: coordinates out of 0-1 range: {coords}")
                except ValueError:
                    problems.append(f"{filename} line {line_num}: coordinates are not valid numbers")

    return problems


def compute_hashes(dataset_dir, split_name):
    """
    Compute a perceptual hash for every image in a split.
    Returns a dict: {hash_value: [list of filenames with that hash]}.
    Images sharing a hash are likely duplicates or near-duplicates.
    """
    images_dir, _ = get_split_paths(dataset_dir, split_name)
    hashes = {}

    for filename in os.listdir(images_dir):
        filepath = os.path.join(images_dir, filename)
        img = Image.open(filepath)
        h = imagehash.phash(img)
        hashes.setdefault(str(h), []).append(filename)

    return hashes


def check_split_leakage(dataset_dir, split_a, split_b):
    """
    Check whether any image (by perceptual hash) appears in both of
    two given splits. Returns the set of overlapping hash values —
    an empty set means no leakage between these two splits.
    """
    hashes_a = compute_hashes(dataset_dir, split_a)
    hashes_b = compute_hashes(dataset_dir, split_b)

    return set(hashes_a.keys()) & set(hashes_b.keys())