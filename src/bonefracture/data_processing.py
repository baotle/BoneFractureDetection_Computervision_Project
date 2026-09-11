"""
Data loading, validation, and integrity-checking utilities for the
bone fracture detection project.
"""

import os
from PIL import Image
import imagehash
import shutil
import yaml

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

def rebuild_dataset_verified(dataset_dir, verbose=True):
    def log(msg):
        if verbose:
            print(msg)

    import kagglehub
    local_path = kagglehub.dataset_download("pkdarabi/bone-fracture-detection-computer-vision-project")

    shutil.rmtree(dataset_dir, ignore_errors=True)
    shutil.copytree(local_path, dataset_dir)
    log("✅ Stage 1: downloaded and copied")

    CHOSEN = "BoneFractureYolo8"
    DUPLICATE = "bone fracture detection.v4-v4.yolov8"
    if os.path.exists(os.path.join(dataset_dir, CHOSEN)):
        for item in os.listdir(os.path.join(dataset_dir, CHOSEN)):
            shutil.move(os.path.join(dataset_dir, CHOSEN, item), os.path.join(dataset_dir, item))
        shutil.rmtree(os.path.join(dataset_dir, CHOSEN))
        if os.path.exists(os.path.join(dataset_dir, DUPLICATE)):
            shutil.rmtree(os.path.join(dataset_dir, DUPLICATE))
    log("✅ Stage 2: flattened")

    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    config['train'], config['val'], config['test'] = 'train/images', 'valid/images', 'test/images'

    old_class_names = config['names']
    OLD_CLASS, NEW_CLASS = 'humerus', 'humerus fracture'

    if OLD_CLASS in old_class_names:
        old_id = old_class_names.index(OLD_CLASS)
        new_class_names = [c for c in old_class_names if c != OLD_CLASS]
        new_id_for_merged = new_class_names.index(NEW_CLASS)

        id_remap = {i: new_class_names.index(old_class_names[i])
                    for i in range(len(old_class_names)) if old_class_names[i] != OLD_CLASS}
        id_remap[old_id] = new_id_for_merged  # route merged class correctly

        total_before, total_after = 0, 0
        for split in ['train', 'valid', 'test']:
            split_labels_dir = os.path.join(dataset_dir, split, 'labels')
            for label_file in os.listdir(split_labels_dir):
                label_path = os.path.join(split_labels_dir, label_file)
                with open(label_path) as f:
                    lines = f.readlines()
                total_before += len(lines)

                new_lines = [r for line in lines if (r := merge_label_line(line, id_remap)) is not None]
                total_after += len(new_lines)

                with open(label_path, 'w') as f:
                    f.writelines(new_lines)

        if total_after < total_before * 0.95:
            log(f"⚠️ WARNING: {total_before} lines before, {total_after} after — investigate!")
        else:
            log(f"✅ Stage 3: merge complete. {total_before} -> {total_after} lines")

        config['nc'] = len(new_class_names)
        config['names'] = new_class_names
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        log("✅ Stage 3: skipped, already merged")

    log(f"\n🎉 Complete. nc={config['nc']}, names={config['names']}")
    return True

def merge_label_line(line, id_remap):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    class_id = int(parts[0])
    parts[0] = str(id_remap.get(class_id, class_id))
    return ' '.join(parts) + '\n'

def convert_split_to_bbox(raw_dir, processed_dir, split_name):
    """
    Read every polygon-format label file in a split, convert to
    bbox format, and write to the corresponding location under
    processed_dir. Images are copied unchanged (only labels differ).
    """
    src_images = os.path.join(raw_dir, split_name, 'images')
    src_labels = os.path.join(raw_dir, split_name, 'labels')
    dst_images = os.path.join(processed_dir, split_name, 'images')
    dst_labels = os.path.join(processed_dir, split_name, 'labels')

    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    # Copy images as-is
    for filename in os.listdir(src_images):
        shutil.copy2(os.path.join(src_images, filename), os.path.join(dst_images, filename))

    # Convert each label file
    for filename in os.listdir(src_labels):
        with open(os.path.join(src_labels, filename)) as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = parts[0]
            coords = [float(v) for v in parts[1:]]
            x_c, y_c, w, h = polygon_to_bbox(coords)
            new_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        with open(os.path.join(dst_labels, filename), 'w') as f:
            f.writelines(new_lines)

    return len(os.listdir(src_labels))

def polygon_to_bbox(coords):
    """
    Convert a polygon (list of x,y pairs) to a YOLO-format bounding box
    (x_center, y_center, width, height), all normalized 0-1.

    We just take the min/max extent of all polygon points — the smallest
    box that fully contains the polygon shape.
    """
    xs = coords[0::2]  # every even index: x1, x2, x3...
    ys = coords[1::2]  # every odd index: y1, y2, y3...

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height
