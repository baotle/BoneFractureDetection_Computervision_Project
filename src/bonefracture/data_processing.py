"""
Data loading, validation, and integrity-checking utilities for the
bone fracture detection project.
"""

import os
from PIL import Image
import imagehash
import shutil
import yaml
import kagglehub


def get_split_paths(dataset_dir, split_name):
    """
    Creates the image and label path strings for in a given dataset/split directory

    Parameters:
      
      dataset_dir(str) : path of the dataset_dir and path (e.g ""BoneFractureYolo8"")
      split_name(str) : path of the split (e.g "train")
    
    Returns:
      images_dir (str) : path to the images (e.g ""BoneFractureYolo8"/train/images")
      labels_dir (str) : path to the labels (e.g ""BoneFractureYolo8"/train/labels")

    """
    images_dir = os.path.join(dataset_dir, split_name, 'images')
    labels_dir = os.path.join(dataset_dir, split_name, 'labels')
    return images_dir, labels_dir


def check_image_integrity(dataset_dir, split_name):
    """
    Loops and opens all the files in the given dataset for the given split (train, val, test) and returns all
    files that cannot be open broken_files

    Parameters

      dataset_dir (str) : path of the dataset directory ( f.e '/content/data/raw' )
      split_name (str) : name of the split as named in the dataset directory ( f.e train, val )

    Returns:

      broken_files (list[str]) : list of all file paths that could not be opened
    """

    images_dir = os.path.join(dataset_dir, split_name, 'images')
    broken_files = []

    for filename in os.listdir(images_dir):
        filepath = os.path.join(images_dir, filename)
        try:
            img = Image.open(filepath)
            img.verify()
        except Exception as e:
            broken_files.append((filename, str(e)))
    return broken_files


def check_label_integrity(dataset_dir, split_name, class_names):

    """
    Checks whether
    1. there are exactly 5 coordiantes
    2. class_id is within the range of our class_ids
    3. if all coordinates are within 0 and 1 to be YOLO compatible

    Parameters

      dataset_dir (str) : path of the dataset directory ( f.e '/content/data/raw' )
      split_name (str) : name of the split as named in the dataset directory ( f.e train, val )
      class_names (str) : class

    Returns:

      broken_files (list[str]) : returns a list of all the broken file names annotated with the reason why they were marked as broken
    """

    labels_dir = os.path.join(dataset_dir, split_name, 'labels')
    problems = []
    for filename in os.listdir(labels_dir):
        filepath = os.path.join(labels_dir, filename)
        with open(filepath) as f:
            for line_num, line in enumerate(f, start=1):
                parts = line.strip().split()

                #1. Check if there are exactly 5 coordiantes
                if len(parts) < 5:
                    problems.append(f"{filename} line {line_num}: expected at least 5 values, got {len(parts)}")
                    continue
                coord_values = parts[1:]
                if len(coord_values) % 2 != 0:
                    problems.append(f"{filename} line {line_num}: odd number of coordinate values")
                    continue

                #2. class_id is within the range of our class_ids and a Valid Integer
                try:
                    class_id = int(parts[0])
                    if not (0 <= class_id < len(class_names)):
                        problems.append(f"{filename} line {line_num}: class_id {class_id} out of range")
                except ValueError:
                    problems.append(f"{filename} line {line_num}: class_id not a valid integer")

                #3. Are my coordinates normalized between 0 and 1 (YOLO compatible)
                try:
                    coords = [float(v) for v in coord_values]
                    if not all(0.0 <= c <= 1.0 for c in coords):
                        problems.append(f"{filename} line {line_num}: coordinates out of 0-1 range")
                except ValueError:
                    problems.append(f"{filename} line {line_num}: coordinates not valid numbers")
    return problems


def compute_hashes(dataset_dir, split_name):
    """
    Computes the perceptioal hashes of an image to compare later with the check_split_leakage function

    Parameters:
      dataset_dir(str) : name of the datset directory (e.g "BoneFractureYolo8" )
      split_name(str) : name of the split (e.g "train" )

    Returns:
      hashes(Dict[str : str]) : In the hashes dictonary the respective filenames of the images are mapped to their perceptual hashes
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
    Computes the sets of the hashes in both splits and returns the intersection (&) between those two.
    Ideally the intersection is {}, which means no near identitical hashes between both splits.

    Parameters :

      dataset_dir(str) : name of the dataset directory
      split_a(str) : name of the first split
      split_b(str) : name of the second split

    Return :

      Returns the set of hashes that both appear in hashes_a and hashes_b 
    """
    hashes_a = compute_hashes(dataset_dir, split_a)
    hashes_b = compute_hashes(dataset_dir, split_b)

    return set(hashes_a.keys()) & set(hashes_b.keys())

def rebuild_dataset_verified(dataset_dir, verbose=True):

  """
  Downloads from kaggle the bonefracture-detection-computer-vision-project dataset 
  and automatically merges the "humerus" and "humerus fracture" class in the directory,
  while overwriting the data.yaml file accordingly. This is done in an idempotent way, such that
  if errors occur or downloading is stopped during the process, this function can just be rerun.

  At every step the process, while verbose = True, is logged.

  Parameters:
    dataset_dir(str) : Path of the dataset directory where the dataset is saved
    verbose (boolean) : If true the logging is enabled and keeps track of the process  
                        Defaults to true and is highly recommended
  """

  #0.1 Enables logging
    def log(msg):
        if verbose:
            print(msg)
    
    #1. Download and copy the dataset and remove preexisting tree

    local_path = kagglehub.dataset_download("pkdarabi/bone-fracture-detection-computer-vision-project")

    shutil.rmtree(dataset_dir, ignore_errors=True)
    shutil.copytree(local_path, dataset_dir)
    log("✅ Stage 1: downloaded and copied")

    #2. Flatten the downloaded structure

    CHOSEN = "BoneFractureYolo8"
    DUPLICATE = "bone fracture detection.v4-v4.yolov8"
    if os.path.exists(os.path.join(dataset_dir, CHOSEN)):
        for item in os.listdir(os.path.join(dataset_dir, CHOSEN)):
            shutil.move(os.path.join(dataset_dir, CHOSEN, item), os.path.join(dataset_dir, item))
        shutil.rmtree(os.path.join(dataset_dir, CHOSEN))
        if os.path.exists(os.path.join(dataset_dir, DUPLICATE)):
            shutil.rmtree(os.path.join(dataset_dir, DUPLICATE))
    log("✅ Stage 2: flattened")

    #3. Overwrite yaml path files to be compatible with the flattened structure
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    config['train'], config['val'], config['test'] = 'train/images', 'valid/images', 'test/images'

    #4.1 If "humerus" is still in the class_names
    old_class_names = config['names']
    OLD_CLASS, NEW_CLASS = 'humerus', 'humerus fracture'

    
    if OLD_CLASS in old_class_names:
      #4.2 Merge the humerus class and humerus fracture class into one and remap all other class names
        old_id = old_class_names.index(OLD_CLASS)
        new_class_names = [c for c in old_class_names if c != OLD_CLASS]
        new_id_for_merged = new_class_names.index(NEW_CLASS)

        id_remap = {i: new_class_names.index(old_class_names[i])
                    for i in range(len(old_class_names)) if old_class_names[i] != OLD_CLASS}
        id_remap[old_id] = new_id_for_merged  # route merged class correctly

        total_before, total_after = 0, 0
      #4.3 Rewrite every label_file
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
      
      #4.4 Sanity Check
        if total_after < total_before * 0.95:
            log(f"⚠️ WARNING: {total_before} lines before, {total_after} after — investigate!")
        else:
            log(f"✅ Stage 3: merge complete. {total_before} -> {total_after} lines")
      
      #4.5 Save the now overwritten data.yaml file
        config['nc'] = len(new_class_names)
        config['names'] = new_class_names
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        log("✅ Stage 3: skipped, already merged")
    #5. Final logging
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
    Converts every polygon file into a bounding box-file by parsing through the given directory
    /raw_dir/split_name/labels
    and saving the bounding boxed ones into a new directory
    /processed_dir/split_name/labels

    images are presevered and copied into the processed_dir

    Parameters:
      raw_dir(str) : name of the "raw data directory" (e.g "raw" )
      processed_dir : name of the processed data ( e.g "processed" )
      split_name : name of the split (e.g test, train, val)

    Returns:

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
    Converts the xs, ys cooridnates of the .yaml data polygons into bounding boxes

    Parameters:
      coords(list[int]) :  even numbered list of varied length, all the even indexed entries are the x-coordinates
                           odd numbered entries are the respective y-coordinates

    Returns:
      x_center(int) : x_cooridnate of the bounding box center
      y_center(int) : y_cooridnate of the bounding box center
      width(int) : width of the bounding box center
      height(int) : height of the bounding box center

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
