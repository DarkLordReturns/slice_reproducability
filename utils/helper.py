import os
import skimage
import cv2
import copy
import numpy as np
import math
from scipy.stats import variation
from scipy.stats import gaussian_kde
from scipy.linalg import LinAlgError
from skimage.segmentation import mark_boundaries
from joblib import Parallel, delayed
import torch
import torchvision.transforms.functional as F
from torchvision.datasets import OxfordIIITPet, VOCSegmentation
from torchvision.transforms import GaussianBlur
from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO
import subprocess
from PIL import Image


# Function to replace NaN values with the median
def replace_nan_with_median(data):
    if np.isnan(data).any():
        median_value = np.nanmedian(data)
        data = np.where(np.isnan(data), median_value, data)
    return data


# Function to cap values
def cap_values(data, lower_percentile, upper_percentile):
    lower_cap = np.percentile(data, lower_percentile)
    upper_cap = np.percentile(data, upper_percentile)
    return np.clip(data, lower_cap, upper_cap)


def get_quartile_information(value_list, method='quartile'):
    mean_value = np.mean(value_list, axis=1)
    if method == 'std':
        std_value = np.std(value_list, axis=1)
        lower_bar = mean_value - std_value
        upper_bar = mean_value + std_value
    else:
        q1_value = np.percentile(value_list, 25, axis=1)
        q3_value = np.percentile(value_list, 75, axis=1)
        lower_bar = mean_value - q1_value
        upper_bar = q3_value - mean_value

    iqr_bar = [lower_bar, upper_bar]
    return iqr_bar, mean_value


def download_dataset(dataset_name, model_preprocess=None):
    if dataset_name == 'oxpets':
        if model_preprocess:
            dataset = OxfordIIITPet(root='images_oxpets', download=True, target_types=["segmentation"],
                                    transform=model_preprocess)
        else:
            dataset = OxfordIIITPet(root='images_oxpets', download=True, target_types=["segmentation"])
        image_filenames = [dataset._images[image_index].as_posix() for image_index in range(len(dataset))]
    elif dataset_name == 'pvoc':
        if model_preprocess:
            dataset = VOCSingleObjectSegmentation(root='images_pvoc', year='2009', image_set='train', transform=model_preprocess)
        else:
            dataset = VOCSingleObjectSegmentation(root='images_pvoc', year='2009', image_set='train')
        image_filenames = [dataset.filtered_data[image_index][0] for image_index in range(len(dataset))]
    else:
        coco_classes = [
                        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
                        "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", 
                        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", 
                        "truck", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
                        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
                        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
                        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
                        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", 
                        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
                        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
                    ]
        dataset = COCODataset(
                        root='coco_dataset',  
                        annFile='coco_dataset/annotations/instances_val2014.json',
                        selected_classes=coco_classes,
                        download=True,
                        transform=model_preprocess
                    )
        image_filenames = []
        for img_id in dataset.img_ids:
            img_info = dataset.coco.loadImgs(img_id)[0]
            img_path = os.path.join(dataset.root,'val2014',img_info['file_name'])
            image_filenames.append(img_path)

    return dataset, image_filenames


def generate_segment_perturbation_encodings(num_perturb, segment_labels):
    num_superpixels = np.unique(segment_labels).shape[0]
    perturbations = np.ones(num_perturb * num_superpixels).reshape((num_perturb, num_superpixels))
    active_superpixel = 0
    for i in range(perturbations.shape[0]):
        perturbations[i, active_superpixel] = 0
        active_superpixel += 1
    return perturbations


def get_segment_labels_for_image_pixels(image_name, image_file, image_tensor, target_image_size, explainer_class,
                                        model, explain_model, pretrained_model, batch_size, num_workers, sigma_value,
                                        save_first_perturb_image, parts_to_run, save_dir_path, strategy='quickshift'):
    try:
        image = skimage.io.imread(image_file)
        image = skimage.transform.resize(image, target_image_size)
        if strategy == 'quickshift':
            final_segment_labels = skimage.segmentation.quickshift(image, kernel_size=5, max_dist=200, ratio=0.2)
        else:
            max_variation = 0
            best_box_size = 0
            final_segment_labels = None
            for segment_box_size in range(int(round(target_image_size[0] / 60) * 10), 110, 10):
                print(f'Checking for segment_box_size: {segment_box_size}')
                segment_labels = np.zeros_like(image[:, :, 0])
                segment_number = 0
                if image.shape[0] % segment_box_size != 0:
                    segments_limit = math.floor(image.shape[0] / segment_box_size)
                    for h in range(segments_limit + 1):
                        for w in range(segments_limit + 1):
                            if h == segments_limit:
                                if w == segments_limit:
                                    segment_labels[h * segment_box_size:, w * segment_box_size:] = segment_number
                                else:
                                    segment_labels[h * segment_box_size:,
                                    w * segment_box_size:(w + 1) * segment_box_size] = segment_number
                            else:
                                if w == segments_limit:
                                    segment_labels[h * segment_box_size:(h + 1) * segment_box_size,
                                    w * segment_box_size:] = segment_number
                                else:
                                    segment_labels[h * segment_box_size:(h + 1) * segment_box_size,
                                    w * segment_box_size:(w + 1) * segment_box_size] = segment_number
                            segment_number += 1
                else:
                    for h in range(0, image.shape[0], segment_box_size):
                        for w in range(0, image.shape[1], segment_box_size):
                            segment_labels[h:h + segment_box_size, w:w + segment_box_size] = segment_number
                            segment_number += 1
                explain_model_object = explainer_class(image_name, image_tensor, segment_labels,
                                                       pretrained_model, model, batch_size, num_workers,
                                                       save_first_perturb_image, parts_to_run, save_dir_path)
                segment_perturbation_encodings = generate_segment_perturbation_encodings(
                    np.unique(segment_labels).shape[0], segment_labels)
                if explain_model == 'GridLime':
                    segment_perturbed_images = explain_model_object.generate_perturbed_images_from_encodings(
                        segment_perturbation_encodings)
                else:
                    segment_perturbed_images = explain_model_object.generate_perturbed_images_from_encodings(
                        segment_perturbation_encodings, sigma_value)

                perturbed_dataset = PerturbedDataset(segment_perturbed_images)
                perturbed_dataloader = DataLoader(perturbed_dataset, batch_size=64, shuffle=False, num_workers=8)
                batch_predictions = Parallel(n_jobs=-1)(delayed(explain_model_object.classify_image_batch)(
                    perturbed_image_batch) for perturbed_image_batch in perturbed_dataloader)
                segment_combined_predictions = np.concatenate(batch_predictions, axis=0)
                segment_variation = variation(
                    np.abs(segment_combined_predictions - explain_model_object.probability_of_prediction), axis=0)[0]
                if (segment_variation != 0) and (segment_variation > max_variation):
                    max_variation = segment_variation
                    best_box_size = segment_box_size
                    final_segment_labels = segment_labels
            print(f'Best segment box size found is: {best_box_size}')
        return final_segment_labels
    except Exception as exception:
        raise exception


def generate_perturbation_encodings(test_sample_size, num_superpixels, existing_perturbations=0):
    num_perturb = test_sample_size - existing_perturbations
    perturbations = np.random.randint(0, 2, num_perturb * num_superpixels).reshape(
        (num_perturb, num_superpixels))
    return perturbations


def get_original_image_prediction(image, model, device='cuda'):
    original_image = copy.deepcopy(image)
    original_image = original_image.to(device)
    with torch.no_grad():
        prediction = model(original_image)

    prediction = torch.nn.functional.softmax(prediction, dim=1).cpu().numpy()
    top_predicted_class = prediction[0].argsort()[-1:][::-1]
    probability_of_prediction = prediction[0][top_predicted_class]
    return top_predicted_class, probability_of_prediction


def perturb_image(perturbation_encoding, image_to_perturb, superpixels, sigma_value=0):
    active_pixels = np.where(perturbation_encoding == 1)[0]
    mask = np.zeros(superpixels.shape)
    for active in active_pixels:
        mask[superpixels == active] = 1

    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    mask_tensor = mask_tensor.expand(image_to_perturb.shape)
    perturbed_image = copy.deepcopy(image_to_perturb)

    if sigma_value == 0:
        perturbed_image = perturbed_image * mask_tensor
        perturbed_image = perturbed_image.squeeze(0)
    else:
        kernel_size = 2 * int(3 * sigma_value) + 1
        mask3d = mask_tensor.squeeze(0)
        gaussian_blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma_value)
        blurred_image = gaussian_blur(copy.deepcopy(perturbed_image).squeeze(0))
        perturbed_image = torch.where(mask3d == 0, blurred_image, perturbed_image.squeeze(0))
    return perturbed_image


def calculate_entropy(data):
    try:
        scipy_kernel = gaussian_kde(data)

        #  We calculate the bandwidth for later use
        optimal_bandwidth = scipy_kernel.factor * np.std(data)

        # Calculate KDE for the entire dataset
        kde = gaussian_kde(data, bw_method=optimal_bandwidth)

        # Create a range of values to represent the KDE
        x = np.linspace(np.min(data), np.max(data), 1000)

        # Evaluate the density at each point in the range
        density = kde(x)

        # Normalize the density function
        normalized_density = density / np.sum(density * (x[1] - x[0]))

        # Calculate the probabilities of positive and negative values
        positive_probability = np.sum(normalized_density[x >= 0] * (x[1] - x[0]))
        negative_probability = np.sum(normalized_density[x < 0] * (x[1] - x[0]))

        if positive_probability == 0 or negative_probability == 0:
            sign_entropy = 0
        else:
            sign_entropy = -positive_probability * np.log2(positive_probability) \
                           - negative_probability * np.log2(negative_probability)

    except LinAlgError as exception:
        # print(f"Warning: {exception}. Returning 0 entropy.")
        sign_entropy = 0

    return sign_entropy


def save_image(original_image_file_name, process, image_to_save, main_save_dir, sigma_value=0):
    save_dir = os.path.join(main_save_dir, original_image_file_name, process)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    pil_image = copy.deepcopy(image_to_save)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    pil_image = F.to_pil_image((pil_image * std) + mean)
    image_name = original_image_file_name + '_sigma_' + f'{sigma_value}' + '.jpg'
    pil_image.save(save_dir + '/' + image_name)


def save_colored_output_image(original_image_file_name, original_image_name, segment_labels, superpixel_list,
                              target_image_size, main_save_dir, process='most_relevant'):
    save_dir = os.path.join(main_save_dir, original_image_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    original_image = cv2.imread(original_image_file_name)
    original_image = skimage.transform.resize(original_image, target_image_size)
    original_image = np.dstack((original_image, np.full(original_image.shape[:2], 1.0, dtype=original_image.dtype)))

    alpha = 0.2

    # Create separate layers
    label_layer = np.zeros_like(original_image)  # Layer for labels
    mask_layer = copy.deepcopy(original_image)  # Layer for colored masks
    boundary_layer = mark_boundaries(original_image[:, :, :3], segment_labels, color=(1, 0, 0), mode='thick')
    boundary_layer = np.dstack((boundary_layer, original_image[:, :, 3]))  # Preserve alpha channel

    for superpixel in superpixel_list:
        # Create the mask
        mask = np.isin(segment_labels, [superpixel]).astype(int)
        mask3d = cv2.merge((mask, mask, mask, np.ones_like(mask)))
        mask3d = mask3d.astype(float)
        ones_mask = np.all(mask3d[:, :, :3] == [1.0, 1.0, 1.0], axis=-1)

        # Adjust mask transparency
        mask3d[:, :, 3] = np.where(ones_mask, alpha, mask3d[:, :, 3])
        mask3d[:, :, :3] = np.where(mask3d[:, :, :3] == [0.0, 0.0, 0.0], np.array([1.0, 1.0, 1.0]), mask3d[:, :, :3])
        mask_layer = mask_layer * mask3d

        # Highlight relevant superpixel with color
        if process == 'most_relevant':
            color_axis = 0
        else:
            color_axis = 2
        mask_layer[:, :, color_axis] = np.where(ones_mask, 1.0, mask_layer[:, :, color_axis])

        # Calculate the centroid of the superpixel
        coords = np.argwhere(segment_labels == superpixel)
        centroid = coords.mean(axis=0).astype(int)
        text_index = np.where(superpixel_list == superpixel)
        text = str(text_index[0][0] + 1)

        # Add labels to the label layer
        cv2.putText(
            label_layer,
            text,
            (centroid[1], centroid[0]),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,  # Slightly larger for better visibility
            color=(255, 255, 255, 255),  # Bright white in RGBA
            thickness=1,  # Increased thickness for better contrast
            lineType=cv2.LINE_AA
        )

    # Combine all layers
    combined_image = (mask_layer * 0.5 + boundary_layer * 0.5 + label_layer).clip(0, 1)

    # Save the output image
    output_image_name = 'top_5_' + process + '_superpixels_combined.png'
    cv2.imwrite(save_dir + '/' + output_image_name, (combined_image * 255).astype(np.uint8))


class PerturbedDataset(Dataset):
    def __init__(self, perturbed_images):
        self.data = perturbed_images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class COCODataset(Dataset):
    def __init__(self, root, annFile, selected_classes, download=False, transform=None):

        self.root = root  # Path to images
        self.annFile = annFile  # Path to annotation file
        self.selected_classes = selected_classes  # List of class names to filter
        self.transform = transform

        if download:
            self.download_data()
         
        # Initialize the COCO object
        self.coco = COCO(self.annFile)  # Load the annotation file

        # Get class IDs for selected classes
        self.class_ids = self._get_class_ids()

        # Filter images with exactly one object and selected class
        self.img_ids = self._get_single_object_images()

    def _get_class_ids(self):
        # Get class ids for selected classes
        class_ids = []
        for class_name in self.selected_classes:
            class_id = self.coco.getCatIds(catNms=[class_name])  # Get category IDs for the class name
            if class_id:
                class_ids.extend(class_id)  # Add the found class IDs
        return class_ids

    def _get_single_object_images(self):
        # Filter out images that have only one object (annotation) and belong to selected classes
        single_object_img_ids = []
        for img_id in self.coco.imgs.keys():
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)  # Get annotations for this image
            if len(ann_ids) == 1:  # Check if image has only one object
                anns = self.coco.loadAnns(ann_ids)  # Load annotation details
                # Check if the object belongs to the selected class
                if any(ann['category_id'] in self.class_ids for ann in anns):
                    single_object_img_ids.append(img_id)
        return single_object_img_ids

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)  # Get annotations for this image
        anns = self.coco.loadAnns(ann_ids)  # Load annotation details

        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root,'val2014',img_info['file_name'])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create segmentation mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            if 'segmentation' in ann:
                for seg in ann['segmentation']:
                    poly = np.array(seg, np.int32).reshape((-1, 2))
                    cv2.fillPoly(mask, [poly], 1)

        # Apply transforms if needed
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(mask, dtype=torch.uint8)

    def __len__(self):
        return len(self.img_ids)

    def download_data(self):
        # Ensure the coco dataset directory exists
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        # Check if the dataset is already downloaded
        if not os.path.exists(os.path.join(self.root, 'val2014')):
            print("Downloading validation images...")
            subprocess.run(['wget', 'http://images.cocodataset.org/zips/val2014.zip', '-P', self.root])  # Download images
            subprocess.run(['unzip', os.path.join(self.root, 'val2014.zip'), '-d', self.root])  # Extract images

        if not os.path.exists(os.path.join(self.root, 'annotations')):
            print("Downloading annotations...")
            subprocess.run(['wget', 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip', '-P', self.root])  # Download annotations
            subprocess.run(['unzip', os.path.join(self.root, 'annotations_trainval2014.zip'), '-d', self.root])  # Extract annotations

        # Re-initialize the COCO annotations after download
        self.coco = COCO(self.annFile)


class VOCSingleObjectSegmentation(Dataset):
    def __init__(self, root, year='2007', image_set='train', transform=None):
        self.voc_dataset = VOCSegmentation(root=root, year=year, image_set=image_set, download=True)
        self.transform = transform
        self.filtered_data = self._filter_single_object_images()

    def _filter_single_object_images(self):
        valid_data = []
        for idx in range(len(self.voc_dataset)):  # Iterate over dataset length
            img_path = self.voc_dataset.images[idx]  # Get image path
            mask_path = self.voc_dataset.masks[idx]  # Get corresponding mask path

            mask = Image.open(mask_path)
            mask_np = np.array(mask)

            unique_labels = np.unique(mask_np)

            if len(unique_labels) <= 3:  # Only one object present
                valid_data.append((img_path, mask_path))
        return valid_data

    def __len__(self):
        return len(self.filtered_data)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.filtered_data[idx]
        image = Image.open(img_path).convert("RGB")
        
        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.int32)
        mask[(mask == 255)] = 0
        mask[(mask != 0)] = 1

        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.int64)
        return image, mask
