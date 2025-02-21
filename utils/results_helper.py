import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rbo
import math
import seaborn as sns
import copy
from PIL import Image
import matplotlib as mpl
import pingouin as pg
import torch
from torchvision import transforms

from utils.constants import dataset_paths, plot_colors
from utils.helper import replace_nan_with_median, cap_values, download_dataset, perturb_image, calculate_entropy, \
    get_original_image_prediction, get_quartile_information
from models.models import load_model_from_name, get_preprocess_pipeline_for_model, \
    get_preprocess_pipeline_for_model_mscoco_version


# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def AFSE(pos_superpixels, neg_superpixels, num_superpixels, iterations):
    sign_matrix = np.zeros(shape=(iterations, num_superpixels))

    for iteration_idx, pos_superpixel in enumerate(pos_superpixels):
        if len(pos_superpixel) > 0:
            sign_matrix[iteration_idx, np.array(pos_superpixel) - 1] = 1

    for iteration_idx, neg_superpixel in enumerate(neg_superpixels):
        if len(neg_superpixel) > 0:
            sign_matrix[iteration_idx, np.array(neg_superpixel) - 1] = -1

    sign_entropies = np.array([])
    for column in range(sign_matrix.shape[1]):
        data = sign_matrix[:, column]
        sign_entropy = calculate_entropy(data)
        sign_entropies = np.append(sign_entropies, sign_entropy)
        sign_entropies = np.append(sign_entropies, sign_entropy)

    return np.mean(sign_entropies)


def ARS(pos_superpixels, neg_superpixels, num_superpixels, iterations):
    ars_pos = 0
    ars_neg = 0
    num_combinations = math.comb(iterations, 2)

    for i, pos_superpixel_i in enumerate(pos_superpixels):
        for j, pos_superpixel_j in enumerate(pos_superpixels):
            if (i < j) and len(pos_superpixel_i) > 0 and len(pos_superpixel_j) > 0:
                curr_ars = rbo.RankingSimilarity(pos_superpixel_i, pos_superpixel_j).rbo_ext(p=0.2)
                ars_pos = ars_pos + curr_ars

    ars_pos = ars_pos / num_combinations

    for i, neg_superpixel_i in enumerate(neg_superpixels):
        for j, neg_superpixel_j in enumerate(neg_superpixels):
            if (i < j) and len(neg_superpixel_i) > 0 and len(neg_superpixel_j) > 0:
                curr_ars = rbo.RankingSimilarity(neg_superpixel_i, neg_superpixel_j).rbo_ext(p=0.2)
                ars_neg = ars_neg + curr_ars
    ars_neg = ars_neg / num_combinations

    return (ars_pos + ars_neg) / 2


def get_ccm_values(folder_name, num_iter=10):
    ccm_list = []
    rejected_list = []
    for folder_img in os.listdir(folder_name):
        with open(folder_name + r'/' + folder_img + r'/run_info.pickle', 'rb') as file:
            d = pickle.load(file)
        with open(folder_name + r'/' + folder_img + r'/segments.pickle', 'rb') as file:
            segments = pickle.load(file)
        img_name = folder_img
        if len(list(d.keys())) != num_iter:
            rejected_list.append(img_name)
            continue
        pos_master_list = []
        neg_master_list = []
        unstable_master_list = []
        num_superpixel = 0
        for i in range(1, num_iter + 1):
            run_str = 'run_' + str(i)
            pos_list = d[run_str]['pos']
            neg_list = d[run_str]['neg']
            unstable_list = []
            if 'slice' in folder_name:
                unstable_list = d[run_str]['h_unstable']
            pos_master_list.append(pos_list)
            neg_master_list.append(neg_list)
            if 'slice' in folder_name:
                unstable_master_list.append(unstable_list)
            num_superpixel = int(np.max(segments))
        afse = AFSE(pos_master_list, neg_master_list, num_superpixel, num_iter)
        ars = ARS(pos_master_list, neg_master_list, num_superpixel, num_iter)
        ccm = (1 - afse) * ars
        ccm_list.append(ccm)
    return ccm_list


def ccm_plot(ccm_list_lime, ccm_list_grid_lime, ccm_list_slice, dataset, model, show_plot=False):
    plt.figure()

    sns.kdeplot(ccm_list_lime, fill=True, color=plot_colors['lime'], label="LIME")
    sns.kdeplot(ccm_list_grid_lime, fill=True, color=plot_colors['GridLime'], label="GRID_LIME")
    sns.kdeplot(ccm_list_slice, fill=True, color=plot_colors['slice'], label="SLICE")

    title_str = dataset + ':' + model

    plt.legend()
    plt.title(title_str)
    plt.xlabel("CCM Scores")
    plt.ylabel("Density")
    plt.xlim([0, 1])
    plt.savefig('Final Plots/KDE plot for ' + title_str + '.png')
    if show_plot:
        plt.show()
    plt.plot()
    plt.close()


def generate_ccm_plots(save_dir_argument, show_plot=False, num_iter=10):
    lime_resnet_oxford_path = os.path.join(save_dir_argument, 'lime_resnet50_oxpets_results')
    grid_lime_resnet_oxford_path = os.path.join(save_dir_argument, 'GridLime_resnet50_oxpets_results')
    slice_resnet_oxford_path = os.path.join(save_dir_argument, 'slice_resnet50_oxpets_results')

    lime_inception_oxford_path = os.path.join(save_dir_argument, 'lime_inceptionv3_oxpets_results')
    grid_lime_inception_oxford_path = os.path.join(save_dir_argument, 'GridLime_inceptionv3_oxpets_results')
    slice_inception_oxford_path = os.path.join(save_dir_argument, 'slice_inceptionv3_oxpets_results')

    lime_resnet_pvoc_path = os.path.join(save_dir_argument, 'lime_resnet50_pvoc_results')
    grid_lime_resnet_pvoc_path = os.path.join(save_dir_argument, 'GridLime_resnet50_pvoc_results')
    slice_resnet_pvoc_path = os.path.join(save_dir_argument, 'slice_resnet50_pvoc_results')

    lime_inception_pvoc_path = os.path.join(save_dir_argument, 'lime_inceptionv3_pvoc_results')
    grid_lime_inception_pvoc_path = os.path.join(save_dir_argument, 'GridLime_inceptionv3_pvoc_results')
    slice_inception_pvoc_path = os.path.join(save_dir_argument, 'slice_inceptionv3_pvoc_results')

    lime_resnet_mscoco_path = os.path.join(save_dir_argument, 'lime_resnet50_mscoco_results')
    grid_lime_resnet_mscoco_path = os.path.join(save_dir_argument, 'GridLime_resnet50_mscoco_results')
    slice_resnet_mscoco_path = os.path.join(save_dir_argument, 'slice_resnet50_mscoco_results')

    lime_inception_mscoco_path = os.path.join(save_dir_argument, 'lime_inceptionv3_mscoco_results')
    grid_lime_inception_mscoco_path = os.path.join(save_dir_argument, 'GridLime_inceptionv3_mscoco_results')
    slice_inception_mscoco_path = os.path.join(save_dir_argument, 'slice_inceptionv3_mscoco_results')

    ccm_list_lime_resnet_oxpets = get_ccm_values(lime_resnet_oxford_path, num_iter=num_iter)
    ccm_list_grid_lime_resnet_oxpets = get_ccm_values(grid_lime_resnet_oxford_path, num_iter=num_iter)
    ccm_list_slice_resnet_oxpets = get_ccm_values(slice_resnet_oxford_path, num_iter=num_iter)
    ccm_plot(ccm_list_lime_resnet_oxpets, ccm_list_grid_lime_resnet_oxpets, ccm_list_slice_resnet_oxpets,
             'Oxford-IIIT Pets', 'Resnet50', show_plot)

    ccm_list_lime_inception_oxpets = get_ccm_values(lime_inception_oxford_path, num_iter=num_iter)
    ccm_list_grid_lime_inception_oxpets = get_ccm_values(grid_lime_inception_oxford_path, num_iter=num_iter)
    ccm_list_slice_inception_oxpets = get_ccm_values(slice_inception_oxford_path, num_iter=num_iter)
    ccm_plot(ccm_list_lime_inception_oxpets, ccm_list_grid_lime_inception_oxpets, ccm_list_slice_inception_oxpets,
             'Oxford-IIIT Pets', 'InceptionV3', show_plot)

    ccm_list_lime_resnet_pvoc = get_ccm_values(lime_resnet_pvoc_path, num_iter=num_iter)
    ccm_list_grid_lime_resnet_pvoc = get_ccm_values(grid_lime_resnet_pvoc_path, num_iter=num_iter)
    ccm_list_slice_resnet_pvoc = get_ccm_values(slice_resnet_pvoc_path, num_iter=num_iter)
    ccm_plot(ccm_list_lime_resnet_pvoc, ccm_list_grid_lime_resnet_pvoc, ccm_list_slice_resnet_pvoc,
             'Pascal VOC', 'Resnet50', show_plot)

    ccm_list_lime_inception_pvoc = get_ccm_values(lime_inception_pvoc_path, num_iter=num_iter)
    ccm_list_grid_lime_inception_pvoc = get_ccm_values(grid_lime_inception_pvoc_path, num_iter=num_iter)
    ccm_list_slice_inception_pvoc = get_ccm_values(slice_inception_pvoc_path, num_iter=num_iter)
    ccm_plot(ccm_list_lime_inception_pvoc, ccm_list_grid_lime_inception_pvoc, ccm_list_slice_inception_pvoc,
             'Pascal VOC', 'InceptionV3', show_plot)

    ccm_list_lime_resnet_mscoco = get_ccm_values(lime_resnet_mscoco_path, num_iter=num_iter)
    ccm_list_grid_lime_resnet_mscoco = get_ccm_values(grid_lime_resnet_mscoco_path, num_iter=num_iter)
    ccm_list_slice_resnet_mscoco = get_ccm_values(slice_resnet_mscoco_path, num_iter=num_iter)
    ccm_plot(ccm_list_lime_resnet_mscoco, ccm_list_grid_lime_resnet_mscoco, ccm_list_slice_resnet_mscoco,
             'MS-COCO', 'Resnet50', show_plot)

    ccm_list_lime_inception_mscoco = get_ccm_values(lime_inception_mscoco_path, num_iter=num_iter)
    ccm_list_grid_lime_inception_mscoco = get_ccm_values(grid_lime_inception_mscoco_path, num_iter=num_iter)
    ccm_list_slice_inception_mscoco = get_ccm_values(slice_inception_mscoco_path, num_iter=num_iter)
    ccm_plot(ccm_list_lime_inception_mscoco, ccm_list_grid_lime_inception_mscoco, ccm_list_slice_inception_mscoco,
             'MS-COCO', 'InceptionV3', show_plot)



def calculate_gto(gt_segment, segments, pos_list, top_k=5):
    pos_segments = pos_list[:top_k]
    mask_with_all_pos_seg = np.isin(segments, pos_segments).astype(np.uint8)

    mask_img_dataset_gt = np.isin(gt_segment.squeeze(0), 1).astype(np.uint8)

    intersection = np.logical_and(mask_with_all_pos_seg, mask_img_dataset_gt)
    metric_value = np.sum(intersection) / np.sum(mask_with_all_pos_seg)

    return metric_value


def generate_gto(folder_name, top_k=5, num_iter=10):
    transform_mask_inception = transforms.Compose([
        transforms.Resize((299, 299), transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])

    transform_mask_resnet = transforms.Compose([
        transforms.Resize((224, 224), transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])

    metric_value_list_overall = []

    for folder_img in os.listdir(folder_name):
        with open(folder_name + r'/' + folder_img + r'/run_info.pickle', 'rb') as file:
            d = pickle.load(file)
        with open(folder_name + r'/' + folder_img + r'/segments.pickle', 'rb') as file:
            d_segments = pickle.load(file)
        with open(folder_name + r'/' + folder_img + r'/ground_truth_mask.pickle', 'rb') as file:
            ground_truth_area = pickle.load(file)
        if 'resnet' in folder_name:
            transform_mask = transform_mask_resnet
        elif 'inception' in folder_name:
            transform_mask = transform_mask_inception
        else:
            transform_mask = None

        if 'pvoc' in folder_name:
            ground_truth_area = np.array(ground_truth_area, dtype=np.uint8)
        ground_truth_area = Image.fromarray(ground_truth_area)
        ground_truth_area_mask = transform_mask(ground_truth_area)

        metric_value_list_per_image = []
        for i in range(1, num_iter + 1):
            run_str = 'run_' + str(i)
            pos_list = d[run_str]['pos']
            neg_list = d[run_str]['neg']
            metric_value = calculate_gto(ground_truth_area_mask, d_segments, pos_list, top_k)
            metric_value_list_per_image.append(metric_value)
        metric_value_list_overall.append(np.mean(metric_value_list_per_image))
    return metric_value_list_overall


def generate_gto_plot(folder_list, k_min=1, k_max=6, show_plot=False, num_iter=10,dataset='Oxforf-IIIT Pets'):
    final_dict = {}
    k_values = range(k_min, k_max + 1)
    for folder in folder_list:
        metric_value_dict_k = {}
        for k in k_values:
            gto = generate_gto(folder, top_k=k, num_iter=num_iter)
            metric_value_dict_k[k] = gto
        final_dict[folder] = metric_value_dict_k

    # Put path as 'lime_resnet50_oxpets_results/' in Snellius
    lime_resnet = [final_dict[folder_list[0]][k] for k in k_values]
    grid_lime_resnet = [final_dict[folder_list[1]][k] for k in k_values]
    slice_resnet = [final_dict[folder_list[2]][k] for k in k_values]
    lime_inception = [final_dict[folder_list[3]][k] for k in k_values]
    grid_lime_inception = [final_dict[folder_list[4]][k] for k in k_values]
    slice_inception = [final_dict[folder_list[5]][k] for k in k_values]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    iqr_bar_lime_inception, mean_lime_inception = get_quartile_information(np.array(lime_inception))
    plt.plot(np.array(list(k_values)) - 0.1, np.mean(np.array(lime_inception), axis=1), label='LIME', marker='x',
             color=plot_colors['lime'])
    plt.errorbar(np.array(list(k_values)) - 0.1, mean_lime_inception, yerr=iqr_bar_lime_inception, fmt='o',
                 color=plot_colors['lime'], capsize=5, label='IQR', alpha=0.4)

    iqr_bar_grid_lime_inception, mean_grid_lime_inception = get_quartile_information(np.array(grid_lime_inception))
    plt.plot(np.array(list(k_values)), np.mean(np.array(grid_lime_inception), axis=1), label='GRID_LIME', marker='x',
             color=plot_colors['GridLime'])
    plt.errorbar(np.array(list(k_values)), mean_grid_lime_inception, yerr=iqr_bar_grid_lime_inception, fmt='o',
                 color=plot_colors['GridLime'], capsize=5, label='IQR', alpha=0.4)

    iqr_bar_slice_inception, mean_slice_inception = get_quartile_information(np.array(slice_inception))
    plt.plot(np.array(list(k_values)) + 0.1, np.mean(np.array(slice_inception), axis=1), label='SLICE', marker='x',
             color=plot_colors['slice'])
    plt.errorbar(np.array(list(k_values)) + 0.1, mean_slice_inception, yerr=iqr_bar_slice_inception, fmt='o',
                 color=plot_colors['slice'], capsize=5, label='IQR', alpha=0.4)

    plt.title(dataset+':InceptionV3')
    plt.xlabel('Top k superpixels')
    plt.ylabel('Ground Truth Overlap Value')
    plt.xticks(np.array(list(k_values)))
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    iqr_bar_lime_resnet, mean_lime_resnet = get_quartile_information(np.array(lime_resnet))
    plt.plot(np.array(list(k_values)) - 0.1, np.mean(np.array(lime_resnet), axis=1), label='LIME', marker='x',
             color=plot_colors['lime'])
    plt.errorbar(np.array(list(k_values)) - 0.1, mean_lime_resnet, yerr=iqr_bar_lime_resnet, fmt='o',
                 color=plot_colors['lime'], capsize=5, label='IQR', alpha=0.4)

    iqr_bar_grid_lime_resnet, mean_grid_lime_resnet = get_quartile_information(np.array(grid_lime_resnet))
    plt.plot(np.array(list(k_values)), np.mean(np.array(grid_lime_resnet), axis=1), label='GRID_LIME', marker='x',
             color=plot_colors['GridLime'])
    plt.errorbar(np.array(list(k_values)), mean_grid_lime_resnet, yerr=iqr_bar_grid_lime_resnet, fmt='o',
                 color=plot_colors['GridLime'], capsize=5, label='IQR', alpha=0.4)

    iqr_bar_slice_resnet, mean_slice_resnet = get_quartile_information(np.array(slice_resnet))
    plt.plot(np.array(list(k_values)) + 0.1, np.mean(np.array(slice_resnet), axis=1), label='SLICE', marker='x',
             color=plot_colors['slice'])
    plt.errorbar(np.array(list(k_values)) + 0.1, mean_slice_resnet, yerr=iqr_bar_slice_resnet, fmt='o',
                 color=plot_colors['slice'], capsize=5, label='IQR', alpha=0.4)

    plt.title(dataset+':Resnet50')
    plt.xlabel('Top k superpixels')
    plt.ylabel('Ground Truth Overlap Value')
    plt.xticks(np.array(list(k_values)))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    plt.savefig('Final Plots/GTO_Plots_'+dataset+'.png')
    if show_plot:
        plt.show()
    plt.close()


def compute_aopc_insertion(original_image, segments, positive_ranks, negative_ranks, model, sigma, po, sigma_constant,
                           num_ranks=5):
    original_image_tensor = original_image[0].unsqueeze(0)
    perturbation = np.zeros(len(np.unique(segments)))
    if sigma_constant == 'no':
        perturbed_image_tensor = perturb_image(perturbation, copy.deepcopy(original_image_tensor), segments, sigma)
    else:
        perturbed_image_tensor = perturb_image(perturbation, copy.deepcopy(original_image_tensor), segments)

    top_predicted_class, original_probability_of_prediction = get_original_image_prediction(
        perturbed_image_tensor.to(device).unsqueeze(0), model, device=device)

    if positive_ranks is None and negative_ranks is None:
        return float('nan')

    if num_ranks == -1:
        num_pos_ranks = len(positive_ranks)
    else:
        num_pos_ranks = num_ranks

    aopcs_pos = []
    if po == 'L':
        positive_ranks = np.flip(positive_ranks)

    for k in range(0, num_pos_ranks):
        perturbation = np.zeros(len(np.unique(segments)))
        cur_ranks = np.array(positive_ranks[0:k + 1], dtype=int)
        perturbation[cur_ranks] = 1
        if sigma_constant == 'no':
            perturbed_image_tensor = perturb_image(perturbation, copy.deepcopy(original_image_tensor), segments, sigma)
        else:
            perturbed_image_tensor = perturb_image(perturbation, copy.deepcopy(original_image_tensor), segments)

        with torch.no_grad():
            perturbation_prediction = model(perturbed_image_tensor.to(device).unsqueeze(0))
        perturbation_prediction_probability = torch.nn.functional.softmax(
            perturbation_prediction, dim=1).cpu().numpy()[0, top_predicted_class]
        probability_diff = perturbation_prediction_probability - original_probability_of_prediction
        aopcs_pos.append(probability_diff)

    if num_ranks == -1:
        num_neg_ranks = len(negative_ranks)
    else:
        num_neg_ranks = num_ranks

    aopcs_neg = []
    if po == 'L':
        negative_ranks = np.flip(negative_ranks)

    for k in range(0, num_neg_ranks):
        perturbation = np.zeros(len(np.unique(segments)))
        cur_ranks = np.array(negative_ranks[0:k + 1], dtype=int)
        perturbation[cur_ranks] = 1
        if sigma_constant == 'no':
            perturbed_image_tensor = perturb_image(perturbation, copy.deepcopy(original_image_tensor), segments, sigma)
        else:
            perturbed_image_tensor = perturb_image(perturbation, copy.deepcopy(original_image_tensor), segments)

        with torch.no_grad():
            perturbation_prediction = model(perturbed_image_tensor.to(device).unsqueeze(0))
        perturbation_prediction_probability = torch.nn.functional.softmax(
            perturbation_prediction, dim=1).cpu().numpy()[0, top_predicted_class]
        probability_diff = original_probability_of_prediction - perturbation_prediction_probability
        aopcs_neg.append(probability_diff)

    return np.mean(np.array(aopcs_pos)),  np.mean(np.array(aopcs_neg))


def compute_aopc_deletion(original_image, segments, positive_ranks, negative_ranks, model, sigma, po, sigma_constant,
                          num_ranks=5):
    original_image_tensor = original_image[0].unsqueeze(0)
    top_predicted_class, original_probability_of_prediction = get_original_image_prediction(original_image_tensor,
                                                                                            model, device=device)

    if positive_ranks is None and negative_ranks is None:
        return float('nan')

    if num_ranks == -1:
        num_pos_ranks = len(positive_ranks)
    else:
        num_pos_ranks = num_ranks

    aopcs_pos = []
    if po == 'L':
        positive_ranks = np.flip(positive_ranks)

    for k in range(0, num_pos_ranks):
        perturbation = np.ones(len(np.unique(segments)))
        cur_ranks = np.array(positive_ranks[0:k + 1], dtype=int)
        perturbation[cur_ranks] = 0
        if sigma_constant == 'no':
            perturbed_image_tensor = perturb_image(perturbation, copy.deepcopy(original_image_tensor), segments, sigma)
        else:
            perturbed_image_tensor = perturb_image(perturbation, copy.deepcopy(original_image_tensor), segments)

        with torch.no_grad():
            perturbation_prediction = model(perturbed_image_tensor.to(device).unsqueeze(0))
        perturbation_prediction_probability = torch.nn.functional.softmax(
            perturbation_prediction, dim=1).cpu().numpy()[0, top_predicted_class]
        probability_diff = original_probability_of_prediction - perturbation_prediction_probability
        aopcs_pos.append(probability_diff)

    if num_ranks == -1:
        num_neg_ranks = len(negative_ranks)
    else:
        num_neg_ranks = num_ranks

    aopcs_neg = []
    if po == 'L':
        negative_ranks = np.flip(negative_ranks)

    for k in range(0, num_neg_ranks):
        perturbation = np.ones(len(np.unique(segments)))
        cur_ranks = np.array(negative_ranks[0:k + 1], dtype=int)
        perturbation[cur_ranks] = 0
        if sigma_constant == 'no':
            perturbed_image_tensor = perturb_image(perturbation, copy.deepcopy(original_image_tensor), segments, sigma)
        else:
            perturbed_image_tensor = perturb_image(perturbation, copy.deepcopy(original_image_tensor), segments)

        with torch.no_grad():
            perturbation_prediction = model(perturbed_image_tensor.to(device).unsqueeze(0))
        perturbation_prediction_probability = torch.nn.functional.softmax(
            perturbation_prediction, dim=1).cpu().numpy()[0, top_predicted_class]
        probability_diff = perturbation_prediction_probability - original_probability_of_prediction
        aopcs_neg.append(probability_diff)

    return np.mean(np.array(aopcs_pos)),  np.mean(np.array(aopcs_neg))


def aggregate_score(original_image, segments, result_file_path, model, sigma, po, metric_func, sigma_constant):
    scores = []
    with open(result_file_path, 'rb') as f:
        run_dict = pickle.load(f)

    runs = list(run_dict.keys())
    for run in runs:
        run_info = run_dict[run]
        pos_ranks = run_info['pos']
        neg_ranks = run_info['neg']

        # Use the provided metric function
        pos_score, neg_score = metric_func(
            original_image, segments, positive_ranks=pos_ranks, negative_ranks=neg_ranks,
            model=model, sigma=sigma, po=po, sigma_constant=sigma_constant)
        scores.append({"pos": pos_score, "neg": neg_score})

    return scores


def generate_aopc_results(base_folder, output_dir, sigma_constant, po='M'):
    # Define fidelity functions
    fidelity_functions = {
        "aopc_ins": compute_aopc_insertion,
        "aopc_del": compute_aopc_deletion
    }

    # Define result folders
    folders = ["slice_inceptionv3_oxpets_results", "lime_inceptionv3_oxpets_results",
               "GridLime_inceptionv3_oxpets_results", "slice_inceptionv3_pvoc_results", "lime_inceptionv3_pvoc_results",
               "GridLime_inceptionv3_pvoc_results", "slice_resnet50_oxpets_results", "lime_resnet50_oxpets_results",
               "GridLime_resnet50_oxpets_results", "slice_resnet50_pvoc_results", "lime_resnet50_pvoc_results",
               "GridLime_resnet50_pvoc_results"]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop over metrics and datasets
    for metric_name, metric_func in fidelity_functions.items():
        for folder_name in folders:
            image_src_dir = dataset_paths['oxpets'] if "oxpets_" in folder_name else dataset_paths['pvoc']
            print(f"Processing metric: {metric_name} in folder: {folder_name}")
            folder_path = os.path.join(base_folder, folder_name)
            metric_output_dir = os.path.join(output_dir, metric_name)
            os.makedirs(metric_output_dir, exist_ok=True)

            model_name = "inceptionv3" if "inceptionv3" in folder_name else "resnet50"
            dataset_name = "oxpets" if "oxpets_" in folder_name else "pvoc"
            model, target_image_size = load_model_from_name(model_name)
            model_preprocess = get_preprocess_pipeline_for_model(target_image_size)
            dataset, image_filenames = download_dataset(dataset_name, model_preprocess=model_preprocess)

            result_data = {}
            for image_folder_name in os.listdir(folder_path):
                image_folder_path = os.path.join(folder_path, image_folder_name)

                # Define paths to .pickle files and image
                run_info_path = os.path.join(str(image_folder_path), 'run_info.pickle')
                segments_path = os.path.join(str(image_folder_path), 'segments.pickle')
                image_path = os.path.join(image_src_dir, f"{image_folder_name}.jpg")

                # Check for necessary files
                if not os.path.exists(run_info_path) or not os.path.exists(segments_path) or not os.path.exists(
                        image_path):
                    print(f"Skipping {image_folder_path} due to missing files.")
                    continue

                # Load run_info and segments
                with open(run_info_path, 'rb') as run_file:
                    run_info = pickle.load(run_file)
                with open(segments_path, 'rb') as seg_file:
                    segments = pickle.load(seg_file)

                # Determine model based on folder name
                xai_name = (
                    "grid_lime" if "GridLime_" in folder_name else
                    "lime" if "lime_" in folder_name else
                    "slice"
                )

                # Use aggregate_score to compute metrics
                try:
                    metric_scores = aggregate_score(
                        original_image=dataset[image_filenames.index(image_path)],
                        segments=segments,
                        result_file_path=run_info_path,
                        model=model,
                        sigma=run_info['run_1']['sel_sigma'],  # Use sel_sigma from run_info
                        po=po,
                        metric_func=metric_func,
                        sigma_constant=sigma_constant
                    )

                    key = f"{dataset_name}_{model_name}_{xai_name}_{image_folder_name}"
                    result_data[key] = metric_scores
                except Exception as e:
                    print(f"Error processing {image_folder_path}: {e}")
                    continue

            # Save results for the current metric and folder
            output_file_path = os.path.join(metric_output_dir, f"{folder_name}_{metric_name}_{po}.pickle")
            with open(output_file_path, 'wb') as output_file:
                pickle.dump(result_data, output_file)

            print(f"Saved AOPC results for {folder_name} to {output_file_path}")


def plot_aopc_results(base_folder, results_folder, output_file='Final Plots/aopc_ins_del_combined.png',
                      show_plot=False, sigma_constant='no'):
    # Generate AOPC results and save
    generate_aopc_results(base_folder, results_folder, sigma_constant, po='M')

    # Define both metrics
    metric_keys = ['aopc_ins', 'aopc_del']

    # Initialize dictionaries to store results for both metrics
    aopc_dicts = {key: {} for key in metric_keys}

    # Process each metric
    for metric_key in metric_keys:
        results_dir = os.path.join(results_folder, metric_key)
        files = os.listdir(results_dir)

        for file in files:
            with open(os.path.join(results_dir, file), 'rb') as f:
                data = pickle.load(f)

            img_keys = data.keys()
            aopcs = []
            pixels = "pos" if 'ins' in metric_key else "neg"  # Determine pos/neg based on metric_key

            for img_key in img_keys:
                img_res = data[img_key]

                if isinstance(img_res, np.ndarray):
                    aopcs.append(np.mean(img_res[0]))
                elif isinstance(img_res, list) and all(isinstance(item, dict) for item in img_res):
                    mean_pos_neg = [np.mean(entry[pixels]) for entry in img_res]
                    aopcs.append(np.mean(mean_pos_neg))

            key_parts = ["oxpets" if "oxpets_" in file else "pvoc",
                         "inceptionv3" if "inceptionv3_" in file else "resnet50",
                         "GridLime" if "GridLime_" in file else "GridSlice" if "GridSlice_" in file else
                         "lime" if "lime_" in file else "slice"]
            aopc_dict_key = '_'.join([key_parts[0], key_parts[1], key_parts[2]])

            aopc_dicts[metric_key][aopc_dict_key] = np.array(aopcs)

    # Plotting
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['font.weight'] = 'bold'

    subplot_size = 8
    fig, axes = plt.subplots(2, 4, figsize=(2 * subplot_size, subplot_size))

    title_index = ['Oxford-IIIT Pets:InceptionV3', 'Oxford-IIIT Pets:ResNet50',
                   'Pascal VOC:Inception V3', 'Pascal VOC:ResNet50']
    slice_keys = ['oxpets_inceptionv3_slice', 'oxpets_resnet50_slice',
                  'pvoc_inceptionv3_slice', 'pvoc_resnet50_slice']
    lime_keys = ['oxpets_inceptionv3_lime', 'oxpets_resnet50_lime',
                 'pvoc_inceptionv3_lime', 'pvoc_resnet50_lime']
    grid_lime_keys = ['oxpets_inceptionv3_GridLime', 'oxpets_resnet50_GridLime',
                      'pvoc_inceptionv3_GridLime', 'pvoc_resnet50_GridLime']

    handles_list = []
    labels_list = []

    # Iterate over metrics and create subplots
    for row, metric_key in enumerate(metric_keys):
        dict_analysis = aopc_dicts[metric_key]

        for col in range(4):
            ax = axes[row, col]

            slice_data = replace_nan_with_median(dict_analysis[slice_keys[col]])
            lime_data = replace_nan_with_median(dict_analysis[lime_keys[col]])
            grid_lime_data = replace_nan_with_median(dict_analysis[grid_lime_keys[col]])

            if 'ResNet50' in title_index[col]:
                lower_cap, upper_cap = (0, 100)
            else:
                lower_cap, upper_cap = (0, 100)

            slice_data = cap_values(slice_data, lower_cap, upper_cap)
            lime_data = cap_values(lime_data, lower_cap, upper_cap)
            grid_lime_data = cap_values(grid_lime_data, lower_cap, upper_cap)

            sns.ecdfplot(slice_data, color=plot_colors['slice'], linewidth=1.5, ax=ax, label='SLICE')
            sns.ecdfplot(lime_data, color=plot_colors['lime'], linewidth=1.5, ax=ax, label='LIME')
            sns.ecdfplot(grid_lime_data, color=plot_colors['GridLime'], linewidth=1.5, ax=ax, label='GRID LIME')

            plot_title = "AOPC Insertion Scores" if metric_key == 'aopc_ins' else "AOPC Deletion Scores"
            ax.set_title(f"{title_index[col]}", fontsize=12, fontweight='bold')
            ax.set_xlabel(plot_title, fontsize=12, fontweight='bold')
            ax.set_ylabel("Cumulative Probability", fontsize=12, fontweight='bold')

            handles, labels = ax.get_legend_handles_labels()
            handles_list.extend(handles)
            labels_list.extend(labels)

    # Display a unique legend for all subplots
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles_list, labels_list)) if l not in labels_list[:i]]
    fig.legend(*zip(*unique), loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize=12, ncol=4)

    plt.tight_layout()
    plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


def auc_value(scores):
    """
    Compute normalized area under the curve using the author's logic.
    If more than one score is available:
      adjusted_sum = (sum(scores) - scores[0]/2 - scores[-1]/2)
      auc = adjusted_sum / (number of intervals)   [i.e. len(scores)-1]
    If only one score is available, return that value.
    """
    scores = np.array(scores)
    if scores.size > 1:
        adjusted_sum = scores.sum() - scores[0] / 2 - scores[-1] / 2
        return adjusted_sum / (scores.size - 1)
    elif scores.size == 1:
        return scores[0]
    else:
        return 0.0


def compute_auc_insertion(original_image, segments, positive_ranks, negative_ranks,
                          model, sigma, po, sigma_constant, num_ranks=5):
    """
    Compute AUC scores for insertion.

    For insertion:
      - Start from a fully perturbed (e.g. blurred) image.
      - Progressively "restore" segments (i.e. insert original pixels)
        in the order specified by positive_ranks or negative_ranks.
      - Record the change in the top-class probability relative to
        the baseline (fully perturbed image).
      - Compute AUC using the adjusted sum formula and then normalize
        by the number of ranks.
    """
    # original_image is assumed to be a tuple/list where element 0 is a tensor [C,H,W]
    original_tensor = original_image[0].unsqueeze(0)  # add batch dimension: [1, C, H, W]
    num_segments = len(np.unique(segments))

    # Create baseline image (fully perturbed image: no segments inserted)
    perturbation_baseline = np.zeros(num_segments)  # 0 means segment remains perturbed
    if sigma_constant == 'no':
        baseline_img = perturb_image(perturbation_baseline, copy.deepcopy(original_tensor), segments, sigma)
    else:
        baseline_img = perturb_image(perturbation_baseline, copy.deepcopy(original_tensor), segments)
    baseline_img = baseline_img.unsqueeze(0)

    # Get the top predicted class and baseline probability (on the fully perturbed image)
    top_pred_class, baseline_prob = get_original_image_prediction(baseline_img.to(device), model, device=device)

    # --- Positive Insertion ---
    ins_pos_scores = [] # for ins positive superpixels, we do (pred_probability - fully perturbed probability)
    if positive_ranks is not None and len(positive_ranks) > 0:
        if po == 'L':
            positive_ranks = np.flip(positive_ranks)
        num_pos_ranks = min(len(positive_ranks), num_ranks)
        for k in range(num_pos_ranks):
            # Create a perturbation encoding: 1 means "restore" that segment.
            perturbation = np.zeros(num_segments)
            cur_ranks = np.array(positive_ranks[0:k+1], dtype=int)
            perturbation[cur_ranks] = 1

            if sigma_constant == 'no':
                perturbed_img = perturb_image(perturbation, copy.deepcopy(original_tensor), segments, sigma)
            else:
                perturbed_img = perturb_image(perturbation, copy.deepcopy(original_tensor), segments)
            perturbed_img = perturbed_img.unsqueeze(0)

            with torch.no_grad():
                _, pred_prob = get_original_image_prediction(perturbed_img.to(device), model, device=device)
            # For insertion, record the increase in probability relative to baseline.
            ins_pos_scores.append(pred_prob - baseline_prob)
    else:
        num_pos_ranks = 0

    # --- Negative Insertion ---
    ins_neg_scores = [] # for ins negative superpixels, we do (fully perturbed probability - pred_probability)
    if negative_ranks is not None and len(negative_ranks) > 0:
        if po == 'L':
            negative_ranks = np.flip(negative_ranks)
        num_neg_ranks = min(len(negative_ranks), num_ranks)
        for k in range(num_neg_ranks):
            perturbation = np.zeros(num_segments)
            cur_ranks = np.array(negative_ranks[0:k+1], dtype=int)
            perturbation[cur_ranks] = 1

            if sigma_constant == 'no':
                perturbed_img = perturb_image(perturbation, copy.deepcopy(original_tensor), segments, sigma)
            else:
                perturbed_img = perturb_image(perturbation, copy.deepcopy(original_tensor), segments)
            perturbed_img = perturbed_img.unsqueeze(0)

            with torch.no_grad():
                _, pred_prob = get_original_image_prediction(perturbed_img.to(device), model, device=device)
            # For negative insertion, the probability is expected to drop.
            ins_neg_scores.append(baseline_prob - pred_prob)
    else:
        num_neg_ranks = 0

    # Compute normalized AUC scores following the author's logic.
    if len(ins_pos_scores) > 1:
        pos_auc = auc_value(ins_pos_scores) / num_pos_ranks
    elif len(ins_pos_scores) == 1:
        pos_auc = (0.5 * (ins_pos_scores[0] - baseline_prob) * 1 + (ins_pos_scores[0] - baseline_prob) * 1) / num_pos_ranks
    else:
        pos_auc = 0.0

    if len(ins_neg_scores) > 1:
        neg_auc = auc_value(ins_neg_scores) / num_neg_ranks
    elif len(ins_neg_scores) == 1:
        neg_auc = (0.5 * (ins_neg_scores[0] - baseline_prob) * 1 + (ins_neg_scores[0] - baseline_prob) * 1) / num_neg_ranks
    else:
        neg_auc = 0.0

    return pos_auc, neg_auc


def compute_auc_deletion(original_image, segments, positive_ranks, negative_ranks,
                         model, sigma, po, sigma_constant, num_ranks=5):
    """
    Compute AUC scores for deletion.

    For deletion:
      - Start from the original (unperturbed) image.
      - Progressively "delete" segments (i.e. replace with blurred/perturbed values)
        according to the provided ranking.
      - Record the drop (or change) in the top-class probability relative to the original.
      - Compute AUC using the adjusted sum formula and normalize by the number of ranks.
    """
    original_tensor = original_image[0].unsqueeze(0)
    with torch.no_grad():
        top_pred_class, original_prob = get_original_image_prediction(original_tensor.to(device), model, device=device)

    # --- Positive Deletion ---
    del_pos_scores = [] # for del positive superpixels, we do (original probability - pred_probability)
    if positive_ranks is not None and len(positive_ranks) > 0:
        if po == 'L':
            positive_ranks = np.flip(positive_ranks)
        num_pos_ranks = min(len(positive_ranks), num_ranks)
        for k in range(num_pos_ranks):
            num_segments = len(np.unique(segments))
            # For deletion, start with an encoding of ones (original image) and set selected segments to 0.
            perturbation = np.ones(num_segments)
            cur_ranks = np.array(positive_ranks[0:k+1], dtype=int)
            perturbation[cur_ranks] = 0

            if sigma_constant == 'no':
                perturbed_img = perturb_image(perturbation, copy.deepcopy(original_tensor), segments, sigma)
            else:
                perturbed_img = perturb_image(perturbation, copy.deepcopy(original_tensor), segments)
            perturbed_img = perturbed_img.unsqueeze(0)

            with torch.no_grad():
                _, pred_prob = get_original_image_prediction(perturbed_img.to(device), model, device=device)
            # Record the drop in probability relative to the original image.
            del_pos_scores.append(-1*(original_prob - pred_prob)) # using - sign to have lower auc for explanations
    else:
        num_pos_ranks = 0

    # --- Negative Deletion ---
    del_neg_scores = [] # for del negative superpixels, we do (pred_probability - original probability)
    if negative_ranks is not None and len(negative_ranks) > 0:
        if po == 'L':
            negative_ranks = np.flip(negative_ranks)
        num_neg_ranks = min(len(negative_ranks), num_ranks)
        for k in range(num_neg_ranks):
            perturbation = np.ones(len(np.unique(segments)))
            cur_ranks = np.array(negative_ranks[0:k+1], dtype=int)
            perturbation[cur_ranks] = 0

            if sigma_constant == 'no':
                perturbed_img = perturb_image(perturbation, copy.deepcopy(original_tensor), segments, sigma)
            else:
                perturbed_img = perturb_image(perturbation, copy.deepcopy(original_tensor), segments)
            perturbed_img = perturbed_img.unsqueeze(0)

            with torch.no_grad():
                _, pred_prob = get_original_image_prediction(perturbed_img.to(device), model, device=device)
            # For negative deletion, record the increase in probability relative to the original.
            del_neg_scores.append(-1* (pred_prob - original_prob))  # using - sign to have lower auc for explanations
    else:
        num_neg_ranks = 0

    if len(del_pos_scores) > 1:
        pos_auc = auc_value(del_pos_scores) / num_pos_ranks
    elif len(del_pos_scores) == 1:
        pos_auc = (0.5 * (del_pos_scores[0] - original_prob) * 1 + (del_pos_scores[0] - original_prob) * 1) / num_pos_ranks
    else:
        pos_auc = 0.0

    if len(del_neg_scores) > 1:
        neg_auc = auc_value(del_neg_scores) / num_neg_ranks
    elif len(del_neg_scores) == 1:
        neg_auc = (0.5 * (del_neg_scores[0] - original_prob) * 1 + (del_neg_scores[0] - original_prob) * 1) / num_neg_ranks
    else:
        neg_auc = 0.0

    return pos_auc, neg_auc


def generate_auc_results(base_folder, output_dir, sigma_constant='no', po='M'):
    # Define fidelity functions for AUC
    fidelity_functions = {
        "auc_ins": compute_auc_insertion,
        "auc_del": compute_auc_deletion
    }

    # Update the xai results folders list as needed.

    folders = ["slice_inceptionv3_pvoc_results", "lime_inceptionv3_pvoc_results", "GridLime_inceptionv3_pvoc_results",
               "slice_resnet50_pvoc_results", "lime_resnet50_pvoc_results", "GridLime_resnet50_pvoc_results",
               "slice_inceptionv3_oxpets_results", "lime_inceptionv3_oxpets_results", "GridLime_inceptionv3_oxpets_results",
               "slice_resnet50_oxpets_results", "lime_resnet50_oxpets_results", "GridLime_resnet50_oxpets_results",
               "slice_inceptionv3_mscoco_results", "lime_inceptionv3_mscoco_results", "GridLime_inceptionv3_mscoco_results",
               "slice_resnet50_mscoco_results", "lime_resnet50_mscoco_results", "GridLime_resnet50_mscoco_results"]

    os.makedirs(output_dir, exist_ok=True)

    for metric_name, metric_func in fidelity_functions.items():
        for folder_name in folders:
            image_src_dir = (dataset_paths['oxpets'] if "oxpets_" in folder_name
                             else dataset_paths['pvoc'] if "pvoc_" in folder_name
                             else dataset_paths['coco'])
            print(f"Processing metric: {metric_name} in folder: {folder_name}")
            folder_path = os.path.join(base_folder, folder_name)
            metric_output_dir = os.path.join(output_dir, metric_name)
            os.makedirs(metric_output_dir, exist_ok=True)

            model_name = "inceptionv3" if "inceptionv3" in folder_name else "resnet50"
            dataset_name = ("oxpets" if "oxpets_" in folder_name
                            else "pvoc" if "pvoc_" in folder_name
                            else "coco")
            model, target_image_size = load_model_from_name(model_name)
            model_preprocess = (get_preprocess_pipeline_for_model_mscoco_version(target_image_size)
                                if "mscoco_" in folder_name
                                else get_preprocess_pipeline_for_model(target_image_size))
            dataset, image_filenames = download_dataset(dataset_name, model_preprocess=model_preprocess)

            result_data = {}
            for image_folder_name in os.listdir(folder_path):
                image_folder_path = os.path.join(folder_path, image_folder_name)
                run_info_path = os.path.join(str(image_folder_path), 'run_info.pickle')
                segments_path = os.path.join(str(image_folder_path), 'segments.pickle')
                image_path = os.path.join(image_src_dir, f"{image_folder_name}.jpg")

                if not os.path.exists(run_info_path) or not os.path.exists(segments_path) or not os.path.exists(image_path):
                    print(f"Skipping {image_folder_path} due to missing files.")
                    continue

                with open(run_info_path, 'rb') as run_file:
                    run_info = pickle.load(run_file)
                with open(segments_path, 'rb') as seg_file:
                    segments = pickle.load(seg_file)

                xai_name = ("grid_lime" if "GridLime_" in folder_name
                            else "lime" if "lime_" in folder_name
                            else "slice")
                try:
                    metric_scores = aggregate_score(
                        original_image=dataset[image_filenames.index(image_path)],
                        segments=segments,
                        result_file_path=run_info_path,
                        model=model,
                        sigma=run_info['run_1']['sel_sigma'],
                        po=po,
                        metric_func=metric_func,
                        sigma_constant=sigma_constant
                    )
                    key = f"{dataset_name}_{model_name}_{xai_name}_{image_folder_name}"
                    result_data[key] = metric_scores
                except Exception as e:
                    print(f"Error processing {image_folder_path}: {e}")
                    continue

                # Save results for the current metric and folder.
            output_file_path = os.path.join(metric_output_dir, f"{folder_name}_{metric_name}_{po}.pickle")
            with open(output_file_path, 'wb') as output_file:
                pickle.dump(result_data, output_file)
            print(f"Saved AUC results for {folder_name} to {output_file_path}")


def plot_auc_results(base_folder, results_folder, output_file='Final Plots/auc_ins_del_combined.png',
                     show_plot=False, sigma_constant='no'):
    # Generate AUC results and save
    generate_auc_results(base_folder, results_folder, sigma_constant, po='M')

    # Define both metrics
    metric_keys = ['auc_ins', 'auc_del']

    # Initialize dictionaries to store results for both metrics
    aopc_dicts = {key: {} for key in metric_keys}

    # Process each metric
    for metric_key in metric_keys:
        results_dir = os.path.join(results_folder, metric_key)
        files = os.listdir(results_dir)

        for file in files:
            with open(os.path.join(results_dir, file), 'rb') as f:
                data = pickle.load(f)

            img_keys = data.keys()
            aopcs = []
            pixels = "pos" if 'ins' in metric_key else "neg"  # Determine pos/neg based on metric_key

            for img_key in img_keys:
                img_res = data[img_key]

                if isinstance(img_res, np.ndarray):
                    aopcs.append(np.mean(img_res[0]))
                elif isinstance(img_res, list) and all(isinstance(item, dict) for item in img_res):
                    mean_pos_neg = [np.mean(entry[pixels]) for entry in img_res]
                    aopcs.append(np.mean(mean_pos_neg))

            key_parts = ["oxpets" if "oxpets_" in file else "pvoc" if "pvoc_" in file else "coco",
                         "inceptionv3" if "inceptionv3_" in file else "resnet50",
                         "GridLime" if "GridLime_" in file else "GridSlice" if "GridSlice_" in file else
                         "lime" if "lime_" in file else "slice"]
            aopc_dict_key = '_'.join([key_parts[0], key_parts[1], key_parts[2]])

            aopc_dicts[metric_key][aopc_dict_key] = np.array(aopcs)

    # Plotting
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['font.weight'] = 'bold'

    subplot_size = 12
    fig, axes = plt.subplots(2, 6, figsize=(2.3 * subplot_size, 0.8 * subplot_size))

    title_index = ['Oxford-IIIT Pets:InceptionV3', 'Oxford-IIIT Pets:ResNet50',
                   'Pascal VOC:Inception V3', 'Pascal VOC:ResNet50',
                   'MSCOCO: Inception V3', 'MSCOCO: ResNet50']
    slice_keys = ['oxpets_inceptionv3_slice', 'oxpets_resnet50_slice',
                  'pvoc_inceptionv3_slice', 'pvoc_resnet50_slice',
                  'coco_inceptionv3_slice', 'coco_resnet50_slice']
    lime_keys = ['oxpets_inceptionv3_lime', 'oxpets_resnet50_lime',
                 'pvoc_inceptionv3_lime', 'pvoc_resnet50_lime',
                 'coco_inceptionv3_lime', 'coco_resnet50_lime']
    grid_lime_keys = ['oxpets_inceptionv3_GridLime', 'oxpets_resnet50_GridLime',
                      'pvoc_inceptionv3_GridLime', 'pvoc_resnet50_GridLime',
                      'coco_inceptionv3_GridLime', 'coco_resnet50_GridLime']

    handles_list = []
    labels_list = []

    # Iterate over metrics and create subplots
    for row, metric_key in enumerate(metric_keys):
        dict_analysis = aopc_dicts[metric_key]

        for col in range(6):
            ax = axes[row, col]

            slice_data = replace_nan_with_median(dict_analysis[slice_keys[col]])
            lime_data = replace_nan_with_median(dict_analysis[lime_keys[col]])
            grid_lime_data = replace_nan_with_median(dict_analysis[grid_lime_keys[col]])

            if 'ResNet50' in title_index[col]:
                lower_cap, upper_cap = (0, 100)
            else:
                lower_cap, upper_cap = (0, 100)

            slice_data = cap_values(slice_data, lower_cap, upper_cap)
            lime_data = cap_values(lime_data, lower_cap, upper_cap)
            grid_lime_data = cap_values(grid_lime_data, lower_cap, upper_cap)

            sns.ecdfplot(slice_data, color=plot_colors['slice'], linewidth=1.5, ax=ax, label='SLICE')
            sns.ecdfplot(lime_data, color=plot_colors['lime'], linewidth=1.5, ax=ax, label='LIME')
            sns.ecdfplot(grid_lime_data, color=plot_colors['GridLime'], linewidth=1.5, ax=ax, label='GRID LIME')

            plot_title = "AUC Insertion Scores" if metric_key == 'auc_ins' else "AUC Deletion Scores"
            ax.set_title(f"{title_index[col]}", fontsize=12, fontweight='bold')
            ax.set_xlabel(plot_title, fontsize=12, fontweight='bold')
            ax.set_ylabel("Cumulative Probability", fontsize=12, fontweight='bold')

            handles, labels = ax.get_legend_handles_labels()
            handles_list.extend(handles)
            labels_list.extend(labels)

    # Display a unique legend for all subplots
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles_list, labels_list)) if l not in labels_list[:i]]
    fig.legend(*zip(*unique), loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize=12, ncol=4)

    plt.tight_layout()
    plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


# Function to calculate the probability of observing negative AOPC scores
def calculate_negative_prob(data_dict):
    return {key: np.mean(values < 0) for key, values in data_dict.items()}


def get_negative_probabilities_table(base_folder, results_folder):
    # Define both metrics
    metric_keys = ['aopc_ins', 'aopc_del']

    if not os.path.isdir(os.path.join(base_folder, results_folder)):
        os.makedirs(os.path.join(base_folder, results_folder))

    # Initialize dictionaries to store results for both metrics
    aopc_dicts = {key: {} for key in metric_keys}

    # Process each metric
    for metric_key in metric_keys:
        results_dir = os.path.join(base_folder, 'AOPC Results', metric_key)
        files = os.listdir(results_dir)

        for file in files:
            with open(os.path.join(results_dir, file), 'rb') as f:
                data = pickle.load(f)

            img_keys = data.keys()
            aopcs = []
            pixels = "pos" if 'ins' in metric_key else "neg"  # Determine pos/neg based on metric_key

            for img_key in img_keys:
                img_res = data[img_key]

                if isinstance(img_res, np.ndarray):
                    aopcs.append(np.mean(img_res[0]))
                elif isinstance(img_res, list) and all(isinstance(item, dict) for item in img_res):
                    mean_pos_neg = [np.mean(entry[pixels]) for entry in img_res]
                    aopcs.append(np.mean(mean_pos_neg))

            key_parts = [
                "oxpets" if "oxpets_" in str(file) else "coco" if "mscoco_" in str(file) else "pvoc",
                "inceptionv3" if "inceptionv3_" in str(file) else "resnet50",
                "gridlime" if "GridLime_" in str(file) else "lime" if "lime_" in str(file) else "slice"
            ]

            aopc_dict_key = '_'.join([key_parts[0], key_parts[1], key_parts[2]])

            aopc_dicts[metric_key][aopc_dict_key] = np.array(aopcs)

    # Calculate probabilities for each method and metric
    negative_probs = {}
    for metric_key in metric_keys:
        negative_probs[metric_key] = calculate_negative_prob(aopc_dicts[metric_key])

    # Create the table as a pandas DataFrame
    methods = ["LIMEins", "SLICEins", "GRIDLIMEins", "LIMEdel", "SLICEdel", "GRIDLIMEdel"]
    columns = ["O I", "O R", "P I", "P R", "C I", "C R"]
    data = [
        [
            negative_probs['aopc_ins']['oxpets_inceptionv3_lime'],
            negative_probs['aopc_ins']['oxpets_resnet50_lime'],
            negative_probs['aopc_ins']['pvoc_inceptionv3_lime'],
            negative_probs['aopc_ins']['pvoc_resnet50_lime'],
            negative_probs['aopc_ins']['coco_inceptionv3_lime'],
            negative_probs['aopc_ins']['coco_resnet50_lime']
        ],
        [
            negative_probs['aopc_ins']['oxpets_inceptionv3_slice'],
            negative_probs['aopc_ins']['oxpets_resnet50_slice'],
            negative_probs['aopc_ins']['pvoc_inceptionv3_slice'],
            negative_probs['aopc_ins']['pvoc_resnet50_slice'],
            negative_probs['aopc_ins']['coco_inceptionv3_slice'],
            negative_probs['aopc_ins']['coco_resnet50_slice']
        ],
        [
            negative_probs['aopc_ins']['oxpets_inceptionv3_gridlime'],
            negative_probs['aopc_ins']['oxpets_resnet50_gridlime'],
            negative_probs['aopc_ins']['pvoc_inceptionv3_gridlime'],
            negative_probs['aopc_ins']['pvoc_resnet50_gridlime'],
            negative_probs['aopc_ins']['coco_inceptionv3_gridlime'],
            negative_probs['aopc_ins']['coco_resnet50_gridlime']
        ],
        [
            negative_probs['aopc_del']['oxpets_inceptionv3_lime'],
            negative_probs['aopc_del']['oxpets_resnet50_lime'],
            negative_probs['aopc_del']['pvoc_inceptionv3_lime'],
            negative_probs['aopc_del']['pvoc_resnet50_lime'],
            negative_probs['aopc_del']['coco_inceptionv3_lime'],
            negative_probs['aopc_del']['coco_resnet50_lime']
        ],
        [
            negative_probs['aopc_del']['oxpets_inceptionv3_slice'],
            negative_probs['aopc_del']['oxpets_resnet50_slice'],
            negative_probs['aopc_del']['pvoc_inceptionv3_slice'],
            negative_probs['aopc_del']['pvoc_resnet50_slice'],
            negative_probs['aopc_del']['coco_inceptionv3_slice'],
            negative_probs['aopc_del']['coco_resnet50_slice']
        ],
        [
            negative_probs['aopc_del']['oxpets_inceptionv3_gridlime'],
            negative_probs['aopc_del']['oxpets_resnet50_gridlime'],
            negative_probs['aopc_del']['pvoc_inceptionv3_gridlime'],
            negative_probs['aopc_del']['pvoc_resnet50_gridlime'],
            negative_probs['aopc_del']['coco_inceptionv3_gridlime'],
            negative_probs['aopc_del']['coco_resnet50_gridlime']
        ]
    ]

    # Save the table
    df = pd.DataFrame(data, columns=columns, index=methods)
    # print("Table of Negative Probabilities:")
    save_path = os.path.join(base_folder, results_folder, 'Table_of_Negative_Probabilities.csv')
    df.to_csv(save_path)


# Function to perform Wilcoxon signed-rank test
def get_wilcoxon_test_result(data1, data2):
    """
    Perform a Wilcoxon signed-rank test and return W-value, p-value, median difference, and count of negative differences.
    """
    result = pg.wilcoxon(data1, data2, alternative='greater')
    w_value = result['W-val'][0]
    p_value = result['p-val'][0]
    median_diff = np.median(data1 - data2)
    neg_count = np.sum((data1 - data2) < 0)  # Count negative differences
    return w_value, p_value, median_diff, neg_count


def perform_wilcoxon_test(base_folder, results_folder, method='aopc'):
    # Define metrics
    metric_keys = [f'{method}_ins', f'{method}_del']
    comparisons = [("slice", "lime"), ("slice", "gridlime")]  # Compare SLICE vs LIME, SLICE vs GridLime

    if not os.path.isdir(os.path.join(base_folder, results_folder)):
        os.makedirs(os.path.join(base_folder, results_folder))

    # Storage for AOPC scores
    metric_dict = {key: {} for key in metric_keys}

    # Load AOPC scores
    for metric_key in metric_keys:
        results_dir = os.path.join(base_folder, f'{method.upper()} Results', metric_key)
        files = os.listdir(results_dir)

        for file in files:
            with open(os.path.join(results_dir, file), 'rb') as f:
                data = pickle.load(f)

            img_keys = data.keys()
            values = []
            pixels = "pos" if 'ins' in metric_key else "neg"  # Determine pos/neg based on metric_key

            for img_key in img_keys:
                img_res = data[img_key]
                if isinstance(img_res, np.ndarray):
                    values.append(np.mean(img_res[0]))
                elif isinstance(img_res, list) and all(isinstance(item, dict) for item in img_res):
                    mean_pos_neg = [np.mean(entry[pixels]) for entry in img_res]
                    values.append(np.mean(mean_pos_neg))

            key_parts = [
                "oxpets" if "oxpets_" in str(file) else "coco" if "mscoco_" in str(file) else "pvoc",
                "inceptionv3" if "inceptionv3_" in str(file) else "resnet50",
                "gridlime" if "GridLime_" in str(file) else "lime" if "lime_" in str(file) else "slice"
            ]
            metric_dict_key = '_'.join([key_parts[0], key_parts[1], key_parts[2]])
            metric_dict[metric_key][metric_dict_key] = np.array(values)

    # Create results table
    columns = ["Test", "D:M", "W", "p-value", "M", "Neg. Count"]
    table_data = []

    # Model-Dataset Configurations
    datasets = {"oxpets": "O", "pvoc": "P", "coco": "C"}
    models = {"inceptionv3": "I", "resnet50": "R"}

    # Iterate over insertion and deletion metrics
    for metric_key in metric_keys:
        metric_label = "Insertion" if metric_key == f"{method}_ins" else "Deletion"
        table_data.append(["", "", "", metric_label, "", ""])

        for (method1, method2) in comparisons:  # Compare SLICE vs LIME, SLICE vs GRIDLIME
            for dataset, dataset_label in datasets.items():
                for model, model_label in models.items():
                    key1 = f"{dataset}_{model}_{method1}"
                    key2 = f"{dataset}_{model}_{method2}"

                    if key1 in metric_dict[metric_key] and key2 in metric_dict[metric_key]:
                        data1 = metric_dict[metric_key][key1]
                        data2 = metric_dict[metric_key][key2]

                        if len(data1) > 0 and len(data2) > 0:
                            # Get Wilcoxon test results
                            w, p, median_diff, neg_count = get_wilcoxon_test_result(data1, data2)

                            # Format table entry
                            test_name = f"{method.upper()}({method1[0].upper()},{method2[0].upper()})"
                            dataset_model = f"{dataset_label}:{model_label}"
                            table_data.append([test_name, dataset_model, w, f"{p:.2e}", median_diff, neg_count])

    # Convert to DataFrame
    df_results = pd.DataFrame(table_data, columns=columns)

    # Save the results table
    # print("Wilcoxon Rank Test Results - AOPC")
    save_path = os.path.join(base_folder, results_folder, f'Wilcoxon_Rank_Test_Results_{method.upper()}.csv')
    df_results.to_csv(save_path, index=False)
