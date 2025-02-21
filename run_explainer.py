import argparse
import os
import shutil
import importlib
import numpy as np
import pickle
import time
import random
import warnings
from torch.utils.data import Subset

from models.models import load_model_from_name, get_preprocess_pipeline_for_model, \
    get_preprocess_pipeline_for_model_mscoco_version
from utils.constants import explainer_class_mapping
from utils.helper import download_dataset, get_segment_labels_for_image_pixels, save_colored_output_image, COCODataset

warnings.filterwarnings("ignore")


def run_explainer_on_pre_trained_model(arguments):
    try:
        if arguments.explain_model == 'GridLime':
            segmentation_strategy = 'grid'
            segmentation_sigma = 0
        else:
            segmentation_strategy = 'quickshift'
            if arguments.explain_model == 'slice':
                segmentation_sigma = 0.3
            else:
                segmentation_sigma = 0

        save_dir_path = os.path.join(
            arguments.save_dir,
            f'{arguments.explain_model}_{arguments.pretrained_model}_{arguments.dataset}_{arguments.parts_to_run}_results')
        if arguments.save_dir != 'Final Results':
            if os.path.isdir(save_dir_path):
                shutil.rmtree(save_dir_path)
            os.makedirs(save_dir_path)

        # Get Explainer model class
        module = importlib.import_module('utils.explainer')
        explainer_class = getattr(module, explainer_class_mapping[arguments.explain_model])

        print('#'*100)
        print(f'Loading Model: {arguments.pretrained_model}')
        # Load the pretrained model
        model, target_image_size = load_model_from_name(arguments.pretrained_model)
        if arguments.dataset == 'mscoco':
            model_preprocess = get_preprocess_pipeline_for_model_mscoco_version(target_image_size)
        else : 
            model_preprocess = get_preprocess_pipeline_for_model(target_image_size)
        print(f'Model: {arguments.pretrained_model} loaded successfully')
        print('-' * 10)

        print(f'Downloading/Loading the dataset: {arguments.dataset}')
        dataset, image_filenames = download_dataset(arguments.dataset, model_preprocess)
        print(f'Downloading/Loading the dataset: {arguments.dataset} completed successfully')
        print('-' * 10)

        print(f'Loading random {arguments.num_images_from_dataset} images from the dataset: {arguments.dataset}')
        random_indices = random.sample(range(len(dataset)), arguments.num_images_from_dataset)
        random_indices = list(sorted(random_indices, key=lambda x: x, reverse=False))

        # Create a subset with the random indices
        data_subset = Subset(dataset, random_indices)
        data_subset_image_filenames = [image_filenames[i] for i in range(len(dataset)) if i in random_indices]
        print(f'Loading random {arguments.num_images_from_dataset} images from the dataset: {arguments.dataset} '
              f'completed successfully')
        print('-' * 10)

        for image_index in range(len(data_subset)):
            print('*'*100)
            image_file_path = data_subset_image_filenames[image_index]
            image_name = image_file_path.split(".")[0].split(r"/")[-1].strip()
            image_dict = dict()
            print(f'Running {arguments.explain_model.capitalize()} for Image: {image_name} of '
                  f'Dataset {arguments.dataset}')
            print('-' * 10)
            if not os.path.isdir(os.path.join(save_dir_path, image_name)):
                os.makedirs(os.path.join(save_dir_path, image_name))

            gt_mask = np.array(data_subset[image_index][1])
            with open(os.path.join(save_dir_path, image_name, 'ground_truth_mask.pickle'), 'wb') as file:
                pickle.dump(gt_mask, file)

            # Get segment labels for the original image
            segment_labels = get_segment_labels_for_image_pixels(image_name, image_file_path,
                                                                 data_subset[image_index][0], target_image_size,
                                                                 explainer_class, model, arguments.explain_model,
                                                                 arguments.pretrained_model, arguments.batch_size,
                                                                 arguments.num_workers, segmentation_sigma,
                                                                 arguments.save_first_perturb_image,
                                                                 arguments.parts_to_run, save_dir_path,
                                                                 strategy=segmentation_strategy)
            with open(os.path.join(save_dir_path, image_name, 'segments.pickle'), 'wb') as file:
                pickle.dump(segment_labels, file)

            if arguments.explain_model == 'slice' and arguments.parts_to_run in ['sigma', 'all']:
                # Get best sigma value for Gaussian Blur to create perturbed images for explainer model
                explain_model_object = explainer_class(image_name, data_subset[image_index][0], segment_labels,
                                                       arguments.pretrained_model, model, arguments.batch_size,
                                                       arguments.num_workers, arguments.save_first_perturb_image,
                                                       arguments.parts_to_run, save_dir_path)
                best_sigma_value = explain_model_object.get_best_sigma_value()
                del explain_model_object
            else:
                best_sigma_value = segmentation_sigma

            if arguments.metrics == 'yes':
                for num_run in range(arguments.num_runs):
                    print('.'*100)
                    print(f'Running {arguments.explain_model.capitalize()} iteration: {num_run + 1}')
                    explain_model_object = explainer_class(image_name, data_subset[image_index][0], segment_labels,
                                                           arguments.pretrained_model, model, arguments.batch_size,
                                                           arguments.num_workers, arguments.save_first_perturb_image,
                                                           arguments.parts_to_run, save_dir_path)
                    unstable_features, pos_feature_ranks, neg_feature_ranks, num_samples_used, pos_dict, \
                    neg_dict = explain_model_object.explain_pre_trained_model(best_sigma=best_sigma_value,
                                                                              tolerance_limit=arguments.tolerance,
                                                                              num_perturb=arguments.num_perturb)

                    ranks = {'pos': pos_feature_ranks.astype('int') if len(pos_feature_ranks) > 0 else np.array([]),
                             'neg': neg_feature_ranks.astype('int') if len(neg_feature_ranks) > 0 else np.array([]),
                             'pos_dict': pos_dict, 'neg_dict': neg_dict, 'sel_sigma': best_sigma_value,
                             'h_unstable': unstable_features, 'num_samples': num_samples_used}

                    del explain_model_object
                    image_dict[f'run_{num_run + 1}'] = ranks
                    print(f'{arguments.explain_model.capitalize()} iteration: {num_run + 1} completed successfully')
                final_positive_superpixel_list = image_dict[f'run_{arguments.num_runs}']['pos']
                final_negative_superpixel_list = image_dict[f'run_{arguments.num_runs}']['neg']
                print(f'{arguments.explain_model.capitalize()} for Image: {image_name} of Dataset {arguments.dataset} '
                      f'completed successfully')
            else:
                explain_model_object = explainer_class(image_name, data_subset[image_index][0], segment_labels,
                                                       arguments.pretrained_model, model, arguments.batch_size,
                                                       arguments.num_workers, arguments.save_first_perturb_image,
                                                       arguments.parts_to_run, save_dir_path)
                unstable_features, pos_feature_ranks, neg_feature_ranks, num_samples_used, pos_dict, \
                neg_dict = explain_model_object.explain_pre_trained_model(best_sigma=best_sigma_value,
                                                                          tolerance_limit=arguments.tolerance,
                                                                          num_perturb=arguments.num_perturb)

                ranks = {'pos': pos_feature_ranks.astype('int') if len(pos_feature_ranks) > 0 else np.array([]),
                         'neg': neg_feature_ranks.astype('int') if len(neg_feature_ranks) > 0 else np.array([]),
                         'pos_dict': pos_dict, 'neg_dict': neg_dict, 'sel_sigma': best_sigma_value,
                         'h_unstable': unstable_features, 'num_samples': num_samples_used}

                del explain_model_object
                image_dict[f'run_1'] = ranks
                final_positive_superpixel_list = image_dict[f'run_1']['pos']
                final_negative_superpixel_list = image_dict[f'run_1']['neg']
                print(f'{arguments.explain_model.capitalize()} for Image: {image_name} of Dataset {arguments.dataset} '
                      f'completed successfully')
            _ = save_colored_output_image(image_file_path, image_name, segment_labels,
                                          final_positive_superpixel_list[:5], target_image_size,
                                          save_dir_path, process='most_relevant')
            _ = save_colored_output_image(image_file_path, image_name, segment_labels,
                                          final_negative_superpixel_list[:5], target_image_size,
                                          save_dir_path, process='least_relevant')
            with open(os.path.join(save_dir_path, image_name, 'run_info.pickle'), 'wb') as file:
                pickle.dump(image_dict, file)
        print('#' * 100)
    except Exception as exception:
        raise exception


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--explain_model', default='slice', type=str,
                        help='Explainer Model', choices=['slice', 'lime', 'GridLime'])
    parser.add_argument('--pretrained_model', default='inceptionv3', type=str,
                        help='Pretrained Model to Explain', choices=['inceptionv3', 'resnet50'])
    parser.add_argument('--dataset', default='oxpets', type=str,
                        help='Dataset to Explain', choices=['oxpets', 'pvoc', 'mscoco'])
    parser.add_argument('--metrics', default='no', type=str,
                        help='Flag to indicate metrics calculation', choices=['yes', 'no'])
    parser.add_argument('--num_runs', default=5, type=int,
                        help='Number of runs for the explainer model to calculate metrics in case metrics argugent is '
                             'passed as "yes". Defaults to 1 just for explaining the image.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch Size for passing to Multithreading')
    parser.add_argument('--num_workers', default=100, type=int,
                        help='Number of Workers for Multithreading')
    parser.add_argument('--tolerance', default=3, type=int,
                        help='Tolerance Parameter for SEFE algorithm for feature elimination')
    parser.add_argument('--num_perturb', default=500, type=int, help='Number of Perturbations to train surrogate model')
    parser.add_argument('--num_images_from_dataset', default=50, type=int,
                        help='Number of random images to select from the dataset to run the explaination model')
    parser.add_argument('--save_first_perturb_image', default='yes', type=str,
                        help='Flag to indicate whether to save intermediate perturb images')
    parser.add_argument('--parts_to_run', default='all', type=str,
                        help='Slice Algorithm subsections to run', choices=['all', 'sigma', 'sefe'])
    parser.add_argument('--segmentation_sigma', default=0.3, type=float,
                        help='Sigma to use in case segmentation strategy is grid')
    parser.add_argument('--save_dir', default='Results', type=str,
                        help='Directory to save the results', choices=['Results', 'Final Results'])

    args = parser.parse_args()
    start_time = time.time()
    run_explainer_on_pre_trained_model(args)
    end_time = time.time()
    print(f'Time taken for run: {(end_time - start_time)/60} minutes')
