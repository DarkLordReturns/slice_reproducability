import torch
import numpy as np
import copy
import random
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
from torch.utils.data import DataLoader

from utils.helper import PerturbedDataset, generate_perturbation_encodings, perturb_image, calculate_entropy, \
    get_original_image_prediction, save_image


class LimeExplainer:
    def __init__(self, image_name, image, segment_labels, model_name, model, batch_size, number_workers,
                 save_first_perturb_image, parts_to_run, main_save_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_filename = image_name
        self.image = image.unsqueeze(0)
        self.model_name = model_name
        self.model = model
        self.batch_size = batch_size
        self.number_workers = number_workers
        self.superpixels = segment_labels
        self.top_predicted_class, self.probability_of_prediction = get_original_image_prediction(self.image,
                                                                                                 self.model,
                                                                                                 device=self.device)
        self.train_mat = np.append(np.ones(np.unique(self.superpixels).shape[0]), self.probability_of_prediction)
        self.train_mat = self.train_mat.reshape((1, len(self.train_mat)))
        self.train_mat_sel_idx = np.zeros(np.unique(self.superpixels).shape[0])
        self.save_first_perturb_image = True if save_first_perturb_image == 'yes' else False
        self.parts_to_run = parts_to_run
        self.main_save_dir = main_save_dir
        self.sigma = None

    def generate_perturbation_encodings_sample(self, test_sample_size):
        perturbations = generate_perturbation_encodings(test_sample_size, np.unique(self.superpixels).shape[0],
                                                        existing_perturbations=self.train_mat.shape[0])
        return perturbations

    def generate_perturbed_images_from_encodings(self, perturbation_encodings):
        perturbed_images = list()
        for index in range(len(perturbation_encodings)):
            perturbed_images.append(perturb_image(perturbation_encodings[index], copy.deepcopy(self.image),
                                                  self.superpixels))
        return perturbed_images

    def classify_image_batch(self, input_batch):
        with torch.no_grad():
            batch_prediction = self.model(input_batch.to(self.device))

        batch_prediction = torch.nn.functional.softmax(batch_prediction, dim=1).cpu().numpy()
        return batch_prediction[:, self.top_predicted_class]

    @staticmethod
    def populate_and_get_train_matrix_copy(train_matrix, new_data):
        return np.vstack((train_matrix, new_data))

    def explain_pre_trained_model(self, best_sigma, num_perturb=500, tolerance_limit=3):
        print('='*100)
        print('Starting Explain Function')
        print('-' * 10)
        self.sigma = best_sigma
        # Get Perturbation Encodings
        print('Generating Perturbation Encodings')
        perturbation_encodings = self.generate_perturbation_encodings_sample(num_perturb)
        print('Perturbation Encodings Created Successfully')
        print('-' * 10)

        print(f'Creating Perturbed Images from encodings')
        # Get Perturbed Images for encodings using multithreading
        perturbed_images = self.generate_perturbed_images_from_encodings(perturbation_encodings)
        print('Perturbation Images created successfully')
        print('-' * 10)

        print('Creating Perturbation Image DataLoader')
        # Create Perturbed Images DataLoader
        perturbed_dataset = PerturbedDataset(perturbed_images)
        perturbed_dataloader = DataLoader(perturbed_dataset, batch_size=self.batch_size, shuffle=False,
                                          num_workers=8)
        print(f'Perturbation Image DataLoader created successfully')
        print('-' * 10)

        print('Getting Pre Trained model Predictions for created perturbed image batches')
        # Get Top Predicted Class Probabilities for Perturbed Images
        batch_predictions = Parallel(n_jobs=-1)(delayed(self.classify_image_batch)(perturbed_image_batch) for
                                                perturbed_image_batch in perturbed_dataloader)
        combined_predictions = np.concatenate(batch_predictions, axis=0)
        new_data = np.hstack((perturbation_encodings, combined_predictions))

        # Populate the train matrix
        train_matrix = self.populate_and_get_train_matrix_copy(copy.deepcopy(self.train_mat), new_data)
        print('Pre Trained model Predictions for created perturbed image batches fetched successfully')
        print('-' * 10)

        x_train = train_matrix[:, :-1]
        y_train = train_matrix[:, -1]
        original_image = np.ones(x_train.shape[1])[np.newaxis, :]
        distances = pairwise_distances(x_train, original_image, metric='cosine').ravel()
        kernel_width = 0.25
        weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))  # Kernel function
        ridge_model = Ridge(alpha=1)
        ridge_model.fit(x_train, y_train, sample_weight=weights)
        coefficients = ridge_model.coef_

        sorted_indices = np.argsort(np.abs(coefficients))[::-1]
        sorted_coefficients = coefficients[sorted_indices]

        # Partition coefficients into positive and negative
        positive_coefficients = sorted_coefficients[sorted_coefficients >= 0]
        negative_coefficients = sorted_coefficients[sorted_coefficients < 0]

        pos_indices = sorted_indices[sorted_coefficients >= 0]
        neg_indices = sorted_indices[sorted_coefficients < 0]
        # Create dictionaries
        pos_dict = {
            'column_names': pos_indices,
            'column_means': positive_coefficients
        }

        neg_dict = {
            'column_names': neg_indices,
            'column_means': negative_coefficients
        }

        return 'NA', pos_indices, neg_indices, num_perturb, pos_dict, neg_dict


class SliceExplainer:
    def __init__(self, image_name, image, segment_labels, model_name, model, batch_size, number_workers,
                 save_first_perturb_image, parts_to_run, main_save_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_filename = image_name
        self.image = image.unsqueeze(0)
        self.model_name = model_name
        self.model = model
        self.batch_size = batch_size
        self.number_workers = number_workers
        self.superpixels = segment_labels
        self.top_predicted_class, self.probability_of_prediction = get_original_image_prediction(self.image,
                                                                                                 self.model,
                                                                                                 device=self.device)
        self.train_mat = np.append(np.ones(np.unique(self.superpixels).shape[0]), self.probability_of_prediction)
        self.train_mat = self.train_mat.reshape((1, len(self.train_mat)))
        self.train_mat_sel_idx = np.zeros(np.unique(self.superpixels).shape[0])
        self.save_first_perturb_image = True if save_first_perturb_image == 'yes' else False
        self.parts_to_run = parts_to_run
        self.main_save_dir = main_save_dir
        self.sigma = None

    def generate_perturbation_encodings_sample(self, test_sample_size):
        perturbations = generate_perturbation_encodings(test_sample_size, np.unique(self.superpixels).shape[0],
                                                        existing_perturbations=self.train_mat.shape[0])
        return perturbations

    def generate_perturbed_images_from_encodings(self, perturbation_encodings, sigma_value, threaded=False,
                                                 index_value=None):
        perturbed_images = list()
        for index in range(len(perturbation_encodings)):
            perturbed_images.append(perturb_image(perturbation_encodings[index], copy.deepcopy(self.image),
                                                  self.superpixels, sigma_value))
        if threaded:
            return index_value, perturbed_images
        else:
            return perturbed_images

    def classify_image_batch(self, input_batch):
        with torch.no_grad():
            batch_prediction = self.model(input_batch.to(self.device))

        batch_prediction = torch.nn.functional.softmax(batch_prediction, dim=1).cpu().numpy()
        return batch_prediction[:, self.top_predicted_class]

    @staticmethod
    def populate_and_get_train_matrix_copy(train_matrix, new_data):
        return np.vstack((train_matrix, new_data))

    @staticmethod
    def get_adjusted_r2(train_matrix):
        simpler_model = Ridge(alpha=1, fit_intercept=True)
        simpler_model.fit(X=train_matrix[:, :-1], y=train_matrix[:, -1])
        coeffs = simpler_model.coef_

        y_pred = simpler_model.predict(train_matrix[:, :-1])
        y_mean = np.mean(train_matrix[:, -1])
        ss_residual = np.sum((train_matrix[:, -1] - y_pred) ** 2)
        ss_total = np.sum((train_matrix[:, -1] - y_mean) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        n = train_matrix.shape[0]
        p = train_matrix.shape[1] - 1
        adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
        return coeffs, adj_r_squared

    def get_best_sigma_value(self, test_sample_size=500):
        print('|'*100)
        print('Get Best Sigma Function Started')
        print('-' * 10)
        # Get Perturbation Encodings
        print('Getting Perturbation Encodings')
        perturbation_encodings = self.generate_perturbation_encodings_sample(test_sample_size)
        print('Perturbation Encodings Created Successfully')
        print('-' * 10)

        sigma_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        print(f'Creating Perturbed Images from encodings for sigmas: {sigma_values}')
        perturbed_images_for_sigmas = list()
        with ThreadPoolExecutor(max_workers=min(len(sigma_values), self.number_workers)) as executor:
            try:
                futures = []
                for index in range(len(sigma_values)):
                    futures.append(executor.submit(self.generate_perturbed_images_from_encodings,
                                                   perturbation_encodings=perturbation_encodings,
                                                   sigma_value=sigma_values[index], threaded=True, index_value=index))

                for future in as_completed(futures):
                    perturbed_images_for_sigmas.append(future.result())
            except Exception as exception:
                raise exception

        perturbed_images_for_sigmas = [x[1] for x in list(sorted(perturbed_images_for_sigmas, key=lambda y: y[0],
                                                                 reverse=False))]
        print('Perturbed Images created successfully from encodings')
        print('-' * 10)

        scores = []
        for index in range(len(sigma_values)):
            print('+'*100)
            print(f'Starting calculation for sigma: {sigma_values[index]}')
            print('-'*10)
            if self.save_first_perturb_image:
                print(f'Saving first perturbed image for: {sigma_values[index]}')
                _ = save_image(self.image_filename, 'sigma_selection', perturbed_images_for_sigmas[index][0],
                               self.main_save_dir, sigma_value=sigma_values[index])
                print(f'First Perturbed image for: {sigma_values[index]} saved successfully')
                print('-' * 10)

            print(f'Creating Perturbation Image DataLoader for: {sigma_values[index]}')
            # Create Perturbed Images DataLoader
            perturbed_dataset = PerturbedDataset(perturbed_images_for_sigmas[index])
            perturbed_dataloader = DataLoader(perturbed_dataset, batch_size=self.batch_size, shuffle=False,
                                              num_workers=8)
            print(f'Perturbation Image DataLoader for: {sigma_values[index]} created successfully')
            print('-' * 10)

            print(f'Getting Pre Trained model Predictions for: {sigma_values[index]}')
            # Get Top Predicted Class Probabilities for Perturbed Images
            batch_predictions = Parallel(n_jobs=-1)(delayed(self.classify_image_batch)(perturbed_image_batch) for
                                                    perturbed_image_batch in perturbed_dataloader)
            combined_predictions = np.concatenate(batch_predictions, axis=0)
            new_data = np.hstack((perturbation_encodings, combined_predictions))

            # Populate the train matrix
            train_matrix = self.populate_and_get_train_matrix_copy(copy.deepcopy(self.train_mat), new_data)
            print(f'Pre Trained model Predictions for: {sigma_values[index]} fetched successfully')
            print('-' * 10)

            print(f'Calculating Adjusted R2 for: {sigma_values[index]}')
            _, adj_r2 = self.get_adjusted_r2(train_matrix)
            scores.append(adj_r2)
            print(f'Adjusted R2 for: {sigma_values[index]} calculated successfully')

        score_array = np.array(scores)
        non_nan_indices = np.where(~np.isnan(score_array))[0]
        sigma_index = non_nan_indices[np.argsort(np.array(score_array)[non_nan_indices])][-1]
        print(f'Best Sigma Value found: {sigma_values[sigma_index]}')
        print('|' * 100)
        return sigma_values[sigma_index]

    @staticmethod
    def train_new_ridge_model_for_sefe(x_train, y_train, weights):
        random_bootstrap_sampling_indices = random.choices(range(len(x_train)), weights=weights, k=len(x_train))
        x_sample, y_sample = x_train[random_bootstrap_sampling_indices], y_train[random_bootstrap_sampling_indices]
        weights_bootstrap_sampling = weights[random_bootstrap_sampling_indices]
        ridge_model = Ridge(alpha=1)
        ridge_model.fit(x_sample, y_sample, sample_weight=weights_bootstrap_sampling)
        return ridge_model.coef_

    def run_sefe(self, max_steps, n_models, num_perturb, tolerance_limit):
        print('|' * 100)
        print('SEFE started')
        print('-' * 10)
        current_tolerance = 0
        final_coeffs_matrix = np.zeros((n_models, len(self.train_mat_sel_idx)))
        final_train_matrix = None

        for step in range(max_steps):
            print('+'*100)
            print(f'SEFE iteration: {step + 1}')
            print('-'*10)
            # Get Perturbation Encodings
            print('Generating Perturbation Encodings')
            perturbation_encodings = self.generate_perturbation_encodings_sample(num_perturb)
            if not np.all(self.train_mat_sel_idx == 1):
                perturbation_encodings[:, self.train_mat_sel_idx.astype(bool)] = 1
            else:
                print('Perturbation Encodings Created Successfully but all superpixels are non-stable')
                print('-' * 10)
                break
            print('Perturbation Encodings Created Successfully and non-stable superpixel indices set to 1')
            print('-' * 10)

            print(f'Creating Perturbed Images from encodings for best sigma: {self.sigma}')
            # Get Perturbed Images for encodings using multithreading
            perturbed_images = self.generate_perturbed_images_from_encodings(perturbation_encodings, self.sigma)
            print(f'Perturbation Images for best sigma: {self.sigma} created successfully')
            print('-' * 10)

            print(f'Creating Perturbation Image DataLoader for: {self.sigma}')
            # Create Perturbed Images DataLoader
            perturbed_dataset = PerturbedDataset(perturbed_images)
            perturbed_dataloader = DataLoader(perturbed_dataset, batch_size=self.batch_size, shuffle=False,
                                              num_workers=8)
            print(f'Perturbation Image DataLoader for: {self.sigma} created successfully')
            print('-' * 10)

            print('Getting Pre Trained model Predictions for created perturbed image batches')
            # Get Top Predicted Class Probabilities for Perturbed Images
            batch_predictions = Parallel(n_jobs=-1)(delayed(self.classify_image_batch)(perturbed_image_batch) for
                                                    perturbed_image_batch in perturbed_dataloader)
            combined_predictions = np.concatenate(batch_predictions, axis=0)
            new_data = np.hstack((perturbation_encodings, combined_predictions))

            # Populate the train matrix
            train_matrix = self.populate_and_get_train_matrix_copy(copy.deepcopy(self.train_mat), new_data)
            final_train_matrix = copy.deepcopy(train_matrix)
            print('Pre Trained model Predictions for created perturbed image batches fetched successfully')
            print('-' * 10)

            x_train = train_matrix[:, :-1]
            y_train = train_matrix[:, -1]
            x_train = x_train[:, np.logical_not(self.train_mat_sel_idx.astype(bool))]
            original_image = np.ones(x_train.shape[1])[np.newaxis, :]
            distances = pairwise_distances(x_train, original_image, metric='cosine').ravel()
            kernel_width = 0.25
            weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

            print('Training Ridge models')
            coeffs_list = Parallel(n_jobs=-1)(delayed(self.train_new_ridge_model_for_sefe)(copy.deepcopy(x_train),
                                                                                           copy.deepcopy(y_train),
                                                                                           weights) for _ in
                                              range(n_models))
            coeffs_list = np.array(coeffs_list)
            print('Ridge models trained successfully')
            print('-' * 10)

            print('Calculating entropy for superpixel coefficients of models')
            sign_entropies = []
            for column in range(coeffs_list.shape[1]):
                data = coeffs_list[:, column]
                sign_entropy = calculate_entropy(data)
                sign_entropies.append(sign_entropy)

            sign_entropies = np.array(sign_entropies)
            print('Entropy calculated successfully')
            print('-' * 10)
            non_spurious_indices = np.where(sign_entropies != 0)[0]

            zero_indices_before_iteration = np.where(self.train_mat_sel_idx == 0)[0]
            mapped_non_spurious_indices = zero_indices_before_iteration[non_spurious_indices]

            if not np.size(mapped_non_spurious_indices) == 0:
                self.train_mat_sel_idx[mapped_non_spurious_indices] = 1
                current_tolerance = 0
            else:
                if current_tolerance == 0:
                    final_coeffs_matrix = coeffs_list
                    current_tolerance = current_tolerance + 1
                else:
                    if current_tolerance < tolerance_limit:
                        current_tolerance = current_tolerance + 1
                    else:
                        break
            print(f'SEFE iteration: {step + 1} completed successfully')
        print('SEFE completed successfully')
        print('|'*100)
        return final_train_matrix

    def explain_pre_trained_model(self, best_sigma, n_models=1000, max_steps=10, num_perturb=500, tolerance_limit=3):
        print('='*100)
        print('Starting Explain Function')
        print('-' * 10)
        self.sigma = best_sigma
        if self.parts_to_run in ['sefe', 'all']:
            final_train_matrix = self.run_sefe(max_steps, n_models, num_perturb, tolerance_limit)
        else:
            print('Generating Perturbation Encodings')
            perturbation_encodings = self.generate_perturbation_encodings_sample(num_perturb)
            print('Perturbation Encodings Created Successfully and non-stable superpixel indices set to 1')
            print('-' * 10)
            print(f'Creating Perturbed Images from encodings for best sigma: {self.sigma}')
            # Get Perturbed Images for encodings using multithreading
            perturbed_images = self.generate_perturbed_images_from_encodings(perturbation_encodings, self.sigma)
            print(f'Perturbation Images for best sigma: {self.sigma} created successfully')
            print('-' * 10)

            print(f'Creating Perturbation Image DataLoader for: {self.sigma}')
            # Create Perturbed Images DataLoader
            perturbed_dataset = PerturbedDataset(perturbed_images)
            perturbed_dataloader = DataLoader(perturbed_dataset, batch_size=self.batch_size, shuffle=False,
                                              num_workers=8)
            print(f'Perturbation Image DataLoader for: {self.sigma} created successfully')
            print('-' * 10)

            print('Getting Pre Trained model Predictions for created perturbed image batches')
            # Get Top Predicted Class Probabilities for Perturbed Images
            batch_predictions = Parallel(n_jobs=-1)(delayed(self.classify_image_batch)(perturbed_image_batch) for
                                                    perturbed_image_batch in perturbed_dataloader)
            combined_predictions = np.concatenate(batch_predictions, axis=0)
            new_data = np.hstack((perturbation_encodings, combined_predictions))

            # Populate the train matrix
            train_matrix = self.populate_and_get_train_matrix_copy(copy.deepcopy(self.train_mat), new_data)
            final_train_matrix = copy.deepcopy(train_matrix)
            print('Pre Trained model Predictions for created perturbed image batches fetched successfully')
            print('-' * 10)
        self.train_mat = final_train_matrix

        print('Creating training data by removing non stable superpixels to train final Ridge model')
        print('-' * 10)
        x_train = self.train_mat[:, 0:(self.train_mat.shape[1] - 1)]
        retained_indices = np.where(~self.train_mat_sel_idx.astype(bool))[0]
        if len(retained_indices) > 0 and \
                np.var(self.train_mat[:, (self.train_mat.shape[1] - 1):self.train_mat.shape[1]]) != 0:
            x_train = x_train[:, retained_indices]
            y_train = self.train_mat[:, (self.train_mat.shape[1] - 1):self.train_mat.shape[1]]
            original_image = np.ones(x_train.shape[1])[np.newaxis, :]
            distances = pairwise_distances(x_train, original_image, metric='cosine').ravel()
            kernel_width = 0.25
            weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))
            ridge_model = Ridge(alpha=1)
            print('Started training final Ridge model')
            ridge_model.fit(x_train, y_train, sample_weight=weights)
            print('Final Ridge model trained successfully')
            print('-' * 10)
            mapped_coefficients = np.zeros(self.train_mat.shape[1] - 1)
            mapped_coefficients[retained_indices] = ridge_model.coef_

            # Get the original indices for positive and negative coefficients
            positive_coef_indices = np.where(mapped_coefficients > 0)[0]
            negative_coef_indices = np.where(mapped_coefficients < 0)[0]

            # Get the coefficients using the indices
            positive_coefs = mapped_coefficients[positive_coef_indices]
            negative_coefs = mapped_coefficients[negative_coef_indices]

            # Get the indices that would sort the coefficients
            positive_coefs_sorted_indices = positive_coef_indices[np.argsort(positive_coefs)[::-1]]
            negative_coefs_sorted_indices = negative_coef_indices[np.argsort(negative_coefs)]

            positive_coefs_sorted = positive_coefs[np.argsort(positive_coefs)[::-1]]
            negative_coefs_sorted = negative_coefs[np.argsort(negative_coefs)]

            pos_dict = {
                'column_names': positive_coefs_sorted_indices,
                'column_means': positive_coefs_sorted
            }

            neg_dict = {
                'column_names': negative_coefs_sorted_indices,
                'column_means': negative_coefs_sorted
            }

            pos_feature_ranks = positive_coefs_sorted_indices
            neg_feature_ranks = negative_coefs_sorted_indices

            print('Explain Function completed successfully')
            print('=' * 100)
            return np.nonzero(self.train_mat_sel_idx)[0], pos_feature_ranks, neg_feature_ranks, \
                   (max_steps - 1) * num_perturb, pos_dict, neg_dict
        else:
            # LIME as the labels for all perturbations are same after removing non-stable superpixels.
            save_first_perturb_image = 'yes' if self.save_first_perturb_image else 'no'
            lime_explainer = LimeExplainer(self.image_filename, copy.deepcopy(self.image).squeeze(0), self.superpixels,
                                           self.model_name, self.model, self.batch_size, self.number_workers,
                                           save_first_perturb_image, self.parts_to_run, self.main_save_dir)

            unstable_features, pos_feature_ranks, neg_feature_ranks, num_samples_used, pos_dict, neg_dict = \
                lime_explainer.explain_pre_trained_model(0)

            print('Explain Function completed successfully')
            print('='*100)
            return unstable_features, pos_feature_ranks, neg_feature_ranks, num_samples_used, pos_dict, neg_dict
