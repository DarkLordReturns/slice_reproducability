import os
import argparse
import time
import warnings

from utils.helper import download_dataset
from utils.results_helper import generate_ccm_plots, generate_gto_plot, plot_aopc_results, plot_auc_results, \
    get_negative_probabilities_table, perform_wilcoxon_test

warnings.filterwarnings("ignore")


def generate_plots(arguments):
    try:
        if not os.path.isdir(f'Final Plots'):
            os.makedirs(f'Final Plots')

        if not os.path.isdir('images_oxpets'):
            print('Downloading oxpets dataset')
            _, _ = download_dataset('oxpets')
            print('Oxpets dataset download successfull')

        if not os.path.isdir('images_pvoc'):
            print('Downloading pvoc dataset')
            _, _ = download_dataset('pvoc')
            print('PVOC dataset download successfull')

        if not os.path.isdir('coco_dataset'):
            print('Downloading coco dataset')
            _, _ = download_dataset('coco')
            print('MSCoco dataset download successfull')

        if arguments.generate_ccm_plots_flag == 'yes':
            generate_ccm_plots(arguments.save_dir, show_plot=arguments.show_plots_in_output_flag,
                               num_iter=arguments.num_iter)

        base_folder_list = ['lime_resnet50_oxpets_results/',
                            'GridLime_resnet50_oxpets_results/',
                            'slice_resnet50_oxpets_results/',
                            'lime_inceptionv3_oxpets_results/',
                            'GridLime_inceptionv3_oxpets_results/',
                            'slice_inceptionv3_oxpets_results/']

        folder_list = [os.path.join(arguments.save_dir, x) for x in base_folder_list]

        if arguments.generate_gto_plots_flag == 'yes':
            generate_gto_plot(folder_list, show_plot=arguments.show_plots_in_output_flag, num_iter=arguments.num_iter)
            
        results_folder = 'AOPC Results'
        aopc_image_name = 'Final Plots/aopc_ins_del_combined.png'
        
        if arguments.generate_aopc_plots_flag == 'yes':
            plot_aopc_results(base_folder=arguments.save_dir, results_folder=results_folder,
                              output_file=aopc_image_name, show_plot=arguments.show_plots_in_output_flag,
                              sigma_constant=arguments.aopc_sigma_constant)

        results_folder = 'AUC Results'
        auc_image_name = 'Final Plots/auc_ins_del_combined.png'
        if arguments.generate_auc_plots_flag == 'yes':
            plot_auc_results(base_folder=arguments.save_dir, results_folder=results_folder, output_file=auc_image_name,
                             show_plot=False)

        results_folder = 'Tables'
        if arguments.generate_tables_flag == 'yes':
            get_negative_probabilities_table(base_folder=arguments.save_dir, results_folder=results_folder)
            perform_wilcoxon_test(base_folder=arguments.save_dir, results_folder=results_folder, method='aopc')
            perform_wilcoxon_test(base_folder=arguments.save_dir, results_folder=results_folder, method='auc')
    except Exception as exception:
        raise exception


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_ccm_plots_flag', default='yes', type=str,
                        help='Set to yes to generate CCM plots', choices=['yes', 'no'])
    parser.add_argument('--generate_gto_plots_flag', default='yes', type=str,
                        help='Set to yes to generate GTO plots', choices=['yes', 'no'])
    parser.add_argument('--generate_aopc_plots_flag', default='yes', type=str,
                        help='Set to yes to generate AOPC plots', choices=['yes', 'no'])
    parser.add_argument('--generate_auc_plots_flag', default='yes', type=str,
                        help='Set to yes to generate AUC plots', choices=['yes', 'no'])
    parser.add_argument('--generate_tables_flag', default='yes', type=str,
                        help='Set to yes to generate Tables', choices=['yes', 'no'])
    parser.add_argument('--show_plots_in_output_flag', default='yes', type=str,
                        help='Set to yes to print plots in output', choices=['yes', 'no'])
    parser.add_argument('--aopc_sigma_constant', default='no', type=str,
                        help='Flag to make sigma value constant i.e. 0 while generating aopc plots',
                        choices=['yes', 'no'])
    parser.add_argument('--save_dir', default='Results', type=str,
                        help='Directory to save the results', choices=['Results', 'Final Results'])
    parser.add_argument('--num_iter', default=10, type=int,
                        help='Number of Iterations for which the models were run')

    args = parser.parse_args()
    start_time = time.time()
    generate_plots(args)
    end_time = time.time()
    print('The requested plots are saved in Final Plots folder')
    print(f'Time taken for generating results: {(end_time - start_time)} seconds')
