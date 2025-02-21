# SLICE: Stabilized LIME for Consistent Explanations for Image Classification

Welcome to the official repository for our reproducibility research of "**SLICE: Stabilized LIME for Consistent Explanations for Image Classification**", an explainability algorithm to identify positive and negetive superpixels in an image with consistancy. The code uses PyTorch, Multi-threading and Multiprocessing to provide an efficient and fast implementation of the algorithms.

The code in this repository has been used to research the reproducability of the experiments conducted in the original 'SLICE' paper. Furthermore, the repository contains our contributions. Our contributions include:
<ul>
  <li><b>GRID-LIME</b>, a new model that provides better stability than LIME and better explainability than SLICE.</li>
  <li><b>Ground Truth Overlap (GTO)</b> metric to measure how well an explainable model explains the classified output.</li>
</ul>


## Table of Content
- [Installation Guide](#installation-guide)
- [Datasets](#datasets)
- [How to Run](#how-to-run)
  - [Experiments Notebook](#experiments-notebook)
  - [Running Individual Explaination Model](#running-individual-explaination-model)
  - [Running Generate Plots Code](#running-generate-plots-code)


## Installation Guide
<ol type=1>
<li> Download the repository as a .zip or clone the repository using:
<br>git clone git@github.com:aritraban21/FACT-Group7.git
</li>
<br>
<li> Run the remaining steps <b>only if</b> running the code on local machine:
<ol type='a'>
<li> Install the correct version of the used packages from the .yml file using the following command:
<b>conda env create -f environment.yml</b>
</li>
<li> Upon installation of the environment, it can be (de)activated using:
<b>conda activate slice_env</b>
<br>
<b>conda deactivate slice_env</b>
</li>

<li> The environment can be deleted using:
<b>conda remove -n slice_env --all</b>
</li>

<li> Additional packages can be installed using pip:
<b>pip install [package_name]</b>
</li>
</ol>
</li>
</ol>

## Datasets
The Oxford-IIIT pets dataset and Pascal VOC dataset are downloaded dynamically via the code (based on the dataset parameter passed when running the run_explainer.py) and do not need to be separately downloaded.

## How to Run
Now that the environment has been correctly installed, it is time to run the code.

### Experiments Notebook
We have created a single jupyter notebook (Run_Experiments.ipynb) that runs the run_explainer.py for all possible combinations of pretrained model (Inceptionv3/Resnet50), explainer model (Lime/Slice/GridLime) and dataset (Oxpets-IIIT pet/Pascal VOC) to generate the results folder and then plot the CCM, GTO and AOPC plots.

It is <b>recommended</b> to run the Run_explainer.ipynb on Google Colab (preferably on A100 GPU). In order to do so, please follow the following steps:
<ol type=1>
<li> Open the Run_Explainer.ipynb on Google Colab.</li>
<li> Change the runtime time to A100 (preferred) or any available GPU.</li>
<li> Insert a new cell at the very top of the notebook and paste the following code:

```
!pip install rbo
from google.colab import files
import os
files.upload()

!unzip FACT-Group7.zip

os.chdir('FACT-Group7')
os.remove('Run_Experiments.ipynb')
```
</li>

<li> Go to the Runtime tab and select Run All option. The first cell will give you an upload file button. Click on the button and upload the .zip file of the repository downloaded in installation step.</li>
</ol>


If running on local machine (Not recommended as it will take a lot of time to run):
<ol type=1>
<li> Open jupyter notebook with the slice_env installed in installation step and run all the cells of the notebook.</li>
</ol>

For both google colab and local machine, the requested plots will be saved in 'Final Plots' folder (created automatically by the code).

### Running Individual Explaination Model

You can also run the individual explaination models by running the run_explainer.py file with appropriate command line arguments.

The following command line arguments are available for use:

<ol type=1>
  <li> explain_model - This argument is used to specify the explainability model to run. It's default value is 'slice'. It can take one of the following three values: 'lime', 'grid_lime' and 'slice'.</li>
  <br>
  <li> pretrained_model - This argument is used to specify the pretrained model is being explained using the explainability model. It's default value is 'inceptionv3'. It can take one of the following two values: 'inceptionv3' and 'resnet50'.</li>
  <br>
  <li> dataset - This argument is used to specify the dataset to use. It's default value is 'oxpets'. It can take one of the following two values: 'oxpets' and 'pvoc'.</li>
  <br>
  <li> metrics - This argument is used to specify whether we are calculating metrics or not. It's default value is 'no'. It can take one of the following two values: 'yes' and 'no'.</li>
  <br>
  <li> num_runs - This argument is used to specify the number of times each explainer algorithm will be run in case we are calculating metrics. It's default value is 5. It can take any positive integer value.</li>
  <br>
  <li> num_perturb - This argument is used to specify the number of perturbations to be used at various steps of the algorithm. It's default value is 500. It can take any positive integer value.</li>
  <br>
  <li> num_images_from_dataset - This argument is used to specify the number of images to be used from the chosen dataset. It's default value is 50. It can take any positive integer value.</li>
</ol>

### Running Generate Plots Code

If the model results have already been generated, you can also directly run the generate_results.py file with appropriate command line arguments to plot the output graphs.

The following command line arguments are available for use:

<ol type=1>
<li> generate_ccm_plots_flag - This argument is used to specify whether we want to generate the plot for CCM metric. It's default value is 'yes'. It can take one of the following two values: 'yes' and 'no'.</li>
<br>
<li> generate_gto_plots_flag - This argument is used to specify whether we want to generate the plot for GTO metric. It's default value is 'yes'. It can take one of the following two values: 'yes' and 'no'.</li>
<br>
<li> generate_aopc_plots_flag - This argument is used to specify whether we want to generate the plot for AOPC metric. It's default value is 'yes'. It can take one of the following two values: 'yes' and 'no'.</li>
<br>
<li> aopc_sigma_constant - This argument is used to specify whether we are using a constant sigma value of 0 for slice AOPC plots or the best sigma value selected during model run. It's default value is 'no'. It can take one of the following two values: 'yes' and 'no'.</li>
<br>
<li> num_iter - This argument is used to specify the number of times the explainer algorithm was run to generate results for calculating the metrics. It's default value is 10. It can take any positive integer value.</li>
</ol>

The requested plots will be saved in 'Final Plots' folder (created automatically by the code).
