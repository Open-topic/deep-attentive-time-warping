# Similarity Assisted Deep Attentive Time Warping

## How to run

### Prepare the dataset
1. Download the UCR Archive from [here](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)

### Training example
1. `python3 code/main-pre-training.py dataset.ID=1 dataset_path=../dataset/`
</br>(For dataset.ID, refer to the UCR_dataset_name.json)
2. `python3 code/main-metric-learning.py dataset.ID=1 dataset_path=../dataset/`

## For reproducibility
Training part

For Modernized baseline, use code/16-main-pre-training.py for pre-training the model and code/16-main-metric-learning.py for metric training sequentially.

For Our proposed method, Similarity Assisted Deep Attentive Time Warping, use code/16-main-pre-training.py for pre-training the model and code/16-similarity-learning-bottle-neck.py for metric training sequentially.

Evaluation part

Evaluation is already done in training script. However, you can run a special script to evaluate discriminability of similarity score only method. <br><br>
`python3 code/16-eval-similarity-learning-bottle-neck.py dataset.ID={your dataset ID} dataset_path=../dataset/`

# Example
For the training part, an example ipynb file containing runs of the experiment is provided in Unet_baseline_non_Bilinear_experiment_example.ipynb
<br><br>
For evaluation of discriminative power of similarity score only, the script examples are in Eval_Copy_of_Yet_another_copy_of_Bilinear_Bottle_Neck_experiment.ipynb