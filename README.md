# DeepGGL
A deep convolutional neural network that integrates residual connections and attention mechanisms with geometric graph representations of a protein-ligand complex to predict binding affinity.

## Package Requirement
Install the following necessary packages in a new conda environment.
```bash
python=3.9.19
tensorflow=2.4.1
sklearn=1.5.1
joblib=1.4.2
numpy=1.21.6
scipy=1.11.4
pandas=2.0.3
biopandas=0.5.1
```
# Example
## 1. Feature generation
Use the code in the `feature_generation` directory to generate geometric subgraph features. For example, to generate features for the CASF 2016 Core Set with kernel index 12 from `utils/kernels.csv` and a cutoff of 10 Å, run: 
```bash
python3 get_ggl_features.py -k 12 -c 10.0 \
    -f '../labled_data/CASF_2016_CoreSet.csv'\
    -dd ${data_directory}\
    -fd ${feature_directory}
```
This will create the features from structures in the `data_directory` directory, and save them in the `feature_directory` directory.


## 2. Protein-Ligand Binding Affinity Prediction
To predict the protein-ligand binding affinity from the generated features, first, download the trained model from the link [⬇️ Download Trained Model](https://kennesawedu-my.sharepoint.com/:u:/g/personal/mrana10_kennesaw_edu/EdFJKY5ZQX5FhKLtpSFo3voBnONt3y2668D6S4J7CkhDuQ?e=ntzyW8). Second, run the `predict.py` script from the `prediction` directory. 
```bash
python3 predict.py \
    -tf ${test_feature_file}\
    -model ${trained_model}\
    -scaler ./scaler\
    -out ${output_directory}
```
## 3. Retrain DeepGGL
To retrain the DeepGGL, first generate features for the training set `labeled_data/training_set.csv` and the validation set `labeled_data/validation_set.csv` for cutoff values of 5 Å, 10 Å, and 15 Å using the command from step 1, and then combine the features. Second, use the script `training/train.py` to retrain the model.
```bash
python3 train.py \
        -tf ${training_feature_file} \
        -vf ${validation_feature_file} \
        -m ${model_outname}
```
