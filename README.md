<div align="center">
    <img alt="Institut Polytechnique de Paris Logo" width="auto" height="100px" src="https://www.ip-paris.fr/sites/default/files/presse/Charte%20Graphique/Logo%20IP%20Paris%206%20%C3%A9coles%20vertical%20png.png" />
</div>


# Machine and Deep Learning Data Challenge: Sub-event Detection in Twitter Streams

__Team Sigmoid Sigmas:__


  - Tim Luka Horstmann (tim.horstmann@ip-paris.fr)
  - Mathieu Antonopoulos (antonopoulos@ip-paris.fr)
  - Baptiste Geisenberger (geisenberger@ip-paris.fr)


[Link to the Data Challenge](https://www.kaggle.com/competitions/sub-event-detection-in-twitter-streams)

## Repository Structure
- `README.md`: This file.
- `code`: Directory containing the code developed as part of this data challenge.
	- `traditional_ml_approach`: Directory containing our traditional ML approach.
	- `graph_approach`: Directory containing our graph approach.
	- `llm_approach`: Directory containing our llm_approach.
- `predictions`: Directory containing predictions made by the models. (we provide some exemplary, more detailed predictions from the best performing submission on Kaggle)
- `challenge_data`: Directory containing the original data for this data challenge. __Data to be placed here!__

## Content of the different directories and instructions on how to run the code for all our approaches:
> Traditional ML Approach (Section 2.1 in the report):

Folder: `traditional_ml_approach`

- `requirements_traditional_ml.txt`: This file lists all the pip dependencies required to set up the environment for running the code. To create the environment, use: ⁠ `pip install -r requirements_traditional_ml.txt`⁠
- `traditional_ml.ipynb`⁠: This Jupyter notebook contains the implementation of the traditional machine learning approach. Detailed instructions for running the code are included within the notebook.

- `feature_matrices`⁠: This directory is used to load and save the design matrices (feature matrices) generated and used during the execution of the code.

> Graph Approach (Section 2.2 in the report):

Folder: `graph_approach`

- `requirements_graph.txt`: This file lists all the pip dependencies required to set up the environment for running the code. To create the environment, use: ⁠ `pip install -r requirements_traditional_ml.txt`⁠
- `Graph_approach.ipynb`⁠: This Jupyter notebook contains the implementation of the Graph representation approach. Detailed instructions for running the code are included within the notebook.

- `stop_words.txt`⁠: This file contains stop words that can be used for preprocessing.


> LLM Approach (Section 2.3 in the report):

Folder: `llm_approach`

- `llm_environment.yml`: Environment/requirements needed to run the code of the llm_approach.
- `preprocessor.py`: Helper script containing the `filter_tweet` method, used as preprocessing for the llm approach.
- `EnhancedPeriodClassifier`: Directory containg the code to our EnhancedPeriodClassifier approach (Section 2.3.2 in the report; approach achieving the highest accuracy).
	- `enhancedperiodclassifier.py` & `enhancedperiodclassifier_modified.py`: scripts defining the model architecture of the EnhancedPeriodClassifier model and its modified version. The model to use is chosen by choosing the file to import in other scripts.
	- `train.py`: main training script - having properly populated all relevant variables at the beginning of the script and below `if __name__ == "__main__":` (variables define important paths, hyperparameters etc.), this script can be executed by running `train.py`.
	- `train_ablation.py`; code very similar to `train.py`. Executed as part of the ablation study for the EnhancedPeriodClassifier model.
	- `inference_wandb_data.py`: inference script used to create predictions using a EnhancedPeriodClassifier model, which is downloaded from wandb. The script can be executed by running `inference_wandb_data.py -w <wandb_run_id>`.
	- `inference_local_data.py`: notebook provided to run inference with local model data, not needing wandb access. Provided for convenience and not used during the development phase. Requires reference to a folder containing the `config.json`, `model.safetensors`, and `normalization_params.json` files obtained during training a model. We provide the two folders below with exemplary model data that can be used to run inference.
	- `BestModel_EnhancedPeriodClassifier` (requires sentiment_score to be included during inference, uses the model defined in `enhancedperiodclassifier.py`) and `BestModel_EnhancedPeriodClassifier_Modified` (does not require sentiment_score to be included during inference, uses the model defined in `enhancedperiodclassifier_modified.py`) folders: folders containing the model data to our best performing models (according to accuracy on test set on Kaggle): the "normal" EnhancedPeriodClassifier and the "modified" EnhancedPeriodClassifier, achieving 0.74609, and 0.76562 accuracy on the test set, respectively.

- `PeriodClassifier`: Directory containing the code to our PeriodClassifier approach (Section 2.3.1 in the report)
	- `datasets.py`: Definition of different dataset classes as these differ for the concatenation and aggregation approach.
	- `periodclassifier.py`: Definition of our model architecture built around a pre-trained BERT-based model. Needed due to aggregation. Allows fine-tuning of the pre-trained model.
	- `train.py`: Script used to train the PeriodClassifier model. Can be run by executing `train.py` (requires prior definition of all relevant variables)
	- `inference.py`: Script used to make predictions using the PeriodClassifier model. Downloads the model data from wandb or uses locally available data. Can be run by executing `inference.py` (requires prior definition of all relevant variables)
- `tweet_level_models` (first initial experiments, code kept for completeness): Directory containing code for models making predictions on a tweet_level
	- `llm_finetuning.py`: code to train the model
	- `inference.py`: code to make predictions 

__Note:__
All relevant variables and paths that are generally necessary for running a script are defined in the python scripts - either at the beginning of the script or clearly highlighted by being capitalized variables (e.g. `DATA_DIR` always defines the path to the "challenge_data" folder). For training, most of the models automatically establish a connection to [Weights & Biases](https://wandb.ai/home). As such, the `wandb.init()` should be adapted before starting training. Here, another wandb project and entity could be provided to enable integrated experiment tracking. Alternatively, the wandb references in these scripts must be removed to train without tracking the experiments (discouraged).

## INPUT
The "challenge_data" directory should be placed in the working directory (see above), and formatted as specified in the data challenge description.