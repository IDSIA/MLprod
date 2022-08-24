# Models folder

All models are stored in this folder.

Before running the application for the first time, run the `ml_code.ipynb` notebook  to populate this folder with at least the baseline model.

Models are saved in a single folder containing the following files:
* `metadata.json` file for metadata information such as features, sizes, etc.;
* `mms.model` model object for the MinMaxScaler pre-processing;
* `skb.model` model object for the SelectKBest pre-processing;
* `neuralnet.model` PyTorch model of the trained Neural Network.
