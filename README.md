# TrajectoryNet
Source code of ECML PKDD 2017 submission: TrajectoryNet

**Title**: TrajectoryNet: An Embedded GPS Trajectory Representation for Point-based Classification Using Recurrent Neural Networks

**Authors**: Xiang Jiang, Erico N de Souza, Ahmad Pesaranghader, Baifan Hu, Daniel L. Silver and Stan Matwin


# Dependencies
The follwoing softwares are required for this source code.
- Python 2.7
- Tensorflow 1.0.1
- sklean 0.18.1
- numpy 1.12.0
- R (optional for preprocessing)

# Data
The data can be downloaded from this [link](https://drive.google.com/open?id=0B_8r6OqflofXXzZheTVEc2h6Nms).
This dataset contains raw data as well as the preprocessed data in .npy format.

# Network Configurations
The file config.json includes the configurations of the model:
- training, validation and test set selection
- number of hidden nodes in each layer
- learning rate
- mini-batch size
- number of layers
- number of epochs
- whether to checkpoint the model after training
- whether to restore the model before training
- the size of truncated sequences
- frequency to evaluate the model while training
- number of threads
- whether to use GPU during training
- etc.

# Network Training
python trajectoryNet.py

# Preprocessing
In case you are interested in the preprocessing and discretization of the data, please refer to file `preprocess.R`.

After the preprocess data are stored in a .csv file, it is required to run `create_npy.py` to transform the data into .npy format to get ready to import to Tensorflow.
