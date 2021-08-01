# MODULATION SPECTRAL SIGNAL REPRESENTATIONAND I-VECTORS FOR ANOMALOUS SOUND DETECTION

This repository can be used to reproduce my submission for my bachelor project " Modulation Spectral Features for Anomalous Sounds Detection "


#### Abstract
Anomalous Sound Detection is essential as noise generation in machines involves many physical principles that can diagnose the machine by measuring and analyzing the noise generated. The challenge encountered here is detecting unknown abnormal sounds while only having normal sound samples as training data. There are different ways to perform this task, the one we are concerned with is combining Modulation Spectral Signal Representation and Gaussian Mixture Model (GMM) systems. The first system mainly detects outliers using a method similar to nearest neighbor search using simple machine learning algorithms. The second system predicts object locations at each frame in a sequence using the number of  mixture components and their means, Mel frequency cepstral  coefficient(MFCC) is the  most popular feature and has become standard in the speaker recognition systems. Two options were explored  to better the combined system's results. One of these options was adding auto-encoders. Denoising and simple auto-encoders were explored, both of them came back with worse results. The other option was amending the existing code which resulted in slightly better results.


**Requirements**
- `librosa`
- [`noisereduce`](https://pypi.org/project/noisereduce/)
- [`SRMRpy`](https://github.com/jfsantos/SRMRpy)
- `networkx == 2.2`
- `scikit-learn`
- `numpy`
- `pandas`
- `tqdm`
- `decorators`
- `audioread`

## Usage

#### 1. Clone this repository
#### 2. Download datasets
- Datasets are available [here](https://zenodo.org/record/3678171)
- Datasets for all machines can be downloaded and unzipped by running
    - `sh download_dev_data.sh` for development data
    - `sh download_eval_data.sh` for evaluation data

#### 3.1 Running System 1
- `cd bin/modspec_graph/`
- `python graph_anom_detection.py d` - for running on development data
    - Modulation Spectrums for each machine-id will be stored in `npy` files in `saved/` in the same directory
    - The results for development data are stored in `modspec_graph_dev_data_results.csv` in the same directory
- `python graph_anom_detection.py e` - for running on evaluation data
    - The results for evaluation data are stored in the submission format in the directory `task2`
########### 3.1.1 Running simple Autoencoder on System 2
- Do as specified in step 3.1
- uncomment line 83 in `graph_anom_detection.py` class `X_train = get_model(np.array(X_train),np.array(X_test))`
########## 3.1.2 Running Denoising Autoencoder on System 2
- Do as specified in step 3.1
- uncomment line 83 in `graph_anom_detection.py` class `X_train = TRAIN_DENOISE(np.array(X_train),np.array(X_test))`

#### 3.2 Running System 2
- i-Vectors for both development and evaluation have been provided in the zip file -  `saved_iVectors/ivector_mfcc_100.zip`
- Unzip `ivector_mfcc_100.zip` in the same directory
    - Code for extracting i-Vectors will be added soon
- `cd bin/iVectors_gmm/`
- `python gmm.py d` - for running on development data
    - The results for development data are stored in `iVectors_gmm_dev_data_results.csv` in the same directory
- `python gmm.py e` - for running on evaluation data
    - The results for evaluation data are stored in the submission format in the directory `task2`
########### 3.2.1 Running simple Autoencoder on System 2
- Do as specified in step 3.2
- uncomment line 77 in `gmm.py` class ``# X_train = get_model(X_train,X_test)``
########## 3.2.2 Running Denoising Autoencoder on System 2
- Do as specified in step 3.2
- uncomment line 77 in `gmm.py` class `X_train = TRAIN_DENOISE(X_train,X_test)`


#### 3.3 Running ensemble of System 1 and System 2
- Run both System 1 and System 2
- `cd bin/ensemble/`
- `python ens.py d` - for running on development data
    - The results for development data are stored in `ensemble_dev_data_results.csv` in the same directory
- `python ens.py e` - for running on evaluation data
    - The results for evaluation data are stored in the submission format in the directory `task2`
########### 3.3.1 Running ensemble of System 1 and System 2 with Denoising Autoencoder
- Perform steps 3.1.2 & 3.2.2
- Do as specified in step 3.3

########### 3.3.1 Running ensemble of System 1 and System 2 with Denoising Autoencoder
- Perform steps 3.1.1 & 3.2.1
- Do as specified in step 3.3
