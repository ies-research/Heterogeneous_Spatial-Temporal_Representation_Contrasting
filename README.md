# Uncertainty_Spatial-Temporal_Attention_Graph_Neural_Network
This code is related to the paper "Time-Series Representation Learning via Heterogeneous Spatial-Temporal Contrasting for Remaining Useful Life Prediction".

## Methodology
This paper introduces a novel contrastive learning paradigm, termed Heterogeneous Spatial-Temporal Representation Contrasting (HSTRC). 
We eliminate sample view augmentation and employ dual branches with a heterogeneous spatial-temporal flipped structure to extract two distinct hidden feature views from the same source data, which avoids disturbing the original time series. 
Leveraging a combination of cross-branch spatial-temporal contrastive and projected feature contrastive loss functions, HSTRC can effectively extract robust representations from unlabeled time series data. 
Remarkably, by only fine-tuning the fully connected layers on top of extracted representations by HSTRC, we achieved the best performance across several RUL prediction datasets, showing up to 19.2% improvements over the state-of-the-art supervised learning method.
Besides, further intensive experiments demonstrate HSTRC's effectiveness in active learning, out-of-distribution testing, and transfer learning scenarios.

## Dataset
The experiment is based on the public data set [Turbofan Engines Degradation Simulation Data Set](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6/about_data)
provided by NASA is being used in this paper for Remaining Useful Life prediction.

## File Structure
```
├───checkpoints          //To save the best fine tunned model 
├───CMAPSSData           //Dataset location 
├───image                //To save the visualization result 
├───requirements.txt     //Requirements for python envoriment 
├───utils.py             //Utilities for dataset loading 
├───HSTRC.ipynb          //Core code of HSTRC  
├───Model                //Self-Supervised Learned Model by HSTRC for each dataset 
└───Result               //It stores the running results of HSTRC step by step on  each dataset 
```

## Setup
Environment setup:

```
python -m venv ./venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=venv
```

Download the dataset:

```
download CMAPSSData.zip 
unzip -d CMAPSSData/ CMAPSSData.zip
```

Run Code:
```
Run HSTRC.ipynb in jupyter server

All the experimental configurations have been set in the code for four datasets FD001, FD002, FD003, FD004. Only change the dataset index, the correspond config will be load automatically. 
IN_IDX = {0,1,2,3}
```