# An Enhanced Wildfire Spread Prediction Using Multimodal Satellite Imagery and Deep Learning Models
#### The original Next Day Wildfire Spread dataset can be downloaded from: https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread 
#### The enhanced dataset we used in this project can be found here: https://www.kaggle.com/datasets/rufaiyusufzakari/enhanced-and-modified-next-day-wildfire-spread 

### Files and Directories Overview:
**`datasets.py`**: Contains custom implementations of PyTorch's `Dataset` class designed specifically for loading and managing the wildfire dataset.
**`pickle_wildfire_datasets.py`**: Extracts random 32×32 patches from the original data, converts them into NumPy arrays, and saves them as pickle files.
**`pickle_full_wildfire_datasets.py`**: Converts the full 64×64 original data directly into NumPy arrays and pickles them.

# References and Acknowledgements
Our code is based on the following repositories, we thank the authors for their excellent contributions.
https://github.com/jfitzg7/paying-attention-to-wildfire and https://github.com/jmichaels32/fireprediction 
