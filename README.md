# An Enhanced Wildfire Spread Prediction Using Multimodal Satellite Imagery and Deep Learning Models
#### The original Next Day Wildfire Spread dataset can be downloaded from: https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread 
#### The enhanced dataset we used in this project can be found here: https://www.kaggle.com/datasets/rufaiyusufzakari/enhanced-and-modified-next-day-wildfire-spread 

### 📁 Files and Directories Overview

- **`datasets.py`**  
  Custom PyTorch `Dataset` classes tailored for loading and handling wildfire data.

- **`pickle_wildfire_datasets.py`**  
  Processes the original dataset by extracting random 32×32 crops, converting them to NumPy arrays, and saving them as pickle files. Used with legacy Dataset implementations.

- **`pickle_full_wildfire_datasets.py`**  
  Converts full 64×64 wildfire data into NumPy arrays and pickles them. Used with the current Dataset implementations.


# References and Acknowledgements
Our code is based on the following repositories, we thank the authors for their excellent contributions.
https://github.com/jfitzg7/paying-attention-to-wildfire and https://github.com/jmichaels32/fireprediction 
