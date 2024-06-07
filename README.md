# Dynamical System Prediction from Sparse Observations Using Deep Neural Networks with Voronoi Tessellation and Physics Constraints (DSOVT)

## Numerical Experiments --- NOAA Sea Surface Temperatur
### NOAA Sea Surface Temperature Data Processing

This script processes sea surface temperature (SST) data to generate training and testing datasets. The script reads SST data from a `.mat` file (available at: [Google Drive Link](https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu)), performs interpolation to handle missing values, generates sparse sampling based on sensor numbers and variables, and saves the processed data to specified output directories as numpy arrays.

#### Usage
Execute the script via command line as follows:

```bash
python ./NOAA/NOAA_data_generation.py <output_dir>
```
Replace <output_dir> with the path to your desired output directory.

### NOAA Kriging for Sea Surface Temperature Data

This Jupyter notebook `./NOAA/NOAA_kriging.ipynb` provides a comprehensive framework for processing sea surface temperature (SST) data using Kriging methods. It includes modules for data generation, generates sparse sampling based on sensor numbers and variables, and utilizes advanced geostatistical methods for spatial and temporal prediction.

### DSOVT (CED-LSTM)
The Jupyter notebook `./NOAA/NOAA_CEDLSTM.ipynb` outlines a framework for predicting sea surface temperature (SST) using the CED-LSTM model. It includes steps for training, predicting, and evaluating the model.


### DSOVT (ConvLSTM)
Similarly, the Jupyter notebook `./NOAA/NOAA_ConvLSTM.ipynb` details a framework for predicting SST using the ConvLSTM model. This approach includes training, multi-step prediction, and model evaluation.

## Numerical Experiments --- Shallow Water Systems
5. 
## Dependencies
The script requires the following dependencies:
- Python 3.7
- NumPy
- Pandas
- SciPy
- h5py
- tqdm
- pykrige
- skimage
- matplotlib
Ensure all dependencies are installed using pip:


