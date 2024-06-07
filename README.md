# Dynamical System Prediction from Sparse Observations Using Deep Neural Networks with Voronoi Tessellation and Physics Constraints (DSOVT)

## NOAA Sea Surface Temperature Data Processing

This script processes sea surface temperature (SST) data to generate training and testing datasets. The script reads SST data from a `.mat` file (available at: [Google Drive Link](https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu)), performs interpolation to handle missing values, generates sparse sampling based on sensor numbers and variables, and saves the processed data to specified output directories as numpy arrays.

### Usage
Execute the script via command line as follows:

```bash
python ./NOAA/NOAA_data_generation.py <output_dir>
```
Replace <output_dir> with the path to your desired output directory.

## NOAA Kriging for Sea Surface Temperature Data

This Jupyter notebook provides a comprehensive framework for processing sea surface temperature (SST) data using Kriging methods. It includes modules for data generation, generates sparse sampling based on sensor numbers and variables, and utilizes advanced geostatistical methods for spatial and temporal prediction.

### Notebook Content
1. **Data Loading**: Load SST data from the provided `.mat` file and true SSR data generated from NOAA_data_generation.py.
2. **Data Preprocessing**: Prepare data for Kriging.
3. **Kriging Implementation**: Implement Ordinary Kriging techniques to predict SST.
4. **Prediction and Visualization**: Generate predictions and visualize the results using matplotlib.
5. **Metrics Evaluation**: Calculate metrics such as SSIM, PSNR and inference time for the predicted outputs.

### Usage
To use this notebook:
1. Open your Jupyter environment.
2. Navigate to the `./NOAA/` directory.
3. Open the `NOAA_kriging.ipynb` notebook.
4. Execute the cells in order to process the data and view the results.

5. 
## Dependencies
The script requires the following dependencies:
- Python 3.x
- NumPy
- Pandas
- SciPy
- h5py
- tqdm
- pykrige
- skimage
- matplotlib
Ensure all dependencies are installed using pip:

```bash
pip install numpy pandas scipy h5py tqdm

