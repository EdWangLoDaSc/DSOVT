import sys
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import h5py
from tqdm import tqdm

def generate_data(f, sen_num_kind_list, sen_num_var_list):
    lat = np.array(f['lat'])
    lon = np.array(f['lon'])
    sst = np.array(f['sst'])
    sst1 = np.nan_to_num(sst)
    sst_reshape = sst[0, :].reshape(len(lat[0, :]), len(lon[0, :]), order='F')
    xv1, yv1 = np.meshgrid(lon[0, :], lat[0, :])

    total_samples = 1040 * len(sen_num_kind_list) * len(sen_num_var_list)
    X_ki = np.zeros((total_samples, len(lat[0, :]), len(lon[0, :]), 2))
    y_ki = np.zeros((total_samples, len(lat[0, :]), len(lon[0, :]), 1))

    for ki, sen_num in enumerate(sen_num_kind_list):
        for va, var in enumerate(sen_num_var_list):
            for t in tqdm(range(1040), desc=f"Processing sensor {sen_num}, variable {var}"):
                y_t = np.nan_to_num(sst[t, :].reshape(len(lat[0, :]), len(lon[0, :]), order='F'))
                sparse_locations_lat = np.random.randint(len(lat[0, :]), size=(sen_num))
                sparse_locations_lon = np.random.randint(len(lon[0, :]), size=(sen_num))
                sparse_locations = np.column_stack((sparse_locations_lat, sparse_locations_lon))

                for s in range(sen_num):
                    while np.isnan(sst_reshape[sparse_locations[s, 0], sparse_locations[s, 1]]):
                        sparse_locations[s] = np.random.randint(len(lat[0, :]), size=2)

                sparse_data = y_t[sparse_locations[:, 0], sparse_locations[:, 1]]
                sparse_locations_ex = np.column_stack((lat[0, sparse_locations[:, 0]], lon[0, sparse_locations[:, 1]]))
                grid_z0 = griddata(sparse_locations_ex, sparse_data, (yv1, xv1), method='nearest')
                grid_z0 = np.nan_to_num(grid_z0)

                mask_img = np.zeros_like(grid_z0)
                mask_img[sparse_locations[:, 0], sparse_locations[:, 1]] = 1

                index = ki * len(sen_num_var_list) * 1040 + va * 1040 + t
                X_ki[index, :, :, 0] = grid_z0
                X_ki[index, :, :, 1] = mask_img
                y_ki[index, :, :, 0] = y_t

    return X_ki[:,:,:,:1], y_ki

def main():
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <output_dir>")
        return
    
    output_dir = sys.argv[1]

    file_path = '/content/drive/MyDrive/Physics/Physics/sst_weekly.mat'
    f = h5py.File(file_path, 'r')
    
    sen_num_kind_list = [200, 240, 280, 320]
    sen_num_var_list = [300, 100, 10]
    X_ki, y_ki = generate_data(f, sen_num_kind_list, sen_num_var_list)
    np.save(f'{output_dir}x_NOAA_train.npy', X_ki)
    np.save(f'{output_dir}y_NOAA_train.npy', y_ki)

    sen_num_kind_list = [200, 240, 280, 300, 320, 340]
    sen_num_var_list = [900]
    X_ki, y_ki = generate_data(f, sen_num_kind_list, sen_num_var_list)
    np.save(f'{output_dir}x_NOAA_test.npy', X_ki)
    np.save(f'{output_dir}y_NOAA_test.npy', y_ki)
    
    f.close()
    x_train = np.load(f'{output_dir}/x_NOAA_train.npy',mmap_mode = 'r')
    y_train = np.load(f'{output_dir}/y_NOAA_train.npy',mmap_mode = 'r')
    x_test = np.load(f'{output_dir}/x_NOAA_test.npy',mmap_mode = 'r')
    y_test = np.load(f'{output_dir}/y_NOAA_test.npy',mmap_mode = 'r')


    x_data = np.vstack([x_train,x_test])

    y_data = np.vstack([y_train,y_test])
    np.save(f'{output_dir}/x_NOAA.npy',x_data)
    np.save(f'{output_dir}/y_NOAA.npy',y_data)

if __name__ == '__main__':
    main()

