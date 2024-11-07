import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# To do list:
# 1. 取出label並配對
# 2. Data analyze

if __name__ == '__main__':

    id = '0a418b57'

    df_train = pd.read_csv('Dataset/train.csv')
    print(df_train.columns)
    print(df_train.head())
    internet_use_level = int(df_train[df_train['id']==id]['sii'].values[0])


    base_path = os.path.dirname(os.path.realpath(__file__))
    parquet_file_name = 'part-0.parquet'
    parquet_file_path = os.path.join(base_path, 'Dataset', 'series_train.parquet', 'id='+id, parquet_file_name)
    df_motion = pd.read_parquet(parquet_file_path, engine='pyarrow')
    print(df_motion.head())

    motion_x = df_motion['X']
    motion_y = df_motion['Y']
    motion_z = df_motion['Z']
    motion = np.sqrt(motion_x**2+motion_y**2+motion_z**2)

    plt.figure(figsize=(9, 4))
    plt.plot(motion_x[:5000], label='X')
    plt.plot(motion_y[:5000], label='Y')
    plt.plot(motion_z[:5000], label='Z')
    plt.plot(motion[:5000], label='XYZ')
    plt.title('ID{}, Internet use level:{}'.format(id, internet_use_level))
    plt.legend()
    plt.show()
