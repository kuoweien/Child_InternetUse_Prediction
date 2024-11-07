
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def set_dataset(dataset_type):

    df_hbn_instruments = None
    df_dataset = pd.DataFrame(columns=['Motion_XYX', 'Label'])

    base_path = os.path.dirname(os.path.realpath(__file__))

    if dataset_type == 'Train':
        df_hbn_instruments = pd.read_csv('./Dataset/train.csv')
        dataset_file_path = os.path.join(base_path, 'Dataset', 'series_train.parquet')

    elif dataset_type == 'Test':
        df_hbn_instruments = pd.read_csv('./Dataset/test.csv')
        dataset_file_path = os.path.join(base_path, 'Dataset', 'series_test.parquet')

    else:
        print('wrong input: dataset type')
        return

    if os.path.exists(dataset_file_path):
        for id_string in os.listdir(dataset_file_path):
            id = id_string.split('=')[-1]
            try:
                sii = int(df_hbn_instruments[df_hbn_instruments['id'] == id]['sii'].values[0])
            except IndexError:
                print('IndexError: ID'+id+' have no sii data')
                continue
            except KeyError:
                print('KeyError: ID'+id+' have no sii data')
                continue

            parquet_file_path = os.path.join(dataset_file_path, id_string, 'part-0.parquet')
            df_motion = pd.read_parquet(parquet_file_path, engine='pyarrow')

            motion_x = df_motion['X']
            motion_y = df_motion['Y']
            motion_z = df_motion['Z']
            motion＿xyz = np.sqrt(motion_x ** 2 + motion_y ** 2 + motion_z ** 2)

            # df_dataset.loc[len(df_dataset)] = [motion_x, motion_y, motion_z, label]
            df_dataset.loc[len(df_dataset)] = [motion＿xyz, sii]

    return df_dataset


def build_CNN_model(train_X, train_y, test_X, test_y):
    print('Start Building NN Model!')
    tf.random.set_seed(42)

    # Create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(4)
    ])

    # Compile the model
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])

    model.fit(train_X, train_y, epochs=5)

    print(model.evaluate(test_X, test_y))



if __name__ == '__main__':

    dataset_train = set_dataset('Train')
    dataset_test = set_dataset('Test')
    train_X = dataset_train['Motion_XYX']
    train_y = dataset_train['Label']
    test_X = dataset_test['Motion_XYX']
    test_y = dataset_test['Label']

    build_CNN_model(train_X, test_X, train_y, test_y)








