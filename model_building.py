
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def set_dataset(analyze_file_count=1):

    df_hbn_instruments = pd.read_csv('./Dataset/train.csv')
    df_hbn_instruments = df_hbn_instruments.dropna(subset="sii")
    ids = df_hbn_instruments['id']
    # df_dataset = pd.DataFrame(columns=['Motion_X', 'Motion_Y', 'Motion_Z', 'Label'])
    df_dataset = pd.DataFrame(columns=['Motion_XYX', 'Label'])

    file_count = 0

    for id in ids:

        if file_count > analyze_file_count:
            break

        label = int(df_hbn_instruments[df_hbn_instruments['id'] == id]['sii'].values[0])

        base_path = os.path.dirname(os.path.realpath(__file__))
        parquet_file_name = 'part-0.parquet'
        parquet_file_path = os.path.join(base_path, 'Dataset', 'series_train.parquet', 'id=' + id, parquet_file_name)

        if os.path.exists(parquet_file_path):

            df_motion = pd.read_parquet(parquet_file_path, engine='pyarrow')

            motion_x = df_motion['X']
            motion_y = df_motion['Y']
            motion_z = df_motion['Z']
            motion＿xyz = np.sqrt(motion_x ** 2 + motion_y ** 2 + motion_z ** 2)

            # df_dataset.loc[len(df_dataset)] = [motion_x, motion_y, motion_z, label]
            df_dataset.loc[len(df_dataset)] = [motion＿xyz, label]

            file_count+=1

        x = df_dataset['Motion_XYX']
        y = df_dataset['Label']

        X_train, X_test, y_train, y_test = train_test_split(x, y)

        return X_train, X_test, y_train, y_test

def build_CNN_model(train_X, train_y, test_X, test_y):
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

    train_X, test_X, train_y, test_y = set_dataset()
    build_CNN_model(train_X, test_X, train_y, test_y)








