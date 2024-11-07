import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    df_hbn_instruments = pd.read_csv('./Dataset/train.csv')
    print(df_hbn_instruments.columns)

    age = df_hbn_instruments['Basic_Demos-Age']
    sex = df_hbn_instruments['Basic_Demos-Sex']
    impairment_score = df_hbn_instruments['PCIAT-PCIAT_Total']
    sii = df_hbn_instruments['sii']
    use_computer_hour_level = ['PreInt_EduHx-computerinternet_hoursday']

    # Draw Distribution
    print()
    impairment_score = impairment_score.dropna()
    data = [age, impairment_score]
    labels = ['age', 'impairment_score']

    fig, ax = plt.subplots()
    bplot = ax.boxplot(data,
                       patch_artist=True,  # fill with color
                       tick_labels=labels)  # will be used to label x-ticks
    plt.show()

