from PyEMD import CEEMDAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sampen import sampen2


def ceemdan_decom(series, name, mode='ceemdan', trials=20, draw=1, isrun=1):
    if isrun:
        print('##################################')
        print('%s decomposition is running.' % str.upper(mode))
        decom = CEEMDAN()
        decom.trials = trials
        decom.noise_seed(12345)
        imfs_emd = decom(series)
        imfs_num = np.shape(imfs_emd)[0]

        if draw:
            # Plot original data
            series_index = range(len(series))
            fig = plt.figure(figsize=(16, 2 * imfs_num))
            plt.subplot(1 + imfs_num, 1, 1)
            plt.plot(series_index, series, color='#0070C0')  # F27F19 orange #0070C0 blue
            plt.ylabel('Original')
            # Plot IMFs
            for i in range(imfs_num):
                plt.subplot(1 + imfs_num, 1, 2 + i)
                plt.plot(series_index, imfs_emd[i, :])
                plt.ylabel('IMF' + str(i+1))
            fig.align_labels()
            plt.tight_layout()
            plt.savefig(name + str.upper(mode) + '.tif', dpi=600, bbox_inches='tight')
            plt.draw()
            plt.pause(3)
            plt.close()
        imfs_df = pd.DataFrame(imfs_emd.T)
        imfs_df.columns = ['imf' + str(i+1) for i in range(imfs_num)]
        pd.DataFrame.to_excel(imfs_df, name + str.upper(mode) + '.xlsx', index=False)
        print('Decomposition complete')
    else:
        imfs_df = pd.read_excel(name + str.upper(mode) + '.xlsx')

    return imfs_df


def sample_entropy(series):
    SE_value = sampen2(series)

    return SE_value[2][1]