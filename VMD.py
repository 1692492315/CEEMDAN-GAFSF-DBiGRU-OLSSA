import random
from vmdpy import VMD
import pandas as pd
import matplotlib.pyplot as plt


def vmd_decom(series, name, mode='vmd', draw=1, isrun=1):
    if isrun:
        print('##################################')
        print('%s decomposition is running.' % str.upper(mode))
        alpha = 1.5*len(series)
        tau = 0
        K = 7
        DC = 0
        init = 1
        tol = 1e-7
        random.seed(12345)
        u, u_hat, omega = VMD(series, alpha, tau, K, DC, init, tol)
        if draw:
            # Plot original data
            fig = plt.figure(figsize=(16, 2*K))
            plt.subplot(1+K, 1, 1)
            plt.plot(series)
            plt.ylabel('Original')
            # Plot IMFs
            for i in range(K):
                plt.subplot(1+K, 1, i+2)
                plt.plot(u[i, :], linewidth=0.2, c='r')
                plt.ylabel('IMF{}'.format(i + 1))
            fig.align_labels()
            plt.tight_layout()
            plt.savefig(name + str.upper(mode) + '.tif', dpi=600, bbox_inches='tight')
            plt.draw()
            plt.pause(3)
            plt.close()
        vmd_df = pd.DataFrame(u.T)
        vmd_df.columns = ['imf1-' + str(i+1) for i in range(K)]
        pd.DataFrame.to_excel(vmd_df, name + str.upper(mode) + '.xlsx', index=False)
        print('Decomposition complete')
    else:
        vmd_df = pd.read_excel(name + str.upper(mode) + '.xlsx')

    return vmd_df