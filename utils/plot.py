import numpy as np
from matplotlib import pyplot as plt
import os

'''<==plot loss curves==>'''


def plot_curve(train_loss, val_loss, wt):
    fontlabel = {'family': 'serif',
                 'style': 'italic',
                 'weight': 'semibold',
                 'color': 'black',
                 'size': 16
                 }

    fonttitle = {'family': 'serif',
                 'style': 'normal',
                 'weight': 'bold',
                 'color': 'black',
                 'size': 17
                 }
    # pretty color
    # https://pic1.zhimg.com/v2-89b08fbdc688b6c09c4c12e30ca39a74_r.jpg

    plt.figure(figsize=(20, 8))
    plt.title("Train/Val Loss", fontsize=18, fontdict=fonttitle)

    # plt.ylim(0,1)

    x = np.arange(1, len(train_loss) + 1)
    plt.plot(x, train_loss, color=(246 / 255, 83 / 255, 20 / 255), linestyle="-", linewidth=2.0, marker='o',
             markersize=2, label='training')
    plt.plot(x, val_loss, color=(124 / 255, 187 / 255, 0), linestyle='-', linewidth=2.0, marker='o', markersize=2,
             label='validation')
    plt.xlabel("Epochs", fontlabel)
    plt.ylabel("MSE(default)", fontlabel)
    plt.tick_params(direction='in', length=4, width=2)

    # hide top/right ticks
    ax = plt.gca()
    for i in ['top', 'right']:
        ax.spines[i].set_visible(False)

    # left/bottom ticks linewidth
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)

    plt.legend(markerscale=1.5, fontsize=16, loc='upper right')
    plt.grid(color='black', linestyle='--', linewidth=0.5)

    plt.savefig(os.path.join('results', wt, 'loss.jpg'))
