import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# smooth handing fuction
def tensorboard_smoothing(x, smooth=0.99):
    x = x.copy()
    weight = smooth
    for i in range(1, len(x)):
        x[i] = (x[i - 1] * weight + x[i]) / (weight + 1)
        weight = (weight + 1) * smooth
    return x

if __name__ == '__main__':

    fig, ax1 = plt.subplots(1, 1, figsize=(7,6))  # a figure with a 1x1 grid of Axes

    #
    ax1.spines['top'].set_visible(True)  # Display the top border of the chart box
    ax1.spines['right'].set_visible(True)
    ax1.spines['bottom'].set_linewidth(1)  # Set the thickness of the bottom axis
    ax1.spines['left'].set_linewidth(1)  # Set the thickness of the left axis

    ax1.grid(True, linewidth=1)  # add grids

    # read CSV files
    len_mean1 = pd.read_csv(r"E:\LLM_results\new\acc_test.csv")


    # plot figures
    ax1.plot(len_mean1['Step'].values, tensorboard_smoothing(len_mean1['Value'].values, smooth=0.7), linestyle='-',
             color="red",
             label='Accuracy of the testing sets during training')


    #
    plt.legend(loc='lower right', fontsize=11)

    #
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)

    #
    y_min = 89
    y_max = 95  #
    ax1.set_ylim(y_min, y_max)

    #
    y_ticks = np.arange(y_min, y_max, 0.5)  #
    ax1.set_yticks(y_ticks)


    # show figures
    plt.show()

    # Save images, which can also be in other formats, such as pdf
    # fig.savefig(fname='./fig1_new/nmse_test.png', format='png')