import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def make_heatmap(data):
    ax = plt.axes()
    sns.heatmap(data, annot=True, cmap="coolwarm", square=True)
    ax.set_title('My Chance of Win(Draw) by Tricks')
    plt.xlabel('My Choice', fontsize = 15) # x-axis label with fontsize 15
    plt.ylabel('Opponent Choice', fontsize = 15) # y-axis label with fontsize 15
    plt.show()