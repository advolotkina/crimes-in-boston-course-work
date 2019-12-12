
import pandas as pd
import numpy as np
from seaborn import countplot
import matplotlib.pyplot as plt

def max_shooting_month():
    df = pd.read_csv('./uploads/tmpex0j7dw9.csv')
    print(df['SHOOTING'])
    max = df.groupby(df['MONTH']).count()
    max_row = np.argmax(max['SHOOTING'].values)
    fig = plt.figure()
    plt.plot(range(12), max['SHOOTING'])
    plt.savefig('./static/shooting_count.png')
    print(max_row)

def average_shooting_month():
    df = pd.read_csv('./uploads/tmpex0j7dw9.csv')
    shoot = df.groupby(df['MONTH']).count()
    shoot_mean = shoot.mean()
    print(shoot_mean['SHOOTING'])