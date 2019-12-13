import pandas as pd
import numpy as np
from seaborn import countplot
import matplotlib.pyplot as plt

def max_shooting_month(filename):
    df = pd.read_csv('./uploads/'+filename)
    print(df['SHOOTING'])
    max = df.groupby(df['MONTH']).count()
    max_row = np.argmax(max['SHOOTING'].values)
    fig = plt.figure()
    plt.plot(range(12), max['SHOOTING'])
    plt.savefig('./static/shooting_count.png')
    print(max_row)
    return max_row

def average_shooting_month(filename):
    df = pd.read_csv('./uploads/'+ filename)
    shoot = df.groupby(df['MONTH']).count()
    shoot_mean = shoot.mean()
    print(shoot_mean['SHOOTING'])
    return shoot_mean['SHOOTING']
