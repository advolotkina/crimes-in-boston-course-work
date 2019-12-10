from seaborn import countplot
import matplotlib.pyplot as plt
import pandas as pd
import os.path


def crimes_per_year():
    if os.path.isfile('./static/crimes_per_year.png'):
        return
    df = pd.read_csv('./uploads/tmpex0j7dw9.csv')
    fig = plt.figure()
    countplot(data=df, x='YEAR').set_title('Количество преступлений/год')
    plt.savefig('./static/crimes_per_year.png')

def crimes_per_month():
    if os.path.isfile('./static/crimes_per_month.png'):
        return
    df = pd.read_csv('./uploads/tmpex0j7dw9.csv')
    fig = plt.figure()
    countplot(data=df, x='MONTH').set_title('Количество преступлений/месяц')
    plt.savefig('./static/crimes_per_month.png')

def crimes_per_day_of_week():
    if os.path.isfile('./static/crimes_per_day_of_week.png'):
        return
    df = pd.read_csv('./uploads/tmpex0j7dw9.csv')
    fig = plt.figure()
    countplot(data=df, x='MONTH').set_title('Количество преступлений/день недели')
    plt.savefig('./static/crimes_per_day_of_week.png')

def crimes_per_hour():
    if os.path.isfile('./static/crimes_per_hour.png'):
        return
    df = pd.read_csv('./uploads/tmpex0j7dw9.csv')
    fig = plt.figure()
    countplot(data=df, x='HOUR').set_title('Количество преступлений/час(0-23)')
    plt.savefig('./static/crimes_per_hour.png')

if __name__ == '__main__':
    crimes_per_year()
    crimes_per_month()
    crimes_per_day_of_week()
    crimes_per_hour()
