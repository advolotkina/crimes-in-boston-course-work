import matplotlib.pyplot as plt
import pandas as pd
def start(filename):

    # в гистограмму
    df = pd.read_csv('./uploads/' + filename)
    # types = df.groupby(df['MONTH'])['OFFENSE_CODE'].value_counts()
    types = df[['MONTH']].groupby(df['OFFENSE_CODE']).count()
    types = types.sort_values(by='MONTH', ascending=False).head()
    types
    names = ['SICK/INJURED/MEDICAL - PERSON', 'INVESTIGATE PERSON', 'M/V - LEAVING SCENE - PROPERTY DAMAGE',
             'VANDALISM', 'ASSAULT & BATTERY']
    plt.figure(figsize=(20, 5))
    plt.bar(names, types['MONTH'])
    plt.savefig('./static/freq_type.png')