import pandas as pd

def clean(filename):
    df = pd.read_csv('./uploads/' + filename)
    df = df.dropna(how='any', subset=['Lat', 'Long'])
    df.to_csv('./uploads/'+filename)

