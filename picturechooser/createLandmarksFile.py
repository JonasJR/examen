import pandas as pd

#We have to remove the unneeded data from the csv file from OpenFace
#This is the output structure: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format
df = pd.read_csv('/home/jeeenas/ownCloud/PHOTOS/Result.txt')
df.columns = pd.Series(df.columns).str.replace(' ', '')
df = df.drop('frame',1)
df = df.drop('timestamp',1)
df = df.drop('confidence',1)
df = df.drop('success',1)
df.to_csv('/home/jeeenas/ownCloud/PHOTOS/ResultModified.txt', header=False, index=False)
