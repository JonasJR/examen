import os
import shutil
import pandas as pd

#I had to get all the 2k images from the 10k folder to get the 2k that was
#rated by the Amazon Mechanical Turk. So I made a script that copies
#the images that was used into the folder
src = '/home/jeeenas/Documents/PHOTOS/10k US Adult Faces Database/Face Images BACKUP/'
dest = '/home/jeeenas/Documents/PHOTOS/10k US Adult Faces Database/2k'
df = pd.read_csv('/home/jeeenas/Documents/PHOTOS/Full Attribute Scores/target-filenames.txt')
saved_column = df.Filename

for file_name in saved_column:
    full_file_name = os.path.join(src,file_name)
    if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, dest)
