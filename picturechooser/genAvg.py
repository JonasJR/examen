import pandas as pd

#I had to make an avarage of the informatiopn provided by the Amazon Mechanical Turk
#so I read in the information and create an avarage of the values provided
df = pd.read_csv('/home/jeeenas/ownCloud/PHOTOS/attractive.csv')
Filename = df.Filename
ImageNumber = df.ImageNumber
Age = df.Age
Attractive = df.Attractive
Gender = df.Gender

counter = 0
ImgList = []
tempAge = 0
tempAtt = 0
tempGen = 0
for i in range(0,26663):
    tempAge += Age[i]
    tempAtt += Attractive[i]
    tempGen += Gender[i]
    try:
        if ImageNumber[i] != ImageNumber[i+1]:
            tempAge = tempAge/12
            tempAtt = tempAtt/12
            tempGen = tempGen/12
            temp = [Filename[i],ImageNumber[i],tempAge,tempAtt,tempGen]
            ImgList.append(temp)
            tempAge = 0
            tempAtt = 0
            tempGen = 0
    except:
        tempAge = tempAge/12
        tempAtt = tempAtt/12
        tempGen = tempGen/12
        temp = [Filename[i],ImageNumber[i],tempAge,tempAtt,tempGen]
        ImgList.append(temp)
        tempAge = 0
        tempAtt = 0
        tempGen = 0

out = pd.DataFrame(ImgList, columns=['Filename', 'ImageNumber', 'Age', 'Attractive', 'Gender'])
out.to_csv('/home/jeeenas/ownCloud/PHOTOS/dataAvg.csv', index=False)
