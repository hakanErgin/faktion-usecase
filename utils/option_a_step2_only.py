import os
import pandas as pd
import tabulate
from matplotlib import pyplot as plt
from utils.visualize_substractions import visualize_substractions


def classifier_A_step2(path):
    df = pd.DataFrame(
        columns=["id", "result"])
    ids = []
    datas = []
    for filename in os.listdir(path):
        id = filename[:-4]
        res = visualize_substractions(inputfile=path+filename)
        ids.append(id)
        datas.append(res)
        dfrow = [id]
        dfrow.append(res)
        df.loc[len(df)] = dfrow
    plt.hist(df.result)
    plt.xlim([0, 1])
    plt.title(path)
    # plt.show()
    return df


df_normal = classifier_A_step2(path='./assets/normal_dice/0/')
for i in range(1, 11):
    df_normal = pd.DataFrame.append(df_normal, classifier_A_step2(path='./assets/normal_dice/'+str(i)+'/'))


print(df_normal.shape)
print(df_normal.describe())
plt.hist(df_normal.result)
plt.xlim([0, 0.1])
plt.title('Minimum sum of pixel values - Normal dice only')
plt.show()


#df = classifier_A_step2(path='./assets/normal_dice/2/')
#df = classifier_A_step2(path='./assets/normal_dice/3/')
# print(df.head().to_markdown())
# df.to_csv('./assets/df_classifier_A_step2.csv', sep=',', index=False, mode='w')

df_anomalous = classifier_A_step2(path='./assets/anomalous_dice/')
print(df_anomalous.describe())

import math
w = 0.02
n = math.ceil((df_anomalous['result'].max() - df_anomalous['result'].min())/w)

plt.hist(df_anomalous.result, bins=n)
plt.xlim([0, 0.1])
plt.title('Minimum sum of pixel values - Anomalous dice only')
plt.show()

print(df_anomalous.head(10).to_markdown())
