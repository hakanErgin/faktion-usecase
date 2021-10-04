import os


def count_images(path):
    counter = 0
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            counter += 1
    return counter

a = count_images('./assets/anomalous_dice/')
print(f'# of anomalous is {a}')

normal_count = []
for i in range(11):
    normal_count.append(count_images('./assets/normal_dice/'+str(i)+'/'))
b = sum(normal_count)
print(f'# of normal is {b}')

p = 100*a/(a+b)
print(f'Proportion of anomalous is {round(p,2)}%')

print(normal_count)
