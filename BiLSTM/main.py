import random
import ReadFile
import BiLSTM
import time

dataset = ReadFile.getEvaluationDatabyCSV("data/waimai_10k.csv")
# dataset = ReadFile.getEvaluationDatabyCSV('NewsTitle.csv')
# numberOfDatum = 8787
# amountOfClass = [2086, 1216, 981, 568, 618, 2022, 1296]    #   HAPPINESS, ANGER, SADNESS, FEAR, DISGUST, SURPRISE, DON'T CARE
path = 'result_waimai.txt'
f = open(path, 'w', encoding='UTF-8')

def find(target, list):
    for item in list:
        if target == item:
            return 1
    return -1

for count in range(0, 12):
    f.write(str(count+1) + '.')
    model = BiLSTM.BiLSTM()
    train_set = []
    test_set = []
    random_index = []

    while(len(random_index) < (11987*0.7)):
        index = random.randint(0, 11987)
        if find(index, random_index) == -1:
            random_index.append(index)

    # while(len(random_index) < 3200):    #   pos 2800 3200
    #     index = random.randint(0, 4000)
    #     if find(index, random_index) == -1:
    #         random_index.append(index)
    # while(len(random_index) < 9585):    #   neg 8391 9585
    #     index = random.randint(4000,11987)
    #     if find(index, random_index) == -1:
    #        random_index.append(index)

    # for emotion in range(0, 7):
    #     while (len(random_index) < amountOfClass[emotion] * 0.7):
    #         index = random.randint(0, amountOfClass[emotion])
    #         if find(index, random_index) == -1:
    #             random_index.append(index)

    random_index.sort()

    for i in range(0, 11987):
        if find(i, random_index) == 1:
            train_set.append(dataset[i])
        else:
            test_set.append(dataset[i])

    # for i in range(0, numberOfDatum):
    #     if find(i, random_index) == 1:
    #         train_set.append(dataset[i])
    #     else:
    #         test_set.append(dataset[i])

    model.preprocess(train_set, test_set)
    model.train()
    acc = model.predict()
    f.write(str(acc) + '\n')

f.close()
