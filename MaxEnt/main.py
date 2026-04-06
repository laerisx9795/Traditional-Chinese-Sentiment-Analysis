import random
import time
import MaxEnt
import ReadFile

start = time.time()
print("Start")
#   split dataset into train_set(1/5) and test_set(4/5)
#   the dataset consist of 4000 positive reviews and 7985 negative reviews, total is 11985
#   choose first 1199 positive reviews and last 1199 negative reviews
# dataset = ReadFile.getEvaluationDatabyCSV("waimai_10k.csv")
#dataset = ReadFile.getEvaluationDatabyCSV("NewsTitle.csv")
dataset_test = ReadFile.getEvaluationDatabyCSV('data/emotional_dataset.csv')
#data = ReadFile.getEmotionalDataset()
numberOfDatum = 8787
files = ['NewsTitle_0.csv', 'NewsTitle_1.csv', 'NewsTitle_2.csv', 'NewsTitle_3.csv', 'NewsTitle_4.csv', 'NewsTitle_5.csv', 'NewsTitle_6.csv']
amountOfClass = [2086, 1216, 981, 568, 618, 2022, 1296]    #   HAPPINESS, ANGER, SADNESS, FEAR, DISGUST, SURPRISE, DON'T CARE
path = 'result.txt'
f = open(path, 'w', encoding='UTF-8')

# for i in range(0, 11987):
#     if i < 2800:    #   Max:4000
#         train_set.append(dataset[i])
#     elif i > 6397:  #   Max:7987
#         train_set.append(dataset[i])
#     else:
#         test_set.append(dataset[i])
def find(target, list):
    for item in list:
        if target == item:
            return 1
    return -1

for count in range(0, 12):
    f.write(str(count) + '.\n')
    model = MaxEnt.MaxEnt()
    models = []
    results = []
    # train_sets = []
    # test_sets = []
    #   training emotional models
    for emotion in range(0, 7):
        f.write('('+ str(emotion) + ')')
        print('Emotion ', emotion, ': ')
        models.append(MaxEnt.MaxEnt())
        dataset = ReadFile.getTrainingDatabyCSV(files[emotion])
        train_set = []
        test_set = []
        random_index = []

        # for datum in dataset:
        #     train_set.append(datum)
        #   choose training data
        while(len(random_index) < amountOfClass[emotion] * 0.7):
            index = random.randint(0, amountOfClass[emotion])
            if find(index, random_index) == -1:
                random_index.append(index)
        while(len(random_index) < (numberOfDatum - amountOfClass[emotion]) * 0.7):
            index = random.randint(amountOfClass[emotion],numberOfDatum)
            if find(index, random_index) == -1:
                random_index.append(index)

        # while(len(random_index) < amountOfClass[i] * 0.7):
        #     l = 0
        #     lb = 0
        #     for amount in amountOfClass:
        #         l += amount*0.7
        #         while (l > amountOfClass[i] * 0.7):
        #             l -= 1
        #
        #         while(len(random_index) < l):
        #             index = random.randint(lb, lb + amount)
        #             if find(index, random_index) == -1:
        #                 random_index.append(index)
        #
        #         lb += amount

        random_index.sort()

    # for i in range(0, 11987):
        for i in range(0, numberOfDatum):
            if find(i, random_index) == 1:
                train_set.append(dataset[i])
            else:
                test_set.append(dataset[i])

        models[emotion].preprocess(train_set)
        models[emotion].train()

        accu = models[emotion].test(test_set)
        f.write(str(accu) + '\n')

    #   predict sample
    for datum in dataset_test:
        index = 0
        emotions = []    #   emotion vector
        for model in models:
            emotions.append(model.predict(datum[1]))  #   predict result

        results.append([emotions, datum[1]])
        f.write(str(index + 1) + '\t')
        index += 1
        f.write(datum[1] + '\t')
        for emotion in emotions:
            f.write(str(emotion)+', ')

        f.write('\n')



end = time.time()
print("End")
print("Run time: ", end - start)
f.close()

