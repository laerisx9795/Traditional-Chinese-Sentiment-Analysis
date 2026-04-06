import codecs
import csv
import pandas as pd
import efaqa_corpus_zh
from hanziconv import HanziConv
import time
import googletrans

punctuation = ['，','．','。','、','／','？','＼','｜','；','：','’','＂','［','］','‵','～','！','＠','＃','＄','％','︿',
               '＆','＊','（','）','－','＋','＝','｛','｝','「','」','『','』','【','】','＜','＞','ˇ','ˋ','ˊ','˙','~',
               '`','!','@','#','$','%','^','&','*','(',')','_','-','+','=','{','}','[',']','|','\\','<','>','?',',',
               '.','/',':','"',';','\'', '…', '《', '》']

def find(char):
    for p in punctuation:
        if(char == p):
            return 1
    return -1

#   function definition
def getTrainingDatabyCSV(filename):
    data = []
    df = pd.read_csv(filename, encoding="UTF-8")
    for i in df.index:
        # data.append((df['label'][i], HanziConv.toTraditional(df['review'][i])))
        data.append((df['label'][i], df['content'][i]))

    return data

def getEvaluationDatabyCSV(filename):
    data = []
    df = pd.read_csv(filename, encoding="UTF-8")
    for i in df.index:
        data.append((df['label'][i], df['message'][i]))

    return data

def getEvaluationDatabyTXT(filename):
    data = []
    f = codecs.open(filename, "r", encoding="UTF-8")
    for line in f.readlines():
        # data.append(HanziConv.toTraditional(line).replace("\n", "").replace("\r", ""))
        data.append(line.replace("\n", "").replace("\r", ""))

    return data

def getEmotionalDataset():
    translator = googletrans.Translator()
    i = 0
    data = []
    # labels = []
    # messages = []
    l = list(efaqa_corpus_zh.load())
    # print("size: %s" % len(l))
    # print(l[0]["title"])

    output = 'emotional_dataset.csv'
    with open(output, 'w', newline='', encoding='UTF-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['label', 'message'])

        count = 15496   #8940    #3825   #3324   #2109
        for index in range(15496, 20000):

        #for datum in l:
            message = l[index]['title']
            if (message[0] == '男') | (message[0] == '女'):
                message = message[1:]

            for m in message:
                if find(m) != -1:   #not letter
                    message = message.replace(m, '')

            if (message != '') & (message != ' '):
                time.sleep(5)
                message = translator.translate(message, dest='zh-tw').text
                label = l[index]['label']['s2']
                d = [label, message]
                data.append(d)
                writer.writerow([label, message])
            print(count)
            count += 1


    return data