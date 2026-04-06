import codecs
import pandas as pd
from hanziconv import HanziConv

#   function definition
def getEvaluationDatabyCSV(filename):
    data = []
    df = pd.read_csv(filename, encoding="UTF-8")
    for i in df.index:
        data.append((df['label'][i], HanziConv.toTraditional(df['review'][i])))
        # data.append((df['label'][i], df['content'][i]))

    return data

def getEvaluationDatabyTXT(filename):
    data = []
    f = codecs.open(filename, "r", encoding="UTF-8")
    for line in f.readlines():
        data.append(HanziConv.toTraditional(line).replace("\n", "").replace("\r", ""))

    return data