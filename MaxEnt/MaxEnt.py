from collections import defaultdict
import math
import ReadFile
import torch.cuda
import re
from matplotlib import pyplot as plt
from ckip_transformers.nlp import CkipWordSegmenter

class MaxEnt(object):
    def __init__(self):
        self.feats = defaultdict(int)   #   記錄各特徵元組出現的次數
        self.trainset = []  # 訓練資料集
        self.labels = set()  # 標籤集
        self.size = int()  # 訓練資料集大小
        self.M = int()  # 訓練樣本中最大的特徵個數
        self.ep_ = float()  # 計算經驗分布的特徵期望
        self.w = float()  # 初始化權重
        self.ep = float()  # 計算模型分布的特徵期望
        self.lastw = float()  # 收斂時的權重
        self.feats_id = defaultdict(int)  # 取出特徵位置的index
        self.punctuation = ['，','．','。','、','／','？','＼','｜','；','：','’','＂','［','］','‵',
                            '～','！','＠','＃','＄','％','︿','＆','＊','（','）','－','＋','＝','｛',
                            '｝','「','」','『','』','【','】','＜','＞','ˇ','ˋ','ˊ','˙','~','`','!',
                            '@','#','$','%','^','&','*','(',')','_','-','+','=','{','}','[',']','|',
                            '\\','<','>','?',',','.','/',':','"',';','\'', '…', '《', '》']
        self.ws_driver = CkipWordSegmenter(model_name="ckiplab/bert-base-chinese-ws")

    def preprocess(self, train_set):
        for datum in train_set:
            sample = []
            label = datum[0]
            content = [datum[1]]
            features = self.ws_driver(content)    #   用斷詞結果作為特徵
            features = features[0]
            self.labels.add(label)
            sample.append(label)

            for f in features:
                feat = f
                for p in self.punctuation:
                    if p in f:
                        feat = f.replace(p, '')
                if feat == '':
                    continue
                self.feats[(label, feat)] += 1 #   紀錄次數
                sample.append(feat)
                #print(label, feat)

            self.trainset.append(sample)


    def train(self, max_iter = 1000):
        self._initparams()
        for i in range(max_iter):
            print('iter ', i+1)
            self.ep = self._expectedValue()  # 計算模型分布的特徵期望
            self.lastw = self.w[:]
            for i, win in enumerate(self.w):
                delta = 1.0 / self.M * math.log(self.ep_[i] / self.ep[i])
                self.w[i] += delta  # 更新 w
            #print(self.w, self.feats)
            #   輸出權重分布圖
            # x = range(0, len(self.feats))
            # y = self.w
            # fig, ax = plt.subplots(figsize=(15, 5))
            # ax.scatter(x, y, s=1)
            # plt.show()
            if self._convergence(self.lastw, self.w):  # 判斷算法是否收斂
                break

    def _initparams(self):
        self.size = len(self.trainset)
        self.M = max([len(record) - 1 for record in self.trainset])  # 訓練樣本中最大的特徵個數
        self.ep_ = [0.0] * len(self.feats)

        for i, f in enumerate(self.feats):  #   i for index, f for item in feats
            counts = self.feats[f]
            self.ep_[i] = float(counts) / float(self.size)  # 計算經驗分布的特徵期望
            self.feats_id[f] = i  # 為每個特徵函數分配id
        self.w = [0.0] * len(self.feats)  # 初始化權重
        #   輸出權重分布圖
        # x = range(0, len(self.feats))
        # y = self.w
        # fig, ax = plt.subplots(figsize=(20, 5))
        # ax.scatter(x, y, s=1)
        # plt.show()

    def _expectedValue(self):
        ep = [0.0] * len(self.feats)
        for record in self.trainset:  # 從訓練集中迭代輸出特徵
            features = record[1:]
            prob = self._calprob(features)  # 計算條件機率 P(y|x)
            for f in features:
                for w, label in prob:
                    if (label, f) in self.feats:  # 來自訓練數據的特徵
                        idx = self.feats_id[(label, f)]  # 獲取特徵id
                        ep[idx] += w * (1.0 / self.size)
        return ep

    def _calprob(self, features):
        wgts = [(self._probwgt(features, label), label) for label in self.labels]
        Z = sum([w for w, label in wgts])
        prob = [(w / Z, label) for w, label in wgts]
        return prob

    def _probwgt(self, features, label):
        wgt = 0.0
        for f in features:
            if (label, f) in self.feats:
                wgt += self.w[self.feats_id[(label, f)]]
        return math.exp(wgt)

    def _convergence(self, lastw, w):
        for w1, w2 in zip(lastw, w):
            print('diff: ', abs(w1 - w2))
            if abs(w1 - w2) >= 0.01: return False
        return True

    def test(self, dataset):
        # TP = 0
        # FP = 0
        # TN = 0
        # FN = 0
        correctAns = 0
        totalAns = 0

        for sample in dataset:
            ans = sample[0]
            content = [sample[1]]
            maxPosibility = 0
            label = 0
            features = self.ws_driver(content)   #   斷詞
            features = features[0]
            selected_features = []
            for f in features:
                feat = f
                for p in self.punctuation:
                    if p in f:
                        feat = f.replace(p, '')
                if feat == '':
                    continue
                else:
                    selected_features.append(feat)

            prob = self._calprob(selected_features)
            for result in prob:
                if result[0] > maxPosibility:   #   根據機率高低判斷是正向或負向
                    maxPosibility = result[0]
                    label = result[1]

            totalAns += 1
            if label == ans:
                #   正確
                correctAns += 1

        #     if label == ans:
        #         #   正確
        #         if ans == 1:    #正向
        #             TP += 1
        #         else:
        #             TN += 1
        #     else:
        #         #   錯誤
        #         if ans == 1:    #正向
        #             FN += 1
        #         else:
        #             FP += 1
        #
        # Precision_pos = 0
        # Recall_pos = 0
        # F1_score_pos = 0
        # Precision_neg = 0
        # Recall_neg = 0
        # F1_score_neg = 0
        #
        #
        # if (TP + FP) != 0:
        #     Precision_pos = TP / (TP + FP)
        # if (TP + FN) != 0:
        #     Recall_pos = TP / (TP + FN)
        # if (Precision_pos + Recall_pos) != 0:
        #     F1_score_pos = 2 * Precision_pos * Recall_pos / (Precision_pos + Recall_pos)
        # if (TN + FN) != 0:
        #     Precision_neg = TN / (TN + FN)
        # if (TN + FP) != 0:
        #     Recall_neg = TN / (TN + FP)
        # if (Precision_neg + Recall_neg) != 0:
        #     F1_score_neg = 2 * Precision_neg * Recall_neg / (Precision_neg + Recall_neg)
        # F1_score_avg = (F1_score_pos + F1_score_neg) / 2
        #
        # print('TP: ', TP, '\tFN: ', FN)
        # print('FP: ', FP, '\tTN: ', TN)
        # print('Precision_pos: ', Precision_pos)
        # print('Recall_pos: ', Recall_pos)
        # print('F1 score_pos: ', F1_score_pos)
        # print('Precision_neg: ', Precision_neg)
        # print('Recall_neg: ', Recall_neg)
        # print('F1 score_neg: ', F1_score_neg)
        # print('F1 score_avg: ', F1_score_avg)

        Accuracy = correctAns / totalAns
        print('Accuracy: ', Accuracy)
        return Accuracy



    def predict(self, input):
        maxPosibility = 0
        features = self.ws_driver([input])  # 斷詞
        features = features[0]
        selected_features = []
        for f in features:
            feat = f
            for p in self.punctuation:
                if p in f:
                    feat = f.replace(p, '')
            if feat == '':
                continue
            else:
                selected_features.append(feat)

        prob = self._calprob(selected_features)
        for result in prob:
            if result[0] > maxPosibility:  # 根據機率高低判斷是否是該情緒
                maxPosibility = result[0]
                label = result[1]

        return label