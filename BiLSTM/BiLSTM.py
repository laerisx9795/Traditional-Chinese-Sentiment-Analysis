import numpy
from numpy import array
from collections import defaultdict
import tensorflow
from tensorflow import keras
from keras import layers
from keras import backend
from keras_preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
from ckip_transformers.nlp import CkipWordSegmenter

#print(tensorflow.__version__)

# def cal_TP(y_true, y_pred):
#     print(backend.int_shape(y_true))
#     print(backend.int_shape(y_pred))
#
#     y_pred_pos = backend.round(backend.clip(y_pred, 0, 1))  #   predict pos
#
#     y_pos = backend.round(backend.clip(y_true, 0, 1))   #   Pos
#
#     TP = backend.sum(y_pos * y_pred_pos)
#     return TP
#
# def cal_FN(y_true, y_pred):
#     y_pred_pos = backend.round(backend.clip(y_pred, 0, 1))  #   predict pos
#     y_pred_neg = 1 - y_pred_pos #   predict_neg
#
#     y_pos = backend.round(backend.clip(y_true, 0, 1))   #   Pos
#
#     FN = backend.sum(y_pos * y_pred_neg)
#     return FN
#
# def cal_FP(y_true, y_pred):
#     y_pred_pos = backend.round(backend.clip(y_pred, 0, 1))  #   predict pos
#
#     y_pos = backend.round(backend.clip(y_true, 0, 1))   #   Pos
#     y_neg = 1 - y_pos   #   Neg
#
#     FP = backend.sum(y_neg * y_pred_pos)
#     return FP
#
# def cal_TN(y_true, y_pred):
#     y_pred_pos = backend.round(backend.clip(y_pred, 0, 1))  #   predict pos
#     y_pred_neg = 1 - y_pred_pos #   predict_neg
#
#     y_pos = backend.round(backend.clip(y_true, 0, 1))   #   Pos
#     y_neg = 1 - y_pos   #   Neg
#
#     TN = backend.sum(y_neg * y_pred_neg)
#     return TN
#
# def recall_pos(y_true, y_pred):
#     # true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
#     # possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
#     # recall = true_positives / (possible_positives + backend.epsilon())
#     TP = cal_TP(y_true, y_pred)
#     FN = cal_FN(y_true, y_pred)
#     recall_p = TP / (TP + FN + backend.epsilon())
#     return recall_p
#
# def recall_neg(y_true, y_pred):
#     FP = cal_FP(y_true, y_pred)
#     TN = cal_TN(y_true, y_pred)
#     recall_n = TN / (FP + TN + backend.epsilon())
#     return recall_n
#
# def precision_pos(y_true, y_pred):
#     # true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
#     # predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
#     # precision = true_positives / (predicted_positives + backend.epsilon())
#     TP = cal_TP(y_true, y_pred)
#     FP = cal_FP(y_true, y_pred)
#     precision_p = TP / (TP + FP + backend.epsilon())
#     return precision_p
#
# def precision_neg(y_true, y_pred):
#     TN = cal_TN(y_true, y_pred)
#     FN = cal_FN(y_true, y_pred)
#     precision_n = TN / (TN + FN + backend.epsilon())
#     return precision_n
#
# def f1_p(y_true, y_pred):
#     # precision = precision_(y_true, y_pred)
#     # recall = recall_(y_true, y_pred)
#     recall_p = recall_pos(y_true, y_pred)
#     precision_p = precision_pos(y_true, y_pred)
#     f1_p = 2 * ((precision_p * recall_p) / (precision_p + recall_p + backend.epsilon()))
#     return f1_p
#
# def f1_n(y_true, y_pred):
#     recall_n = recall_neg(y_true, y_pred)
#     precision_n = precision_neg(y_true, y_pred)
#     f1_n = 2 * ((precision_n * recall_n) / (precision_n + recall_n + backend.epsilon()))
#     return f1_n

class BiLSTM(object):
    def __init__(self):
        self.max_features = 9000 # 只考慮 9000 個字彙
        self.maxlen = 100    # 每則只考慮前 100 個字
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.encoded_input = []
        self.padded_input = []
        self.M = int()  # 訓練樣本中最大的字詞數
        self.size = int()   #   訓練樣本數
        self.punctuation = ['，', '．', '。', '、', '／', '？', '＼', '｜', '；', '：', '’', '＂', '［', '］', '‵',
                            '～', '！', '＠', '＃', '＄', '％', '︿', '＆', '＊', '（', '）', '－', '＋', '＝', '｛',
                            '｝', '「', '」', '『', '』', '【', '】', '＜', '＞', 'ˇ', 'ˋ', 'ˊ', '˙', '~', '`', '!',
                            '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '-', '+', '=', '{', '}', '[', ']', '|',
                            '\\', '<', '>', '?', ',', '.', '/', ':', '"', ';', '\'', '…', ' ', '\t', '\n', '《', '》',
                            '\u3000', '〈', '〉']
        self.ws_driver = CkipWordSegmenter(model_name="ckiplab/bert-base-chinese-ws")
        self.word2vec_layer = layers.TextVectorization(max_tokens=self.max_features, output_mode="int") #   change word into vector
        self.words_id = defaultdict(int)  # index for words
        self.model = []
        # (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
        #     num_words=self.max_features
        # )

        # 不足長度，後面補0
        #x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=self.maxlen)
        #x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=self.maxlen)

    def preprocess(self, train_set, test_set):
        count = 0
        for datum in train_set:
            sample= []
            label = datum[0]
            review = [datum[1]]
            features = self.ws_driver(review)  # 斷詞
            features = features[0]

            self.y_train.append(label)

            for f in features:
                feat = f
                for p in self.punctuation:
                    if p in f:
                        feat = f.replace(p, '')
                if feat == '':
                    continue
                self.words_id[feat] = count
                count += 1
                sample.append(feat)
            self.word2vec_layer.adapt(sample)
            self.x_train.append(sample)

        self.y_train = array(self.y_train)
        for datum in test_set:
            sample= []
            label = datum[0]
            review = [datum[1]]
            features = self.ws_driver(review)  # 斷詞
            features = features[0]

            self.y_test.append(label)

            for f in features:
                feat = f
                for p in self.punctuation:
                    if p in f:
                        feat = f.replace(p, '')
                if feat == '':
                    continue
                self.words_id[feat] = count
                count += 1
                sample.append(feat)
            self.word2vec_layer.adapt(sample)
            self.x_test.append(sample)

        self.y_test = array(self.y_test)
        self.M = max([len(record) - 1 for record in self.x_train])  # 訓練樣本中最大的字詞數
        self.size = len(self.x_train)   # 訓練樣本數
        self.encoded_input = [self.word2vec_layer(d) for d in self.x_train]#[one_hot(d, self.M * self.size) for d in self.x_train]  # 轉為數字向量
        self.padded_input = pad_sequences(self.encoded_input, maxlen=self.M, padding='post')   # 補0

    def train(self):
        inputs = keras.Input(shape=(None,), dtype="int64")  # 可輸入不定長度的整數陣列
        x = layers.Embedding(self.max_features, 128, input_length=self.M, trainable=False)(inputs)
        # 使用 2 個 bidirectional LSTM
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)    #   64
        x = layers.Bidirectional(layers.LSTM(64))(x)
        # 分類
        outputs = layers.Dense(1, activation="sigmoid")(x)  #sigmoid
        self.model = keras.Model(inputs, outputs)
        self.model.summary()
        self.model.compile("adam", "binary_crossentropy", metrics=["accuracy"])#, f1_p, f1_n, precision_pos, precision_neg,
                                                                   #recall_pos, recall_neg, cal_TP, cal_FN, cal_FP, cal_TN])
        self.model.fit(self.padded_input, self.y_train, batch_size=32, epochs=2, validation_split=0.2)  #2

    def predict(self):
        encoded_input = [self.word2vec_layer(d) for d in self.x_test]#[one_hot(d, self.max_features) for d in self.x_test]  # 轉為數字向量
        padded_input = pad_sequences(encoded_input, maxlen=self.M, padding='post')  # 補0
        loss, accuracy = self.model.evaluate(padded_input, self.y_test)
        #, f1_p, f1_n, precision_p, precision_n, recall_p, recall_n, TP, FN, FP, TN
        print("Accuracy:\t", accuracy)
        return accuracy
        # print("TP:\t", TP, "FN:\t", FN)
        # print("FP:\t", FP, "TN:\t", TN)
        # print("Precision pos:\t", precision_p)
        # print("Recall pos:\t", recall_p)
        # print("F1 Score pos:\t", f1_p)
        # print("Precision neg:\t", precision_n)
        # print("Recall neg:\t", recall_n)
        # print("F1 Score neg:\t", f1_n)
        # print("Average F1 Score:\t", (f1_p + f1_n) / 2)