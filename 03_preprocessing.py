import pandas as pd

pd.options.display.max_rows = 20
pd.set_option('display.unicode.east_asian_width', True)
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import warnings

warnings.filterwarnings(action='ignore')
import pickle

df = pd.read_csv('./crawling/naver_news.csv')
print(df.head())
print(df.info())

X = df['title']
Y = df['category']

encoder = LabelEncoder()
labeled_Y = encoder.fit_transform(Y)
label = encoder.classes_

print(labeled_Y[0])
print(label)
with open('./models/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

onehot_Y = to_categorical(labeled_Y)
print(onehot_Y)

okt = Okt()

# print(type(X))
# okt_morph_X = okt.morphs(X[2], stem=True)
# print(X[0])
# print(okt_morph_X)
#
# okt_pos_X = okt.pos(X[2], stem=True)
# print(X[0])
# print(okt_pos_X)
#
# okt_nouns_X = okt.nouns(X[2])
# print(X[0])
# print(okt_nouns_X)

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
print(X)

stopwords = pd.read_csv('./crawling/stopwords.csv', index_col=0)

for j in range(len(X)):
    words = []
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:
            if len(X[j][i]) not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)
print(X)

'''토큰화'''
# 문자 -> 숫자로 변환하는것.
token = Tokenizer()
token.fit_on_texts(X)
tokened_X = token.texts_to_sequences(X)
print(tokened_X[:5])

with open('./models/news_token.pickle', 'wb') as f:
    pickle.dump(token, f)

print(token.index_word)

word_size = len(token.word_index) + 1
print(word_size)

'''문장 길이 맞추기'''
# 제일 긴 문장 길이 찾기
maximum = 0
for i in range(len(tokened_X)):
    if maximum < len(tokened_X[i]):
        maximum = len(tokened_X[i])
print(maximum)

X_pad = pad_sequences(tokened_X, maximum)
print(X_pad[-5:])

'''모델만들기'''
X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.1)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
np.save('./models/news_data_max_{}_size_{}'.format(maximum, word_size), xy)

print('\n\n=====DONE=====')
print('    code 0     ')
print('==============')
