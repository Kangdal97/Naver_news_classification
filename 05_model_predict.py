import pandas as pd

pd.options.display.max_rows = 20
pd.set_option('display.unicode.east_asian_width', True)
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import *
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import warnings
warnings.filterwarnings(action='ignore')
import pickle

# import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0], [
#             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)

df = pd.read_csv('./crawling/naver_headline_news_21.11.16.csv')
print(df.head())
print(df.info())

X = df['title']
Y = df['category']

'''Encoder 불러오기'''
with open('./models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)
labeled_Y = encoder.transform(Y)
label = encoder.classes_
onehot_Y = to_categorical(labeled_Y)
print(onehot_Y)

okt = Okt()

for i in range(len(X)):
    X[i] = okt.morphs(X[i])
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

'''토큰 불러오기'''
with open('./models/news_token.pickle', 'rb') as f:
    token = pickle.load(f)
tokened_X = token.texts_to_sequences(X)
print(token.index_word)

for i in range(len(tokened_X)):
    if 26 < len(tokened_X[i]):
        tokened_X[i] = len(tokened_X[i][:26])

X_pad = pad_sequences(tokened_X, 26)
print(X_pad[-5:])


'''모델 불러오기'''
model = load_model('./models/news_category_classification_model_0.7571.h5')

'''모델 예측하기'''
# my_sample = np.random.randint(30)
# sample = X_pad[my_sample]
# sample = sample.reshape(1,26)
# pred = model.predict(sample)
# print('\n정답은 :', label[np.argmax(onehot_Y[my_sample])])
# print('예측값은 :',label[np.argmax(pred)])
preds = model.predict(X_pad)
predicts = []
for pred in preds:
    predicts.append(label[np.argmax(pred)])
print(predicts)
df['predicts'] = predicts
print(df.head(30))

pd.options.display.max_columns = 20
pd.set_option('display.unicode.east_asian_width', True)

# df['OX'] = 0
# for i in range(len(df)):
#     if df.loc[i, 'category']==df.loc[i,'predict']:
#         df.loc[i, "OX"] = 'O'
#     else:
#         df.loc[i, "OX"] = 'X'
# print(df.head(30))
# print(df['OX'].value_counts())

print('\n\n=====DONE=====')
print('    code 0     ')
print('==============')
