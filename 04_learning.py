import pandas as pd
pd.options.display.max_rows = 20
pd.set_option('display.unicode.east_asian_width', True)
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load('./crawling/news_data_max_26_size_13488.npy', allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

'''모델 생성'''
model = Sequential()
model.add(Embedding(13488, 400, input_length=26))
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))
# LSTM 은 무조건 tanh 만쓴다. / 그리고 무조건 return_sequences = True를 줘야함
# return_sequences 를 True를 안주면 맨 마지막껏만(결과14) 들어감 1~13까지 다시 LSTM 에 넣어주려면 True를 줘야함
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
print(model.summary())

'''모델 학습'''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=200, epochs=2, validation_data=(X_test,Y_test))

'''모델 저장'''
model.save('./models/news_category_classification_model_{}.h5'.format(str(fit_hist.history['val_accuracy'][-1])[:6]))

plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.plot(fit_hist.history['val_accuracy'],  label='val_accuracy')
plt.legend()
plt.show()
print('\n\n=====DONE=====')
print('    code 0     ')
print('==============')