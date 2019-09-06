import os
import pandas as pd

from keras.utils import to_categorical
from keras import layers, Input
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras.regularizers import L1L2

from resources import STRING, sampling
from sklearn.model_selection import train_test_split


file_list = [filename for filename in os.listdir(STRING.train_processed) if
             filename.endswith('.csv')]

df = pd.read_csv(STRING.train_processed + file_list[0], sep=',', encoding='utf-8')
'''
import matplotlib.pyplot as plot
import numpy as np
import seaborn as sns
N = len(df.index)
area = (25 * np.random.rand(N)) ** 2
df_0 = df[df['target'] == 0]
df_1 = df[df['target'] == 1]
for i in df.drop('id',axis=1).columns:
    print(i)
    print(df[i].head(10))
    f, ax = plot.subplots(2)
    sns.countplot(df_0[i], ax=ax[0])
    sns.countplot(df_1[i], ax=ax[1])

    plot.show()
'''
predictors = df.drop(['id', 'target'], axis=1)
n_cols = predictors.shape[1]

x_train, x_valid, y_train, y_valid = train_test_split(predictors, df['target'], train_size=0.75, shuffle=True,
                                                      random_state=42)
x_train, y_train = sampling.over_sampling(x_train, y_train, model='ADASYN', neighbors=500)

y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)

input_tensor = Input(shape=(n_cols, ))
x = layers.Dense(40, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.))(input_tensor)
x = layers.Dense(20, activation='relu')(x)
# x = layers.Dropout(0.15)(x)
output_tensor = layers.Dense(2, activation='sigmoid')(x)

model = Model(input_tensor, output_tensor)
print(model.summary())


model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['binary_accuracy'])
model.fit(x_train, y_train, steps_per_epoch=20, epochs=100,
          shuffle=True, verbose=True, callbacks=[EarlyStopping(patience=2, monitor='val_acc')],
          validation_data=(x_valid, y_valid),
          validation_steps=5)
model.save(STRING.model_path + 'model.h5')
