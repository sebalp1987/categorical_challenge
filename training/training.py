import os
import pandas as pd

from keras.utils import to_categorical
from keras import layers, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from resources import STRING

file_list = [filename for filename in os.listdir(STRING.train_processed) if
             filename.endswith('.csv')]

df = pd.read_csv(STRING.train_processed + file_list[0], sep=',', encoding='utf-8')

y = to_categorical(df['target'])
predictors = df.drop(['id', 'target'], axis=1)
n_cols = predictors.shape[1]

input_tensor = Input(shape=(n_cols, ))
x = layers.Dense(8, activation='tanh')(input_tensor)
output_tensor = layers.Dense(2, activation='sigmoid')(x)

model = Model(input_tensor, output_tensor)
print(model.summary())

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(predictors, y, steps_per_epoch=10, epochs=100,
          shuffle=True, verbose=True, callbacks=[EarlyStopping(patience=2)])
model.save(STRING.model_path + 'model.h5')