import os
import pandas as pd
import numpy as np

from keras.utils import to_categorical
from keras import layers, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

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

model.compile(optimizer=SGD(), loss='binary_crossentropy', metrics=['accuracy'])


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_pred_score = np.empty(shape=[0, 2])
predicted_index = np.empty(shape=[0, ])
for train_index, test_index in skf.split(predictors, y[:, 1]):
    X_train, X_test = predictors.loc[train_index].values, predictors.loc[test_index].values
    y_train, y_test = y[train_index, :], y[test_index, :]

    model.fit(X_train, y_train, steps_per_epoch=10, shuffle=True, verbose=True, callbacks=[EarlyStopping(patience=2)])
    prediction_i = model.predict(X_test)
    y_pred_score = np.append(y_pred_score, prediction_i, axis=0)
    predicted_index = np.append(predicted_index, test_index)
    del X_train, X_test, y_train, y_test

y_pred_score = np.delete(y_pred_score, 0, axis=1)

tresholds = np.linspace(0.01, 1.0, 1000)
scores = []
for treshold in tresholds:
    y_hat = (y_pred_score > treshold).astype(int)
    y_hat = y_hat.tolist()
    y_hat = [item for sublist in y_hat for item in sublist]

    scores.append([
        roc_auc_score(y_true=y[:, 1], y_score=y_pred_score)])

scores = np.array(scores)
best_values = scores[scores[:, 0].argmax()]
final_tresh = tresholds[scores[:, 0].argmax()]
print(best_values)
print(final_tresh)
