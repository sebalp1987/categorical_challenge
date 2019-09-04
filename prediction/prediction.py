from resources import STRING

import pandas as pd
import numpy as np
from keras.models import load_model

model = load_model(STRING.model_path + 'model.h5')
model.summary()
test = pd.read_csv(STRING.test, sep=',', encoding='utf-8')
test_id = test[['id']]

# preprocessing
test = test[['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']]
test['bin_3'] = np.where(test['bin_3'] == 'T', 1, 0)
test['bin_4'] = np.where(test['bin_4'] == 'Y', 1, 0)

# prediction
prediction = model.predict(test)
prediction = prediction[:, 1]
print(prediction)
prediction = (prediction > 0.5).astype(int)

# post-process
df = pd.DataFrame(prediction, columns=['target'])
df = pd.concat([test_id, df], axis=1)
df.to_csv(STRING.submission, index=False, sep=',', encoding='utf-8')
