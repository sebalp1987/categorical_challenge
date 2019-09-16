from resources import STRING
import pandas as pd
import os
from keras.models import load_model

model = load_model(STRING.model_path + 'model.h5')
model.summary()
file_list = [filename for filename in os.listdir(STRING.test_processed) if
             filename.endswith('.csv')]
test = pd.read_csv(STRING.test_processed + file_list[0], sep=',', encoding='utf-8')
test_id = test[['id']]

# preprocessing
test = test.drop('id', axis=1)

# prediction
prediction = model.predict(test)
prediction = prediction[:, 1]
print(prediction)

# post-process
df = pd.DataFrame(prediction, columns=['target'])
df = pd.concat([test_id, df], axis=1)
df.to_csv(STRING.submission, index=False, sep=',', encoding='utf-8')
