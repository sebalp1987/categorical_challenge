import os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_input_path = root_path + "/data_input/"
data_output_path = root_path + "/data_output/"
model_input_path = root_path + "/model_input/"
model_output_path = root_path + "/model_output/"
model_path = root_path + "/models/"
train = data_input_path + "train.csv"
test = data_input_path + "test.csv"

train_processed = model_input_path + "/train_processed.csv/"
test_processed = model_input_path + "/test_processed.csv/"
submission = model_output_path + "submission.csv"