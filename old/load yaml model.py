import random
from keras.models import model_from_yaml
import numpy as np

# load YAML and create model
yaml_file = open('/Git/dynamic-follow-tf/models/yamlmodel.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("/Git/dynamic-follow-tf/models/model.h5")
print("Loaded model from disk")


print(loaded_model.predict(np.asarray([[50, 50, random.uniform(1,100)]])))