from keras.models import model_from_json
import cv2
import glob
import numpy as np


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")

'''loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
'''
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

image_list = [cv2.imread(file) for file in glob.glob('test/normal/*.png')]


img = cv2.imread("test/normal/2_63new.png")
img = cv2.resize(img, (50,50))
print(img.shape)
img = img.reshape(1, 50, 50, 3)

print(img.shape)
print(np.argmax(loaded_model.predict(img, verbose=1)))
print(loaded_model.predict(img))
