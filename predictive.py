import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/SOUVIK/Downloads/assesment/trained_model.sav', 'rb'))

input_data = np.array([60, 1, 0, 100, 110, 235])   #taking user input

input_data_as_numpy_array = np.asarray(input_data) # Convert to numpy array


input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # it's reshaping the data into a 2D array with one row and as many columns as there are elements in 
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is Happy :)')
else:
  print('The person is not Happy :(')
