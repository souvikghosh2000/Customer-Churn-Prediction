import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open('C:/Users/SOUVIK/Downloads/assesment/trained_model.sav', 'rb'))

def pred(input_data):
    input_data = np.array([60, 1, 0, 100, 110, 235])   #taking user input

    input_data_as_numpy_array = np.asarray(input_data) # Convert to numpy array

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # it's reshaping the data into a 2D array with one row and as many columns as there are elements in 
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is Happy :)'
    else:
        return 'The person is not Happy :('


def main():
    # providing a title
    st.title('Customer Churn Web Application')

    # getting input from user 
    

    Age = st.text_input("Enter the Age")
    Gender = st.text_input("Enter the Gender;   Male-0   Female-1")
    Location = st.text_input("Enter the Location;   Los Angeles = 0 , New York = 1, Miami = 2, Chicago = 3, Houston = 4")
    Subscription_Length_Months = st.text_input("Enter the Subscription Length in terms of Months")

    Monthly_Bill= st.text_input("Enter the Monthly Bill")
    Total_Usage_GB= st.text_input("Enter Total GB Used")


    # prdiction value
    output = ' '
    if st.button("Churn Test"):
        input_data = np.array([Age, Gender, Location,Subscription_Length_Months , Monthly_Bill, Total_Usage_GB]) 
        output = pred(input_data)
    st.success(output)


if __name__ == '__main__':
    main()
