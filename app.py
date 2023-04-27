import streamlit as st
import numpy as np
#import joblib
import pickle
from sklearn.linear_model import LinearRegression
from prediction import get_prediction, scaler



#model = joblib.load(r'models/WildBlueberryYieldPrediction.joblib')
model = pickle.load(open('models/WildBlueberryYieldPrediction.pkl', 'rb'))


st.set_page_config(page_title="Wild Blueberry Yield Prediction App", page_icon="🍇", layout="wide")


#creating option list for dropdown menu
st.markdown("<h1 style='text-align: center;'>Wild Blueberry Yield Prediction App 🍇</h1>", unsafe_allow_html=True)
st.markdown("This project is part of my personal portfolio.")
st.markdown("In this, an attempt is made to predict the yield of the wild blueberry agro-ecosystem.")
st.markdown("The target feature is **:red[yield]** .")
st.markdown("The metric used for evaluation is **:green[RMSE]**")
st.write("You can check the source code on [GitHub](https://github.com/LuisAnibalVasquez/Wild-Blueberry-Yield-Prediction)")

def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                clonesize = st.number_input('Clone size: ', min_value = -10.0, max_value = 10.0, value  = 0.0, step =0.01, format = "%f") 
            with col2:
                honeybee = st.number_input('Honeybee density: ', min_value = -10.0, max_value = 10.0, value  = 0.0, step =0.01, format = "%f")
            with col3:
                bumbles = st.number_input('bumbles density: ', min_value = -10.0, max_value = 10.0, value  = 0.0, step =0.01, format = "%f") 
            with col4:     
                andrena = st.number_input('andrena bee density: ', min_value = -10.0, max_value = 10.0, value  = 0.0, step =0.01, format = "%f") 
        with st.container():
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                osmia = st.number_input('osmia bee density: ', min_value = -10.0, max_value = 10.0, value  = 0.0, step =0.01, format = "%f") 
            with col6:
                MaxOfUpperTRange = st.number_input('Max Of Upper Temp Range: ', min_value = -10.0, max_value = 10.0, value  = 0.0, step =0.01, format = "%f") 
            with col7:
                MinOfUpperTRange = st.number_input('Min Of Upper Temp Range: ', min_value = -10.0, max_value = 10.0, value  = 0.0, step =0.01, format = "%f") 
            with col8:
                AverageOfUpperTRange = st.number_input('Average Of Upper Temp Range: ', min_value = -10.0, max_value = 10.0, value  = 0.0, step =0.01, format = "%f") 
        with st.container():
            col9, col10, col11, col12 = st.columns(4)
            with col9:
                MaxOfLowerTRange = st.number_input('Max Of Lower Temp Range: ', min_value = -10.0, max_value = 10.0, value  = 0.0, step =0.01, format = "%f") 
            with col10:
                MinOfLowerTRange = st.number_input('Min Of Lower Temp Range: ', min_value = -10.0, max_value = 10.0, value  = 0.0, step =0.01, format = "%f") 
            with col11:
                AverageOfLowerTRange = st.number_input('Average Of Lower Temp Range: ', min_value = -10.0, max_value = 10.0, value  = 0.0, step =0.01, format = "%f") 
            with col12: 
                RainingDays = st.number_input('Raining Days: ', min_value = -10.0, max_value = 10.0, value  = 0.0, step =0.01, format = "%f") 
        with st.container():
            col13, col14, col15, col16 = st.columns(4)
            with col13:
                AverageRainingDays = st.number_input('Average Raining Days: ', min_value = -10.0, max_value = 10.0, value  = 0.0, step =0.01, format = "%f") 
            with col14:
                fruitset = st.number_input('Fruit Set: ', min_value = -10.0, max_value = 10.0, value  = 0.0, step =0.01, format = "%f") 
            with col15:
                fruitmass  = st.number_input('Fruit Mass: ', min_value = -10.0, max_value = 10.0, value  = 0.0, step =0.01, format = "%f") 
            with col16:
                seeds = st.number_input('Seeds: ', min_value = -10.0, max_value = 10.0, value  = 0.0, step =0.01, format = "%f") 
        
        submit = st.form_submit_button("Predict")

        if submit:     
            data = np.array([clonesize, honeybee, bumbles, andrena, osmia, 
                             MaxOfUpperTRange, MinOfUpperTRange, AverageOfUpperTRange,
                             MaxOfLowerTRange, MinOfLowerTRange, AverageOfLowerTRange, 
                             RainingDays, AverageRainingDays, 
                             fruitset, fruitmass, seeds]).reshape(1,-1)
            pred = get_prediction(data=data, model=model)
            st.subheader(f"The predicted Yield is:  {pred[0]}")

if __name__ == '__main__':
    main()