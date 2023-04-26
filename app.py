import streamlit as st
import numpy as np
import joblib

from prediction import get_prediction, ordinal_encoder

model = joblib.load(r'models/SiteEnergyIntensityPrediction.joblib')

st.set_page_config(page_title="Wild Blueberry Yield Prediction App", page_icon="⚡", layout="wide")


#creating option list for dropdown menu
st.markdown("<h1 style='text-align: center;'>Wild Blueberry Yield Prediction App ⚡</h1>", unsafe_allow_html=True)
st.markdown("This project is part of my personal portfolio.")
st.markdown("In this, an attempt is made to predict the yield of the wild blueberry agro-ecosystem.")
st.markdown("The target feature is **:red[yield]** .")
st.markdown("The metric used for evaluation is **:green[RMSE]**")
st.write("You can check the source code on [GitHub](https://github.com/LuisAnibalVasquez/Wild-Blueberry-Yield-Prediction)")

def main():
    with st.form('prediction_form'):
        
        st.subheader("Enter the input for following features:")
        
        year_factor = st.slider('Year factor: ', 1, 6, value=0, format="%d")
        floor_area = st.slider('Floor area: ', 1, 1000000, value=0, format="%d")
        january_avg_temp = st.slider('January avg temp: ', -20, 100, value=0, format="%d") 
        january_max_temp = st.slider('January max temp: ', -20, 100, value=0, format="%d") 
        january_min_temp = st.slider('January min temp: ', -20, 100, value=0, format="%d") 
        february_min_temp = st.slider('February min temp: ', -20, 100, value=0, format="%d") 
        february_avg_temp = st.slider('February avg temp: ', -20, 100, value=0, format="%d") 
        elevation = st.slider('Elevation: ', 0, 10000, value=0, format="%d") 
        energy_start_rating = st.slider('Energy start rating: ', 0, 100, value=0, format="%d") 
        year_buillt = st.slider('YEar built: ', 1900, 2023, value=0, format="%d") 
        submit = st.form_submit_button("Predict")

        if submit:            
            data = np.array([year_factor,floor_area,january_avg_temp,january_max_temp, 
                             january_min_temp, february_min_temp, february_avg_temp, 
                             elevation,energy_start_rating,year_buillt]).reshape(1,-1)
            pred = get_prediction(data=data, model=model)
            st.subheader(f"The predicted Site EUI is:  {pred[0]}")

if __name__ == '__main__':
    main()