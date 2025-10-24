from pyexpat import features

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import os

#page layout
st.set_page_config(
    page_title = "kaggle competition on Road Risk Prediction",
    page_icon = "🚗",
    layout = "wide"
)
st.title("Road Risk Prediction quiz from kaggle")
st.write("pls. choose diff features for roads to find the safer one")




# with st.sidebar:
#     st.switch_page("Road Risk Prediction")

#data handle
#features =
yes_no =[True,False]
num_lanes =['1','2','3','4','5','6','7','8','9']
#user input logic

def create_features_input_form(col,title):
    with col:
        st.subheader(title)
        road_type = st.selectbox("road type",['urban','rural','highway'] ,key=f"rt_{title}")
        num_lanes = st.slider("number of lanes", 1, 6, 1,3 ,key=f"nl_{title}")
        curvature = st.slider("road curvature", 0.0, 1.0, 0.1, 0.01 ,key=f"c_{title}")
        speed_limit = st.slider("speed limit", min_value =10, max_value = 100, step =10, value =60 ,key=f"sl_{title}")
        #speed_limit = st.slider("road speed limit", 0.0, 1.0, 0.1, 0.01 ,key=f"sl_{title}")
        lighting = st.selectbox("road type", ['daylight','dim','night'] ,key=f"lt_{title}")
        weather = st.selectbox("road type", ['rainy','foggy','clear'] ,key=f"wt_{title}")

        road_signs =st.selectbox("road signs",yes_no,key=f"rs_{title}")
        public_road = st.selectbox("public_road", yes_no, key=f"pr_{title}")
        #time_of_day = st.selectbox("time_of_day", time_of_days, key=f"t_{title}")
        time_of_day = st.selectbox("road type", ['morning', 'afternoon', 'evening'], key=f"td_{title}")
        holiday = st.selectbox("holiday", yes_no, key=f"h_{title}")
        school_season = st.selectbox("school_season", yes_no, key=f"ss_{title}")
        num_reported_accidents = st.slider("num_reported_accidents", 0, 5, 1, key=f"na_{title}")

        return {
            "road_type": road_type,
            "num_lanes": num_lanes,
            "curvature": curvature,
            "speed_limit": speed_limit,
            "lighting": lighting,
            "weather": weather,
            #"time_of_day": time_of_day,
            "road_signs": road_signs,
            "public_road": public_road,
            "time_of_day": time_of_day,
            "holiday": holiday,
            "school_season": school_season,
            "num_reported_accidents": num_reported_accidents
        }

col1,col2 = st.columns(2)
roadA_feature = create_features_input_form(col1,"roadA_feature")
roadB_feature = create_features_input_form(col2,"roadB_feature")



# road_types = ['urban','rural','highway']
# lightings = ['daylight','dim','night']
# weathers = ['rainy','foggy','clear']
# time_of_days = ['morning','afternoon','evening']


#we have one-host encode in model train
def encode_features(features):
    "convert user inputed features to models-used one"
    encoded = {
        'num_lanes': features['num_lanes'],
        'speed_limit': features['speed_limit'],
        'curvature': features['curvature'],
        'road_sign_present': 1 if features['road_signs'] else 0,
        'public_road': 1 if features['public_road'] else 0,
        'holiday': 1 if features['holiday'] else 0,
        'school_season': 1 if features['school_season'] else 0,
        'num_reported_accidents': features['num_reported_accidents']
    }
    #roadtype
    for road_tp in features['road_type']:
        encoded[f'road_type_{road_tp}'] = 1 if features["road_type"] == road_tp else 0
    #lighting
    for lt in features['lighting']:
        encoded[f'lighing_{lt}'] = 1 if features["lighting"] == lt else 0
    #weather
    for wt in features['weather']:
        encoded[f'weather_{wt}'] = 1 if features['weather'] == wt else 0
    #time
    for td in features['time_of_day']:
        encoded[f'time_of_day_{td}'] = 1 if features['time_of_day'] == td else 0

    #columns order need to be same as train
    columns = ['num_lanes', 'speed_limit', 'curvature', 'road_signs_present', 'public_road', 'holiday',
                'school_season','num_reported_accidents',
               'road_type_highway', 'road_type_rural', 'road_type_urban',
                'lighting_daylight', 'lighting_dim', 'lighting_night',
               'weather_clear', 'weather_foggy','weather_rainy',
                'time_of_day_afternoon', 'time_of_day_evening', 'time_of_day_morning']
    #return pd.DataFrame(encoded, columns=columns)
    return pd.DataFrame([encoded], columns=columns)

#load model
def load_model():
    try:
        model = joblib.load(Path(os.getcwd()).parent.parent/'downloads'/'playground_series_s5e10'/'road_risk_model.pkl')
        return model
    except Exception as e:
        st.error(e)
        return None

model = load_model()
#interactive logic



if st.button('predict road risk') and model:
    roadA_encoded_features = encode_features(roadA_feature)
    print(type(roadA_feature))
    roadB_encoded_features = encode_features(roadB_feature)

    risk1 = model.predict(roadA_encoded_features.to_numpy())[0]
    risk2 = model.predict(roadB_encoded_features.to_numpy())[0] #slice here is for?

    st.subheader("road risk result:")
    col1, col2, col3 = st.columns(3)
    with col1:
        #st.write(risk1)
        st.metric("roadA risk",risk1)
    with col2:
        #st.write(risk2)
        st.metric("roadB risk",risk2)
    with col3:
        if risk1 > risk2:
            st.success("RoadB is safer than RoadA")
        elif risk1 < risk2:
            st.success("RoadA is safer than RoadB")
        else:
            st.info("roadA is safer same as roadB")




#notes
with st.expander("explanation"):
    st.write('choose diff features for roadA in left and roadB in right')
    st.write("click the predict rick button")
    st.write('lower the risk, safer the road')
