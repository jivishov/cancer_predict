import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from pathlib import Path

def get_clean_data():
  data = pd.read_csv("https://raw.githubusercontent.com/jivishov/cancer_predict/main/data/data.csv")
 
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data


def add_sidebar():
  st.sidebar.header("Hüceyrə nüvələrinin ölçüləri")
  
  data = get_clean_data()
  
  slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )
    
  return input_dict


def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict
  

def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig


def add_predictions(input_data):
  pkl_path = Path(__file__).parents[0]
  model = pickle.load(open(f"{pkl_path}/model.pkl", "rb"))
  scaler = pickle.load(open(f"{pkl_path}/scaler.pkl", "rb"))
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  input_array_scaled = scaler.transform(input_array)
  prediction = model.predict(input_array_scaled)
  
  st.markdown("""
    <style>
    .red-background {
        background-color: red;
        color: white;
        padding: 10px;
        text-align:center;
    }
    </style><h5 class='red-background'>Hüceyrə klasteri proqnozu</h5)""", unsafe_allow_html=True)
  st.write("Hüceyrə klasteri:")
  
  if prediction[0] == 0:
    st.write("<div class='centered-container'><span class='diagnosis benign'>Xoşxassəli</span></div>", unsafe_allow_html=True)
  else:
    st.write("<div class='centered-container'><span class='diagnosis malicious'>Bədxassəli</span></div>", unsafe_allow_html=True)
    
  
  st.write("Xoşxassəli şişin olma ehtimalı (%): ", round(model.predict_proba(input_array_scaled)[0][0]*100,2))
  st.write("Bədxassəli şişin olma ehtimalı (%): ", round(model.predict_proba(input_array_scaled)[0][1]*100,2))
  
  st.markdown("""
    <style>
    .red-background {
        background-color: red;
        color: white;
        padding: 10px;
    }
    </style>
    <div class="red-background">
        Nəticələr diaqnozun təyin edilməsində kömək edə bilər, lakin mütəxəssis rəyini əvəzləməməli.
    </div>
    """, unsafe_allow_html=True)


def main():
  st.set_page_config(
    page_title="Sinə Xərçəngi Proqnozu",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  
  input_data = add_sidebar()
  
  with st.container():
    st.header("Sinə Xərçəngi Proqnozu")
    st.write("Modelin proqnoz nəticəlirini sitologiya laboratoriyası ilə birgə dəyərləndirilməsi, toxuma nümunənisinin sinə xərçəngini diaqnoz etməyə kömək edə bilər. Bu tətbiq maşın öyrənmə modelindən istifadə edərək, sitologiya laboratoriyasından alınan ölçülərə əsasən bir kütlənin xoşxassəli və ya bədxassəli olub-olmadığını proqnozlaşdırır. Soldakı menyudakı slayderlərdən istifadə edərək ölçüləri əl ilə də yeniləyə bilərsiniz.")

    with st.expander("Model Dəyərləndirmə Ölçüləri", expanded=False):
        st.markdown("""
            | Dəyər     | Xoşxassəli  | Bədxassəli  | Ümumi |
            |------------|---------|---------|---------|
            | Precision  | 0.97    | 0.98    |         |
            | Recall     | 0.99    | 0.95    |         |
            | F1-score   | 0.98    | 0.96    |         |
            | Support    | 71      | 43      | 114     |
        \n\n""")
        
        st.markdown("""
            **Xülasə:**
            - Accuracy: 0.97 (97%)
            - Macro Avg: Precision 0.97, Recall 0.97, F1-score 0.97
            - Weighted Avg: Precision 0.97, Recall 0.97, F1-score 0.97
        """)
        
        st.text("Model qiymətləndirməsi proqnozların etibarlılığını anlamaq üçün təqdim edilir. \nKəsrlər 100 ilə vurularaq faiz olaraq dəyərləndirilə bilər.")
  col1, col2 = st.columns([4,1])
  
  with col1:
    radar_chart = get_radar_chart(input_data)
    st.plotly_chart(radar_chart)
  with col2:
    add_predictions(input_data)

  css_path = Path(__file__).parents[0]
  with open(f"{css_path}/css/style.css") as f:
       st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
 
if __name__ == '__main__':
  main()
