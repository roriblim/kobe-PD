import streamlit as st
import pickle
import numpy as np
import pandas as pd

@st.cache_resource
def load_model():
    with open('../data/06_models/DT_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict_local(data):
    model = load_model()
    input_df = pd.DataFrame([data])  # Usa os nomes das colunas!
    probs = model.predict_proba(input_df)
    return probs.tolist()

st.markdown("""
# Kobe - Predição de acertos das cestas - alternativo  
### Esta página permite ao usuário prever se o Kobe irá acertar uma cesta ou não, utilizando o modelo de <span style="color:red">Árvore de Decisão</span>, sem API.            
""", unsafe_allow_html=True)

opcoes_shot_type = ['2PT Field Goal', '3PT Field Goal']
opcoes_combined = ['Jump Shot', 'Layup', 'Dunk', 'Tip Shot', 'Hook Shot', 'Bank Shot']
opcoes_area = ['Right Side(R)', 'Left Side Center(LC)', 'Left Side(L)', 'Right Side Center(RC)', 'Center(C)', 'Back Court(BC)']
opcoes_range = ['16-24 ft.', '8-16 ft.', 'Less Than 8 ft.', '24+ ft.', 'Back Court Shot']


col1, col2, col3 = st.columns(3)

with col1:
    lat = st.number_input('lat', value=33.8183)
    loc_x = st.number_input('loc_x', value=-117)
    minutes_remaining = st.number_input('minutes_remaining', value=8)
    shot_type = st.selectbox('shot_type', opcoes_shot_type)

with col2:
    lon = st.number_input('lon', value=-118.3868)
    loc_y = st.number_input('loc_y', value=226)
    period = st.number_input('period', value=2)
    shot_zone_area = st.selectbox('shot_zone_area', opcoes_area)

with col3:
    combined_shot_type = st.selectbox('combined_shot_type', opcoes_combined)
    seconds_remaining = st.number_input('seconds_remaining', value=50)
    playoffs = int(st.checkbox('playoffs'))
    shot_distance = st.number_input('shot_distance', value=25)
    shot_zone_range = st.selectbox('shot_zone_range', opcoes_range)

input_data = {
    'lat': lat,
    'lon': lon,
    'minutes_remaining': minutes_remaining,
    'period': period,
    'playoffs': playoffs,
    'shot_distance': shot_distance,
    'combined_shot_type': combined_shot_type,
    'loc_x': loc_x,
    'loc_y': loc_y,
    'seconds_remaining': seconds_remaining,
    'shot_type': shot_type,
    'shot_zone_area': shot_zone_area,
    'shot_zone_range': shot_zone_range
}

colunas_necessarias = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']
data_processed = {col: input_data[col] for col in colunas_necessarias}
data_processed['lat_quadra'] = data_processed['lat'] - 34.0443
data_processed['lon_quadra'] = data_processed['lon'] + 118.2698
del data_processed['lat']
del data_processed['lon']

st.markdown("### Dados processados:")
st.json(list(data_processed.values()))

st.markdown("### Resultados do Predict:")
resposta = predict_local(data_processed)

pred_cesta_resposta_erro = resposta[0][0]
pred_cesta_resposta_acerto = resposta[0][1]
pred_cesta = 0 if pred_cesta_resposta_erro > pred_cesta_resposta_acerto else 1

st.write(f'Probabilidade de acerto: {pred_cesta_resposta_acerto}')
st.write(f'Probabilidade de erro: {pred_cesta_resposta_erro}')

if pred_cesta == 1:
    st.success("🏀 Acertou a cesta!")
else:
    st.error("💔 Errou a cesta!")
