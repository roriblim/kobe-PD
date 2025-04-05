import streamlit as st
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def call_api(data):

    rows = [
        list(data.values())
    ]
    
    resp = requests.post(
        'http://localhost:5002/invocations',
        json={
            'inputs': rows
        },
        
        headers={"Content-Type": "application/json"}
    )

    return resp.json()

def plot_distribuicao_e_dado_novo(df_treino, novo_dado):
    cols = df_treino.columns
    num_features = len(cols)
    cols_plot = 3
    rows_plot = (num_features // cols_plot) + int(num_features % cols_plot > 0)

    fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(15, 5 * rows_plot))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.violinplot(y=df_treino[col], ax=axes[i], inner=None, color="skyblue")
        axes[i].axhline(novo_dado[col], color='red', linestyle='--', label='Valor novo')
        axes[i].set_title(col)
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    st.pyplot(fig)


st.markdown("""
# Kobe - Predição de acertos das cestas
Essa aplicação permite ao usuário prever se o Kobe irá acertar uma cesta ou não.            
""")


opcoes = ['Jump Shot', 'Layup', 'Dunk', 'Tip Shot', 'Hook Shot', 'Bank Shot']
combined_shot_type = st.selectbox('combined_shot_type', opcoes)

input_data = {
    'lat' : st.number_input('lat', value=33.81),
    'lon' : st.number_input('lon', value=-118.3638),
    'loc_x' : st.number_input('loc_x', value=-94),
    'loc_y' : st.number_input('loc_y', value=238),
    'minutes_remaining' : st.number_input('minutes_remaining', value=1),
    'period' : st.number_input('period', value=2),
    'playoffs' : int(st.checkbox('playoffs')),
    'seconds_remaining' : st.number_input('seconds_remaining', value=8),
    'shot_distance' : st.number_input('shot_distance', value=25),
    'shot_type' : st.selectbox('shot_type', ['2PT Field Goal','3PT Field Goal']),
    'shot_zone_area' : st.selectbox('shot_zone_area', ['Right Side(R)', 'Left Side Center(LC)', 'Left Side(L)', 'Right Side Center(RC)', 'Center(C)','Back Court(BC)']),
    'shot_zone_range' : st.selectbox('shot_zone_range', ['16-24 ft.', '8-16 ft.', 'Less Than 8 ft.', '24+ ft.','Back Court Shot'])
}

colunas_necessarias = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs','shot_distance']


data_processed = {coluna: input_data[coluna] for coluna in colunas_necessarias}

data_processed['lat_quadra'] = data_processed['lat'] - 34.0443
data_processed['lon_quadra'] = data_processed['lon'] + 118.2698

del data_processed['lat']
del data_processed['lon']

st.markdown("""
### Dados processados que entrarão na API:       
""")

st.json(list(data_processed.values()))


st.markdown("""
### Resultados da consulta à API:       
""")
resposta = call_api(data_processed)
pred_cesta_resposta = resposta['predictions'][0]
pred_cesta_resposta_erro = pred_cesta_resposta[0]
pred_cesta_resposta_acerto = pred_cesta_resposta[1]

pred_cesta = 0 if pred_cesta_resposta_erro > pred_cesta_resposta_acerto else 1
is_acertou_msg = "Acertou a cesta!!!" if pred_cesta == 1 else "Errou a cesta! :()"

st.write(f'Probabilidade de acerto: {pred_cesta_resposta_acerto}')
st.write(f'Probabilidade de erro: {pred_cesta_resposta_erro}')
st.write(f'Previsão final: {is_acertou_msg}')

df_treino = pd.read_csv("../data/05_model_input/base_train.csv")
df_treino_x = df_treino.drop('shot_made_flag', axis=1)

st.markdown("### Comparação - dados de treino e dado de produção enviado para a API")
plot_distribuicao_e_dado_novo(df_treino_x, data_processed)