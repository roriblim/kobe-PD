import streamlit as st
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def call_api(data):
    rows = [list(data.values())]
    resp = requests.post(
        'http://localhost:5002/invocations',
        json={'inputs': rows},
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


def salvar_dado_producao(novo_dado):
    df_novo = pd.DataFrame([novo_dado])
    path = "../data/07_model_output/inferencia_streamlit.csv"
    try:
        df_existente = pd.read_csv(path)
        df_novo = pd.concat([df_existente, df_novo], ignore_index=True)
    except FileNotFoundError:
        pass
    df_novo.to_csv(path, index=False)


def plota_data_drift(df_treino, df_producao):
    cols = df_treino.columns
    num_features = len(cols)
    cols_plot = 3
    rows_plot = (num_features // cols_plot) + int(num_features % cols_plot > 0)

    fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(15, 5 * rows_plot))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.violinplot(y=df_treino[col], ax=axes[i], inner=None, color="skyblue")
        sns.stripplot(y=df_producao[col], ax=axes[i], color='red', alpha=0.6, jitter=0.1, size=4, label='Produção')
        axes[i].set_title(col)
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    st.pyplot(fig)


df_treino = pd.read_csv("../data/05_model_input/base_train.csv")
df_treino_x = df_treino.drop('shot_made_flag', axis=1)
aba = st.sidebar.radio("Navegação", ["Predição", "Monitoramento"])

if aba == "Predição":
    st.markdown("""
    # Kobe - Predição de acertos das cestas
    ### Essa aplicação permite ao usuário prever se o Kobe irá acertar uma cesta ou não, utilizando o modelo de <span style="color:red">Regressão Logística</span> por meio do consumo à <span style="color:red">API</span>.            
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        lat = st.number_input('lat', value=33.81)
        lon = st.number_input('lon', value=-118.3638)

    with col2: 
        minutes_remaining = st.number_input('minutes_remaining', value=1)
        period = st.number_input('period', value=2)

    with col3:
        playoffs = int(st.checkbox('playoffs'))
        shot_distance = st.number_input('shot_distance', value=20)

    input_data = {
        'lat': lat,
        'lon': lon,
        'minutes_remaining': minutes_remaining,
        'period': period,
        'playoffs': playoffs,
        'shot_distance': shot_distance
    }

    colunas_necessarias = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs','shot_distance']

    data_processed = {coluna: input_data[coluna] for coluna in colunas_necessarias}
    data_processed['lat_quadra'] = data_processed['lat'] - 34.0443
    data_processed['lon_quadra'] = data_processed['lon'] + 118.2698
    del data_processed['lat']
    del data_processed['lon']

    st.markdown("### Dados processados que entrarão na API:")
    st.json(list(data_processed.values()))

    st.markdown("### Resultados da consulta à API:")
    resposta = call_api(data_processed)
    salvar_dado_producao(data_processed)

    pred_cesta_resposta = resposta['predictions'][0]
    pred_cesta_resposta_erro = pred_cesta_resposta[0]
    pred_cesta_resposta_acerto = pred_cesta_resposta[1]

    pred_cesta = 0 if pred_cesta_resposta_erro > pred_cesta_resposta_acerto else 1

    st.write(f'Probabilidade de acerto: {pred_cesta_resposta_acerto}')
    st.write(f'Probabilidade de erro: {pred_cesta_resposta_erro}')

    if pred_cesta == 1:
        st.success("🏀 Acertou a cesta!")
    else:
        st.error("💔 Errou a cesta!")

    st.markdown("### Comparação - dados de treino e dado de produção enviado para a API")
    plot_distribuicao_e_dado_novo(df_treino_x, data_processed)

elif aba == "Monitoramento":
    st.markdown("### Análise de Data Drift da API")
    
    with st.spinner('Carregando análise de Data Drift...'):
        df_producao = pd.read_csv("../data/07_model_output/inferencia_streamlit.csv")
        plota_data_drift(df_treino_x, df_producao)

    st.success("Análise carregada com sucesso!")

