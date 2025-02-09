import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# Configuração da página Streamlit
st.set_page_config(page_title="Estratificação de Riscos para Pacientes em Pronto Atendimento", layout="wide")

# =============================================
# Seção 1: Cabeçalho com Logos e Bandeiras
# =============================================

# Carregar logos e bandeiras
logo_uff = Image.open("logouff_vertical_fundo_azul-1.png")  
logo_ps = Image.open("prevent-senior.png") 
bandeira_br = Image.open("bandeira_br.png")  
bandeira_uk = Image.open("bandeira_uk.png") 
bandeira_it = Image.open("bandeira_it.png") 

# Layout do cabeçalho
col1, col2, col3 = st.columns([2, 3, 2])
with col1:
    st.image([logo_uff, logo_ps], width=100)
with col2:
    st.title("Estratificação de Riscos para Pacientes em Pronto Atendimento")
with col3:
    st.write("Selecione o idioma:")
    if st.button("🇧🇷"):
        st.session_state.idioma = "pt"
    if st.button("🇬🇧"):
        st.session_state.idioma = "en"
    if st.button("🇮🇹"):
        st.session_state.idioma = "it"

# Definir idioma padrão
if "idioma" not in st.session_state:
    st.session_state.idioma = "pt"

# Textos traduzidos
textos = {
    "pt": {
        "titulo": "Estratificação de Riscos para Pacientes em Pronto Atendimento",
        "adicionar_paciente": "Adicionar Paciente",
        "resultados": "Resultados da Priorização",
        "grafico": "Gráfico de Dispersão",
        "dataframe_pfn": "DataFrame PFN",
        "dataframe_distancias": "DataFrame Distâncias",
        "dataframe_final": "DataFrame Final",
        "analise_resultados": "Análise dos Resultados",
    },
    "en": {
        "titulo": "Risk Stratification for Emergency Patients",
        "adicionar_paciente": "Add Patient",
        "resultados": "Prioritization Results",
        "grafico": "Scatter Plot",
        "dataframe_pfn": "PFN DataFrame",
        "dataframe_distancias": "Distances DataFrame",
        "dataframe_final": "Final DataFrame",
        "analise_resultados": "Results Analysis",
    },
    "it": {
        "titulo": "Stratificazione del Rischio per Pazienti in Pronto Soccorso",
        "adicionar_paciente": "Aggiungi Paziente",
        "resultados": "Risultati della Priorizzazione",
        "grafico": "Grafico a Dispersione",
        "dataframe_pfn": "DataFrame PFN",
        "dataframe_distancias": "DataFrame Distanze",
        "dataframe_final": "DataFrame Finale",
        "analise_resultados": "Analisi dei Risultati",
    }
}

# =============================================
# Seção 2: Definição das Variáveis do Problema
# =============================================

# Criando as variáveis do problema
SNC = ctrl.Antecedent(np.arange(1, 5.1, 0.1), 'Neuroatividade')
FR = ctrl.Antecedent(np.arange(0, 51, 1), 'Frequencia Respiratoria')
SatO2 = ctrl.Antecedent(np.arange(0, 101, 1), 'Saturacao de Oxigenio')
FC = ctrl.Antecedent(np.arange(0, 301, 1), 'Frequencia Cardiaca')
PAS = ctrl.Antecedent(np.arange(0, 251, 1), 'Pressao Arterial Sistolica')
PAD = ctrl.Antecedent(np.arange(0, 131, 1), 'Pressao Arterial Diastolica')
TC = ctrl.Antecedent(np.arange(32, 44, 1), 'Temperatura Corporal')
EG = ctrl.Consequent(np.arange(1, 6, 1), 'Estado Geral')

# =============================================
# Seção 3: Definição das Funções de Pertinência
# =============================================

# Neuroatividade
SNC['Inconsciente'] = fuzz.gaussmf(SNC.universe, 1, 0.1)
SNC['Responde a Dor'] = fuzz.gaussmf(SNC.universe, 2, 0.1)
SNC['Responde a Voz'] = fuzz.gaussmf(SNC.universe, 3, 0.1)
SNC['Alerta'] = fuzz.gaussmf(SNC.universe, 4, 0.1)
SNC['Hiperalerta'] = fuzz.gaussmf(SNC.universe, 5, 0.1)

# Frequencia Respiratoria
FR['Muito Baixa'] = fuzz.gaussmf(FR.universe, 4, 4)
FR['Baixa'] = fuzz.gaussmf(FR.universe, 10, 2)
FR['Normal'] = fuzz.gaussmf(FR.universe, 16, 4)
FR['Alta'] = fuzz.gaussmf(FR.universe, 24, 4)
FR['Muito Alta'] = fuzz.gaussmf(FR.universe, 30, 4)

# Saturacao de Oxigenio
SatO2['Muito Baixa'] = fuzz.gaussmf(SatO2.universe, 60, 10)
SatO2['Baixa'] = fuzz.gaussmf(SatO2.universe, 80, 10)
SatO2['Normal'] = fuzz.gaussmf(SatO2.universe, 93, 3)
SatO2['Alta'] = fuzz.gaussmf(SatO2.universe, 96, 1)
SatO2['Muito Alta'] = fuzz.gaussmf(SatO2.universe, 99, 1)

# Frequencia Cardiaca
FC['Muito Baixa'] = fuzz.gaussmf(FC.universe, 20, 20)
FC['Baixa'] = fuzz.gaussmf(FC.universe, 40, 20)
FC['Normal'] = fuzz.gaussmf(FC.universe, 80, 20)
FC['Alta'] = fuzz.gaussmf(FC.universe, 120, 20)
FC['Muito Alta'] = fuzz.gaussmf(FC.universe, 180, 40)

# Pressao Arterial Sistolica
PAS['Muito Baixa'] = fuzz.gaussmf(PAS.universe, 60, 10)
PAS['Baixa'] = fuzz.gaussmf(PAS.universe, 80, 10)
PAS['Normal'] = fuzz.gaussmf(PAS.universe, 105, 15)
PAS['Alta'] = fuzz.gaussmf(PAS.universe, 140, 20)
PAS['Muito Alta'] = fuzz.gaussmf(PAS.universe, 200, 40)

# Pressao Arterial Diastolica
PAD['Muito Baixa'] = fuzz.gaussmf(PAD.universe, 30, 10)
PAD['Baixa'] = fuzz.gaussmf(PAD.universe, 50, 10)
PAD['Normal'] = fuzz.gaussmf(PAD.universe, 70, 10)
PAD['Alta'] = fuzz.gaussmf(PAD.universe, 90, 10)
PAD['Muito Alta'] = fuzz.gaussmf(PAD.universe, 120, 20)

# Temperatura Corporal
TC['Muito Baixa'] = fuzz.gaussmf(TC.universe, 33, 1)
TC['Baixa'] = fuzz.gaussmf(TC.universe, 35, 1)
TC['Normal'] = fuzz.gaussmf(TC.universe, 37, 1)
TC['Alta'] = fuzz.gaussmf(TC.universe, 39, 1)
TC['Muito Alta'] = fuzz.gaussmf(TC.universe, 41, 1)

# =============================================
# Seção 4: Função para Cálculo do PFN
# =============================================

def calcular_pfn_automatico(valor, variavel_fuzzy):
    pertinencias = {termo: fuzz.interp_membership(variavel_fuzzy.universe, variavel_fuzzy[termo].mf, valor)
                    for termo in variavel_fuzzy.terms}
    positivo = max(pertinencias.get('Normal', 0.0), pertinencias.get('Alerta', 0.0))
    negativo = max(pertinencias.get('Muito Baixa', 0.0), pertinencias.get('Muito Alta', 0.0),
                   pertinencias.get('Inconsciente', 0.0), pertinencias.get('Responde a Dor', 0.0))
    neutro = max(pertinencias.get('Baixa', 0.0), pertinencias.get('Alta', 0.0),
                 pertinencias.get('Responde a Voz', 0.0))
    soma = positivo + negativo + neutro
    if soma > 0:
        positivo /= soma
        negativo /= soma
        neutro /= soma
    return round(positivo, 4), round(negativo, 4), round(neutro, 4)

# =============================================
# Seção 5: Interface Streamlit
# =============================================

# Armazenamento dos pacientes na sessão do Streamlit
if 'pacientes' not in st.session_state:
    st.session_state.pacientes = []

# Entrada de dados dos pacientes
st.sidebar.header(textos[st.session_state.idioma]["adicionar_paciente"])

with st.sidebar.form("paciente_form"):
    st.write("Insira os dados do paciente:")
    neuro = st.slider("Neuroatividade (1-5)", 1.0, 5.0, 3.0)
    fr = st.slider("Frequência Respiratória (0-50)", 0, 50, 20)
    sat = st.slider("Saturação de Oxigênio (0-100)", 0, 100, 95)
    fc = st.slider("Frequência Cardíaca (0-300)", 0, 300, 80)
    pas = st.slider("Pressão Arterial Sistólica (0-250)", 0, 250, 120)
    pad = st.slider("Pressão Arterial Diastólica (0-130)", 0, 130, 80)
    temp = st.slider("Temperatura Corporal (32-43)", 32, 43, 37)
    submitted = st.form_submit_button(textos[st.session_state.idioma]["adicionar_paciente"])
    if submitted:
        paciente = {
            "Neuroatividade": neuro,
            "Frequência Respiratória": fr,
            "Saturação de Oxigênio": sat,
            "Frequência Cardíaca": fc,
            "Pressão Arterial Sistólica": pas,
            "Pressão Arterial Diastólica": pad,
            "Temperatura Corporal": temp
        }
        st.session_state.pacientes.append(paciente)
        st.success("Paciente adicionado!")

# =============================================
# Seção 6: Processamento e Exibição de Resultados
# =============================================

if st.session_state.pacientes:
    # Cálculo dos PFNs
    matrizes_pacientes = []
    for idx, paciente in enumerate(st.session_state.pacientes):
        valores = list(paciente.values())
        MPF = np.array([calcular_pfn_automatico(valor, variavel) for valor, variavel in zip(valores, [SNC, FR, SatO2, FC, PAS, PAD, TC])])
        matrizes_pacientes.append(MPF)

    # Aplicação do TOPSIS
    pesos = np.array([0.3384, 0.2413, 0.0123, 0.2057, 0.0902, 0.1054, 0.0087])
    df_pacientes = pd.DataFrame(columns=["ID", "Positive", "Negative", "Neutral"])

    for idx, MPF in enumerate(matrizes_pacientes, start=1):
        weighted_matrix = np.round(MPF * pesos[:, np.newaxis], 4)
        vetor_agregado = np.round(weighted_matrix.sum(axis=0), 4)
        df_pacientes.loc[idx - 1] = [f"P{idx}", vetor_agregado[0], vetor_agregado[1], vetor_agregado[2]]

    # Cálculo do TOPSIS
    PFS_PIS = (df_pacientes["Positive"].max(), df_pacientes["Negative"].min(), df_pacientes["Neutral"].min())
    PFS_NIS = (df_pacientes["Positive"].min(), df_pacientes["Negative"].max(), df_pacientes["Neutral"].max())

    df_pacientes["D+"] = np.sqrt((df_pacientes["Positive"] - PFS_PIS[0])**2 +
                                 (df_pacientes["Negative"] - PFS_PIS[1])**2 +
                                 (df_pacientes["Neutral"] - PFS_PIS[2])**2)

    df_pacientes["D-"] = np.sqrt((df_pacientes["Positive"] - PFS_NIS[0])**2 +
                                 (df_pacientes["Negative"] - PFS_NIS[1])**2 +
                                 (df_pacientes["Neutral"] - PFS_NIS[2])**2)

    df_pacientes["Csi"] = df_pacientes["D-"] / (df_pacientes["D+"] + df_pacientes["D-"])
    df_pacientes["Ranking"] = df_pacientes["Csi"].rank(ascending=True, method="dense")

    # Análise dos resultados
    df_pacientes["Categoria_Positive"] = ""
    df_pacientes["Categoria_Negative"] = ""
    df_pacientes["Categoria_Neutral"] = ""

    for index, row in df_pacientes.iterrows():
        # Classificação do valor positivo (μ)
        estado_geral_positivo = row["Positive"]
        if 0.0 <= estado_geral_positivo < 0.2:
            categoria_positivo = "muito crítico"
        elif 0.2 <= estado_geral_positivo < 0.4:
            categoria_positivo = "crítico"
        elif 0.4 <= estado_geral_positivo < 0.6:
            categoria_positivo = "instável"
        elif 0.6 <= estado_geral_positivo < 0.8:
            categoria_positivo = "próximo dos limites normais"
        elif 0.8 <= estado_geral_positivo <= 1:
            categoria_positivo = "normal"
        else:
            categoria_positivo = "fora da escala válida"

        # Classificação do valor negativo (ν)
        estado_geral_negativo = row["Negative"]
        if 0.0 <= estado_geral_negativo < 0.2:
            categoria_negativo = "risco muito baixo"
        elif 0.2 <= estado_geral_negativo < 0.4:
            categoria_negativo = "risco baixo"
        elif 0.4 <= estado_geral_negativo < 0.6:
            categoria_negativo = "risco moderado"
        elif 0.6 <= estado_geral_negativo < 0.8:
            categoria_negativo = "risco alto"
        elif 0.8 <= estado_geral_negativo <= 1:
            categoria_negativo = "risco muito alto"
        else:
            categoria_negativo = "fora da escala válida"

        # Classificação do valor neutro (η)
        estado_geral_neutro = row["Neutral"]
        if 0.0 <= estado_geral_neutro < 0.2:
            categoria_neutro = "muito pouca incerteza"
        elif 0.2 <= estado_geral_neutro < 0.4:
            categoria_neutro = "pouca incerteza"
        elif 0.4 <= estado_geral_neutro < 0.6:
            categoria_neutro = "incerteza moderada"
        elif 0.6 <= estado_geral_neutro < 0.8:
            categoria_neutro = "considerável incerteza"
        elif 0.8 <= estado_geral_neutro <= 1:
            categoria_neutro = "muita incerteza"
        else:
            categoria_neutro = "fora da escala válida"

        # Atualizando as categorias no DataFrame
        df_pacientes.at[index, "Categoria_Positive"] = categoria_positivo
        df_pacientes.at[index, "Categoria_Negative"] = categoria_negativo
        df_pacientes.at[index, "Categoria_Neutral"] = categoria_neutro

    # Renomeando as colunas
    df_pacientes.rename(columns={
        "Categoria_Positive": "Criticidade",
        "Categoria_Negative": "Risco",
        "Categoria_Neutral": "Incerteza"
    }, inplace=True)

    # Abas para visualização
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        textos[st.session_state.idioma]["grafico"],
        textos[st.session_state.idioma]["dataframe_pfn"],
        textos[st.session_state.idioma]["dataframe_distancias"],
        textos[st.session_state.idioma]["dataframe_final"],
        textos[st.session_state.idioma]["analise_resultados"]
    ])

    with tab1:
        # Gráfico de dispersão
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df_pacientes["D+"], df_pacientes["D-"], c=df_pacientes["Csi"], cmap='RdYlGn', s=100, edgecolor='black')
        for i, txt in enumerate(df_pacientes["ID"]):
            ax.annotate(txt, (df_pacientes["D+"][i], df_pacientes["D-"][i]), textcoords="offset points", xytext=(10, -10), fontsize=8, ha='center')
        ax.set_xlabel("Distância até SIP (D+)")
        ax.set_ylabel("Distância até SIN (D-)")
        ax.set_title("Estado Geral Relativo dos Pacientes")
        plt.colorbar(scatter, label="Coeficiente de Proximidade Relativa (ξ)")
        st.pyplot(fig)

    with tab2:
        st.subheader(textos[st.session_state.idioma]["dataframe_pfn"])
        st.dataframe(df_pacientes[["ID", "Positive", "Negative", "Neutral"]])

    with tab3:
        st.subheader(textos[st.session_state.idioma]["dataframe_distancias"])
        st.dataframe(df_pacientes[["ID", "D+", "D-"]])

    with tab4:
        st.subheader(textos[st.session_state.idioma]["dataframe_final"])
        st.dataframe(df_pacientes[["ID", "Csi", "Ranking", "Criticidade", "Risco", "Incerteza"]])

    with tab5:
        st.subheader(textos[st.session_state.idioma]["analise_resultados"])
        for index, row in df_pacientes.iterrows():
            st.write(f"**Paciente {row['ID']}:**")
            st.write(f"- Criticidade: {row['Criticidade']}")
            st.write(f"- Risco: {row['Risco']}")
            st.write(f"- Incerteza: {row['Incerteza']}")
            st.write("---")

else:
    st.info("Adicione pacientes para ver os resultados.")

# =============================================
# Seção 7: Rodapé
# =============================================

st.markdown("---")
st.markdown("""
**Todos os direitos reservados.**  
O uso não comercial (acadêmico) deste software é gratuito.  
A única coisa que se pede em troca é citar este software quando os resultados são usados em publicações.  
Para citar o software: SILVA, Antonio Sergio da; SANTOS, Marcos dos; GOMES, Carlos Francisco Simões;  
EMERO PSF TOPSIS Software Web (v.1). 2025.  

""")
