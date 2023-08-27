# portfolio_app.py

# Importando as bibliotecas necessárias
import streamlit as st

# Funções para cada projeto
def projeto1():
    st.write("## Projeto 1: Análise de Vendas")
    st.write("Aqui você pode adicionar visualizações, informações e análises sobre a análise de vendas.")
    # exemplo: st.line_chart(dataframe_do_seu_projeto)

def projeto2():
    st.write("## Projeto 2: Análise de Redes Sociais")
    st.write("Aqui você pode adicionar visualizações, informações e análises sobre a análise de redes sociais.")
    # exemplo: st.bar_chart(dataframe_do_seu_outro_projeto)

# Layout do aplicativo
st.title('Portfólio de Análise de Dados')
st.write('Bem-vindo ao meu portfólio. Escolha um projeto abaixo para visualizar.')
st.write('Me chamo Lisandra Menezes, sou estudade de Inteligência Artificial na Universidade Federal de Goiás.')
st.write('Visualize também meu GitHub (https://github.com/LisandraMoura).')

# Menu lateral para seleção do projeto
opcao = st.sidebar.selectbox("Escolha o projeto:", ["", "Projeto 1: Análise de Sentimento", "Projeto 2: Bares de Goiânia - Análise de Sentimento"])

if opcao == "Projeto 1: Análise de Sentimento":
    projeto1()
elif opcao == "Projeto 2: Bares de Goiânia, melhores e piores de acordo com as avaliações dos clientes":
    projeto2()
