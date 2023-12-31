import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Função para carregar o modelo
def load_model(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)

# Função para fazer a predição
def make_prediction(user_text, selected_model, model):
    import nltk
    nltk.download('punkt')

    # Processar o texto
    sentences = nltk.sent_tokenize(user_text)
    embeddings = [model.encode(sentence) for sentence in sentences]
    essay_embedding = sum(embeddings) / len(embeddings)
    
    # Fazer a predição
    prediction = selected_model.predict([essay_embedding])[0]
    
    # Converter 0s para "Não" e 1s para "Sim"
    return ['Sim' if pred == 1 else 'Não' for pred in prediction]

# Função principal da aplicação Streamlit
def main():
    # Menu lateral para navegação
    menu = ["Home", "Projetos"]
    escolha = st.sidebar.selectbox("Menu", menu)

    if escolha == "Home":
        st.header("Bem-vindo ao meu portfólio. ")
        st.write('Aqui você pode explorar diversos projetos que desenvolvi ao longo da minha faculdade em Inteligência Artificial.')
        st.write('Me chamo Lisandra Menezes e  você pode visitar meu github por esse link (https://github.com/LisandraMoura).')

    elif escolha == "Projetos":
        st.subheader("Projetos")
        modelo_ia = st.selectbox("Escolha uma das opções", ["Soft K - Capacitação 4.0", "Outros"])

        if modelo_ia == "Soft K - Capacitação 4.0":
            st.write("Na capacitação 4.0 desenvolvemos um projeto de computação da personalidade, isto é, nosso objetivo era atraves de técnicas computacionais determinar quais são os traços de personalidade de uma pessoa. Esse tipo de predição pode ser útil em entrevistas de emprego, ou outras análises que envolva perceber sutilezadas no comportamento humano. Os modelos disponíveis aqui são fruto da nossa primeira entrega na capacitação, com apenas 1 treinamento e sem otimização.")
            st.write("Escolha entre o Randon Forest no dataset Essays ou Randon Forest no dataset MyPersonality, respectivamente, sua entrada deve ser um ensaios sobre você ou uma frase curta que você postaria no twitter ( ou X)")
            
            st.write("Ex:")
            st.write("Essays: Well, right now I just woke up from a mid-day nap. It's sort of weird, but ever since I moved to Texas, I have had problems concentrating on things. I remember starting my homework in  10th grade as soon as the clock struck 4 and not stopping until it was done. Of course it was easier, but I still did it. But when I moved here, the homework got a little more challenging and there was a lot more busy work, and so I decided not to spend hours doing it, and just getting by. But the thing was that I always paid attention in class and just plain out knew the stuff, and now that I look back, if I had really worked hard and stayed on track the last two years without getting  lazy, I would have been a genius, but hey, that's all good. ")
            st.write("MyPersonality: is so sleepy it's not even funny that's she can't get to sleep.")
            # Carregar o SentenceTransformer model
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

            # Seleção do modelo de IA
            model_option = st.selectbox("Selecione o modelo de IA:", ["Randon Forest em Ensaios", "Randon Forest em textos curtos"])

            if model_option == "Randon Forest em Ensaios":
                loaded_model = load_model(r"random_forest_model.pkl")
            elif model_option == "Randon Forest em textos curtos":
                loaded_model = load_model(r"mypersonality_forest.pkl")
            # Entrada de texto do usuário
            user_text = st.text_area("Por favor, insira o texto (em inglês):")

            if st.button("Fazer Predição"):
                prediction = make_prediction(user_text, loaded_model, model)
                
                columns = ["Extroversão", "Neuroticismo", "Simpatia", "Conscienciosidade ", "Abertura ao novo"]
                df = pd.DataFrame([prediction], columns=columns)
                
                st.write("Resultado da Predição:")
                st.dataframe(df)

# Executar a aplicação
if __name__ == '__main__':
    main()
