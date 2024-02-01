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
    menu = ["Home", "Soft K - Capacitação 4.0", "Off-Topics"]
    escolha = st.sidebar.selectbox("Menu", menu)

    if escolha == "Home":
        st.header("Junior Data Scientist")
        st.write('Desejo apresentar formalmente os projetos que elaborei ao longo do meu período de graduação em Inteligência Artificial na Universidade Federal de Goiás (UFG). Por meio deste aplicativo Streamlit, compartilho a interação com os modelos que fui responsável por treinar, bem como alguns textos que foram desenvolvidos ao longo da minha trajetória acadêmica.')
        st.write('Me chamo Lisandra Menezes e  você pode visitar meu github por esse link (https://github.com/LisandraMoura).')
        st.write('Uma forma de comunicação possível comigo é através do LinkedIn (https://www.linkedin.com/in/lisandra-menezes-9320a5209/)')
        
    elif escolha == "Soft K - Capacitação 4.0":
        st.subheader("Projetos desenvolvidos: Soft K - Capacitação 4.0")
        
        st.write("Durante a Capacitação 4.0, concebemos um projeto voltado à computação da personalidade, cujo propósito reside na utilização de técnicas computacionais para identificar os traços de personalidade de um indivíduo. Este método de predição revela-se benéfico em contextos como entrevistas de emprego ou outras análises que requerem a percepção de nuances no comportamento humano. Os modelos disponibilizados nesta plataforma constituem o produto inicial de nossa contribuição à capacitação, resultando de um único ciclo de treinamento e ausência de otimizações.")
        st.write("Selecione entre o algoritmo Random Forest aplicado ao conjunto de dados 'Essays' ou ao conjunto de dados 'MyPersonality'. Para cada opção, forneça, respectivamente, um ensaio autobiográfico ou uma breve frase adequada para publicação no Twitter.")

        modelo_ia = st.selectbox("Escolha uma das opções", ["Random Forest Essays", "Random Forest MyPersonality"])
        

        if modelo_ia == "Random Forest Essays":
            
            st.write("Exemplo para predição:")
            st.write("Well, right now I just woke up from a mid-day nap. It's sort of weird, but ever since I moved to Texas, I have had problems concentrating on things. I remember starting my homework in  10th grade as soon as the clock struck 4 and not stopping until it was done. Of course it was easier, but I still did it. But when I moved here, the homework got a little more challenging and there was a lot more busy work, and so I decided not to spend hours doing it, and just getting by. But the thing was that I always paid attention in class and just plain out knew the stuff, and now that I look back, if I had really worked hard and stayed on track the last two years without getting  lazy, I would have been a genius, but hey, that's all good. ")

            # Carregar o SentenceTransformer model
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            loaded_model = load_model(r"random_forest_model.pkl")
                
            # Entrada de texto do usuário
            user_text = st.text_area("Por favor, insira o texto (em inglês):")

            if st.button("Fazer Predição"):
                prediction = make_prediction(user_text, loaded_model, model)
                
                columns = ["Extroversão", "Neuroticismo", "Simpatia", "Conscienciosidade ", "Abertura ao novo"]
                df = pd.DataFrame([prediction], columns=columns)
                
                st.write("Resultado da Predição:")
                st.dataframe(df)
                
        elif modelo_ia == "Random Forest MyPersonality":
            
            st.write("Exemplo para predição:")
            st.write("is so sleepy it's not even funny that's she can't get to sleep.")
            
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            loaded_model = load_model(r"mypersonality_forest.pkl")
            
            user_text = st.text_area("Por favor, insira o texto (em inglês):")

            if st.button("Fazer Predição"):
                prediction = make_prediction(user_text, loaded_model, model)
                
                columns = ["Extroversão", "Neuroticismo", "Simpatia", "Conscienciosidade ", "Abertura ao novo"]
                df = pd.DataFrame([prediction], columns=columns)
                
                st.write("Resultado da Predição:")
                st.dataframe(df)
            
    # Criação de um outro tópico no menu principal
#    elif escolha == "Off-Topics":
#        st.subheader("Blog Pessoal")
#        texto = st.selectbox("Escolha uma das opções", ["Artigo sobre Nietszche", "Cloangem de voz", "X"])
#        
#        if texto == "Artigo sobre Nietszche":
#            st.write("O texto será publicado no Medium")
#        elif texto == "Cloangem de voz":
#            st.write("O texto será publicado no Medium")
#        elif texto == "Continua":
#            st.write("O texto será publicado no Medium")
        
        
        

# Executar a aplicação
if __name__ == '__main__':
    main()
