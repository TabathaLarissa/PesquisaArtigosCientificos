import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
nltk.download('stopwords')
# !pip install bs4
from bs4 import BeautifulSoup
#import transformers
from transformers import pipeline

#imagem de apresentação
imagem = 'alexandria.png'
st.image(imagem)

#Configuração do resumidor
summarizer = pipeline("summarization")

# Função para realizar a pesquisa e extrair informações dos resultados
def realizar_pesquisa(termo_pesquisa):
    url = f"https://scholar.google.com/scholar?q={termo_pesquisa}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extrair informações dos resultados de pesquisa
    resultados = soup.find_all('div', {'class': 'gs_ri'})[:5]  # limitando a análise aos primeiros 5 resultados
    titulos = [resultado.find('h3', {'class': 'gs_rt'}).text for resultado in resultados]
    autores = [resultado.find('div', {'class': 'gs_a'}).text for resultado in resultados]
    resumos = [resultado.find('div', {'class': 'gs_rs'}).text for resultado in resultados]
    anos = [resultado.find('div', {'class': 'gs_a'}).text.split('-')[-1].strip() for resultado in resultados]

    # Retornar os dados extraídos
    return titulos, autores, resumos, anos

# Configuração da página Streamlit
st.title("Análise de Artigos Científicos no Google Scholar")

# Obtenção do termo de pesquisa
termo_pesquisa = st.text_input("Digite o termo a ser pesquisado:", value='')

# Adiciona um botão na página
botao_pressionado = st.button("Pesquisar")

# Executa ação quando o botão for pressionado
if botao_pressionado:
    # Realiza a pesquisa e extrai informações dos resultados
    titulos, autores, resumos, anos = realizar_pesquisa(termo_pesquisa)

    # Cria um resumo geral de todos os resumos
    resumo_geral = summarizer(' '.join(resumos), max_length=400, min_length=200, do_sample=False)[0]['summary_text']

    # Exibe o resumo geral
    st.header("Resumo Geral:")
    st.write(resumo_geral)
    st.write("---")

    # Exibe os resultados individuais na página
    st.header("Resultados da Pesquisa:")
    for titulo, autor, resumo, ano in zip(titulos, autores, resumos, anos):
        st.subheader(titulo)
        st.write(f"Autor(es): {autor}")
        st.write(f"Ano de Publicação: {ano}")
        st.write(f"Resumo: {resumo}")
        st.write("---")

    #análise de frequência de palavras-chave com base no TF-IDF
    st.header("Análise de Frequência de Palavras-Chave (TF-IDF)")
    stopwords_nltk = stopwords.words('english')
    vectorizer = TfidfVectorizer(stop_words=stopwords_nltk)
    tfidf_matrix = vectorizer.fit_transform(resumos)
    feature_names = vectorizer.get_feature_names_out()
    termos_mais_importantes = tfidf_matrix.mean(axis=0).tolist()[0]
    termos_importantes_indices = sorted(range(len(termos_mais_importantes)), key=lambda i: termos_mais_importantes[i], reverse=True)[:10]
    termos = [feature_names[i] for i in termos_importantes_indices]
    frequencias = [termos_mais_importantes[i] for i in termos_importantes_indices]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(termos, frequencias)
    ax.set_xlabel('Importância (TF-IDF)')
    ax.set_ylabel('Palavras-Chave')
    ax.set_title('Termos Mais Importantes (TF-IDF)')
    st.pyplot(fig)

    #nuvem de palavras
    st.header("Nuvem de Palavras")
    text = ' '.join(resumos)
    wordcloud = WordCloud(width=800, height=400).generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


    #análise de Coocorrência de Palavras
    st.header("Análise de Coocorrência de Palavras")
    coocorrencia = Counter()
    for resumo in resumos:
        palavras_resumo = resumo.split()
        coocorrencia.update(Counter(zip(palavras_resumo, palavras_resumo[1:])))
    termos_coocorrencia, frequencias_coocorrencia = zip(*coocorrencia.items())
    termos_coocorrencia = [' '.join(termo) for termo in termos_coocorrencia]
    frequencias_coocorrencia = list(frequencias_coocorrencia)
    
    #definir quantidade limite de coocorrências a serem exibidas
    limite_coocorrencias = 20
    termos_coocorrencia = termos_coocorrencia[:limite_coocorrencias]
    frequencias_coocorrencia = frequencias_coocorrencia[:limite_coocorrencias]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(termos_coocorrencia, frequencias_coocorrencia)
    ax.set_xlabel('Frequência')
    ax.set_ylabel('Coocorrência de Palavras')
    ax.set_title('Coocorrência de Palavras (Top 10)')
    st.pyplot(fig)