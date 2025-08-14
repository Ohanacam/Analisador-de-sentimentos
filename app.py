import streamlit as st
import joblib
import pickle
import os
import pandas as pd
import unidecode
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuração inicial robusta do NLTK
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('punkt_tab')
        except Exception as e:
            st.error(f"Erro ao baixar recursos do NLTK: {str(e)}")
            return False
    return True

if not setup_nltk():
    st.error("Não foi possível configurar o NLTK. O app não pode continuar.")
    st.stop()

# Função de pré-processamento segura
def preprocess_text(text):
    try:
        puncts = list(string.punctuation)
        stopwords_pt = stopwords.words("portuguese")
        adicionais = ["...", "..", "etc", "diz", "ficou", "app", "aplicativo"]
        stopwords_puncts = list(set(stopwords_pt + puncts + adicionais))
        
        text = unidecode.unidecode(text.lower())
        tokens = nltk.word_tokenize(text, language='portuguese')
        tokens = [word for word in tokens if word not in stopwords_puncts]
        return " ".join(tokens)
    except Exception as e:
        st.error(f"Erro no pré-processamento: {str(e)}")
        return ""

# Carregamento seguro dos componentes
@st.cache_resource
def load_components():
    try:
        # Caminhos consistentes (usando raw strings ou barras normais)
        model_path = os.path.join('models', 'thirdanalysis_new_model.pkl')
        vectorizer_path = os.path.join('models', 'vectorizer_thirdanalysis_new.pkl')
        
        # Verifica se os arquivos existem
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            st.error("Arquivos do modelo não encontrados na pasta 'models'")
            return None, None
            
        # Carrega primeiro o vetorizador
        try:
            vectorizer = joblib.load(vectorizer_path)
        except:
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
                
        # Verifica o vetorizador
        if not hasattr(vectorizer, 'vocabulary_'):
            st.error("Vetorizador não foi treinado corretamente")
            return None, None
            
        # Carrega o modelo
        try:
            model = joblib.load(model_path)
        except:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
        return model, vectorizer
        
    except Exception as e:
        st.error(f"Erro crítico ao carregar componentes: {str(e)}")
        return None, None

# Interface principal
def main():
    st.set_page_config(page_title="Analisador de Sentimentos", page_icon="😊")
    st.title("📝 Analisador de Sentimentos em Português")
    st.write("Digite uma avaliação em português para análise de sentimento")

    # Carrega os componentes
    model, vectorizer = load_components()
    
    if model is None or vectorizer is None:
        st.error("Não foi possível carregar os componentes necessários")
        st.stop()  # Encerra o app completamente

    user_input = st.text_area("Digite sua avaliação:", help="Escreva uma avaliação em português sobre um produto ou serviço")
    
    if st.button("Analisar Sentimento", type="primary"):
        if not user_input.strip():
            st.warning("Por favor, digite uma avaliação.")
            return
            
        with st.spinner('Analisando...'):
            processed_text = preprocess_text(user_input)
            
            try:
                text_vector = vectorizer.transform([processed_text])
                prediction = model.predict(text_vector)
                proba = model.predict_proba(text_vector)[0]
                
                # Resultado
                st.subheader("Resultado:")
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction[0] == 1:
                        st.success("✅ Positivo")
                    else:
                        st.error("❌ Negativo")
                
                with col2:
                    st.metric("Confiança", 
                              f"{max(proba)*100:.1f}%",
                              delta=f"{max(proba)*100 - 50:.1f}%")
                
                # Gráfico
                prob_data = pd.DataFrame({
                    'Sentimento': ['Negativo', 'Positivo'],
                    'Probabilidade': proba
                })
                st.bar_chart(prob_data.set_index('Sentimento'))
                
            except Exception as e:
                st.error(f"Erro na análise: {str(e)}")
                st.exception(e)  # Mostra detalhes do erro para debug

if __name__ == "__main__":
    main()
