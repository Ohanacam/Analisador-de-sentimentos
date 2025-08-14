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

# Carregamento seguro do vetorizador com verificação
@st.cache_resource
def load_vectorizer():
    try:
        # Tenta carregar com joblib
        vectorizer = joblib.load('vectorizer_thirdanalysis_new.pkl')
        
        # Verifica se o vetorizador está fitted
        if not hasattr(vectorizer, 'vocabulary_'):
            st.error("Vetorizador não foi treinado corretamente")
            return None
            
        return vectorizer
    except:
        try:
            # Fallback para pickle
            with open('vectorizer_thirdanalysis_new.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
                
            if not hasattr(vectorizer, 'vocabulary_'):
                st.error("Vetorizador não foi treinado corretamente")
                return None
                
            return vectorizer
        except Exception as e:
            st.error(f"Erro ao carregar vetorizador: {str(e)}")
            return None

# Carregamento seguro do modelo
@st.cache_resource
def load_model():
    try:
        return joblib.load('thirdanalysis_new_model.pkl')
    except:
        try:
            with open('thirdanalysis_new_model.pkl', 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {str(e)}")
            return None

# Interface principal
def main():
    st.set_page_config(page_title="Analisador de Sentimentos", page_icon="😊")
    st.title("📝 Analisador de Sentimentos em Português")
    st.write("Digite uma avaliação em português para análise de sentimento")

    # Carrega os componentes
    vectorizer = load_vectorizer()
    model = load_model()
    
    if vectorizer is None or model is None:
        st.error("Não foi possível carregar os componentes necessários")
        return

    user_input = st.text_area("Digite sua avaliação:")
    
    if st.button("Analisar Sentimento"):
        if not user_input.strip():
            st.warning("Por favor, digite uma avaliação.")
            return
            
        processed_text = preprocess_text(user_input)
        
        try:
            # Verificação final do vetorizador
            if not hasattr(vectorizer, 'transform'):
                st.error("Vetorizador não está pronto para transformação")
                return
                
            text_vector = vectorizer.transform([processed_text])
            prediction = model.predict(text_vector)
            proba = model.predict_proba(text_vector)[0]
            
            st.subheader("Resultado:")
            if prediction[0] == 1:
                st.success(f"✅ Positivo ({(proba[1]*100):.1f}% de confiança)")
            else:
                st.error(f"❌ Negativo ({(proba[0]*100):.1f}% de confiança)")
            
            # Visualização
            prob_data = pd.DataFrame({
                'Sentimento': ['Positivo', 'Negativo'],
                'Probabilidade': [proba[1], proba[0]]
            })
            st.bar_chart(prob_data.set_index('Sentimento'))
            
        except Exception as e:
            st.error(f"Erro na análise: {str(e)}")

if __name__ == "__main__":

    main()
