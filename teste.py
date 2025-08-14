# Substitua as funções de load por:
import joblib

def load_model():
    return joblib.load('thirdanalysis_new_model.pkl')

def load_vectorizer():
    return joblib.load('vectorizer_thirdanalysis_new.pkl')