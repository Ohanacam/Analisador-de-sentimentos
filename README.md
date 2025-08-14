# 📱 Analisador de Sentimentos para Avaliações de Apps em Português
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://seu-app.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

## 📌 Introdução

Este projeto implementa um modelo de Machine Learning para análise de sentimentos em avaliações de aplicativos em Português Brasileiro. O sistema classifica textos como **positivos** ou **negativos** com base em padrões linguísticos aprendidos a partir de dados reais. desenvolvido com:

- 🧠 Scikit-learn (Regressão Logística)
- 🔤 NLTK para NLP em português
- 🚀 Streamlit para interface web
  
**Aplicações práticas**:
- Monitoramento automático de reviews em app stores
- Análise de satisfação do usuário
- Identificação de problemas em produtos digitais

## 🔍 Sobre o Projeto

### Base de Dados
Utilizamos o dataset [Brazilian Portuguese Sentiment Analysis Datasets](https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets) do Kaggle, especificamente a parte **UTLC-Apps** que contém:

- O modelo alcançou **88% de acurácia** na validação.
- 130,000+ avaliações da Google Play Store
- Textos em português brasileiro natural
- Classificações de 1-5 estrelas

### Arquitetura do Modelo
- **Pré-processamento**: Limpeza de texto, remoção de stopwords e lematização
- **Vetorização**: TF-IDF com n-grams (1-3)
- **Algoritmo**: Regressão Logística com otimização LBFGS

## 🚀 Teste o Modelo

Acesse a implementação em produção no Streamlit:  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.com/seu-usuario/sentiment-analysis-app/main/app.py)

**⚠️ Atenção**: Apesar da boa acurácia (88%), o modelo pode:

- Classificar incorretamente textos ambíguos ou irônicos
- Ter desempenho reduzido em gírias muito regionais
- Apresentar viés em domínios específicos

## 📈 Performance do Modelo

### Métricas gerais
| Métrica       | Positivo | Negativo |
|---------------|----------|----------|
| Precision     | 0.89     | 0.87     |
| Recall        | 0.85     | 0.90     |
| F1-Score      | 0.87     | 0.88     |

### 📌 Matriz de Confusão

| Métrica       | Positivo | Negativo |
|---------------|----------|----------|
Real Negativo   | 918      |  81      |
Real Positivo   | 148      | 852      |

## ⚠️ Limitações e Melhorias Futuras

**Limitações atuais**:
- Dificuldade com linguagem informal e gírias
- Desempenho reduzido em textos curtos
- Não detecta sarcasmo/ironia

**Próximas atualizações**:
- [ ] Adicionar suporte a emojis
- [ ] Implementar novos algoritmos 
- [ ] Coletar mais dados

---
Desenvolvido com ❤️ por Ohana - [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://github.com/Ohanacam)
