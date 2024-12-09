import streamlit as st
import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import train_test_split
#from sklearn.svm import LinearSVC
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Charger les données
@st.cache
def load_data():
    # Exemple de données (remplacez par vos propres données)
    data = pd.DataFrame({
        'abstract': [
            'Heart disease treatment',
            'Diabetes prevention study',
            'Cancer immunotherapy',
            'Ophthalmology and vision',
            'Cardiology and blood pressure',
        ],
        'clean_category': ['Cardiology', 'Endocrinology', 'Oncology', 'Ophthalmology', 'Cardiology'],
        'language': ['en', 'en', 'en', 'en', 'en']
    })
    return data

df = load_data()

# Prétraitement des données
#tfidf = TfidfVectorizer(max_features=500)
#X = tfidf.fit_transform(df['abstract'])
#y = df['clean_category']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Entraînement des modèles
@st.cache
def train_models(C=1.0, alpha=1.0):
    # SVM
    svm = LinearSVC(C=C, random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    svm_report = classification_report(y_test, y_pred_svm, output_dict=True)

    # Naive Bayes
    nb = MultinomialNB(alpha=alpha)
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    nb_report = classification_report(y_test, y_pred_nb, output_dict=True)

    return svm, nb, y_pred_svm, y_pred_nb, svm_report, nb_report

# Interface Streamlit
st.title("Exploration Interactive des Données et Résultats")

# Tableau interactif
st.subheader("Exploration des données")
filter_category = st.selectbox("Filtrer par catégorie", options=['Toutes'] + list(df['clean_category'].unique()))
filter_language = st.selectbox("Filtrer par langue", options=['Toutes'] + list(df['language'].unique()))

filtered_data = df.copy()
if filter_category != 'Toutes':
    filtered_data = filtered_data[filtered_data['clean_category'] == filter_category]
if filter_language != 'Toutes':
    filtered_data = filtered_data[filtered_data['language'] == filter_language]

st.write("Données filtrées :", filtered_data)

# Ajuster les paramètres des modèles
st.sidebar.header("Paramètres des modèles")
C_value = st.sidebar.slider("Coefficient de régularisation (C) - SVM", 0.1, 10.0, 1.0, step=0.1)
alpha_value = st.sidebar.slider("Smoothing Parameter (alpha) - Naive Bayes", 0.1, 10.0, 1.0, step=0.1)

# Entraînement des modèles avec paramètres ajustés
#svm, nb, y_pred_svm, y_pred_nb, svm_report, nb_report = train_models(C=C_value, alpha=alpha_value)

# Exploration des prédictions
st.subheader("Exploration des prédictions")
model_choice = st.radio("Choisissez un modèle pour voir les prédictions", ['SVM', 'Naive Bayes'])
if model_choice == 'SVM':
    st.write("Prédictions du modèle SVM :")
     #y_pred = y_pred_svm
elif model_choice == 'Naive Bayes':
    st.write("Prédictions du modèle Naive Bayes :")
    #y_pred = y_pred_nb

# Documents mal classés
#misclassified = X_test[np.where(y_pred != y_test)]
st.write("Exemples de documents mal classés :")
for i, doc in enumerate(misclassified[:5]):
    st.write(f"**Document {i+1} :** {tfidf.inverse_transform(doc)}")
    st.write(f"**Classe réelle :** {y_test.iloc[i]} | **Classe prédite :** {y_pred[i]}")

# Visualisation des performances
st.subheader("Performances des modèles")
metrics = ['precision', 'recall', 'f1-score']
#svm_scores = [svm_report['weighted avg'][metric] for metric in metrics]
#nb_scores = [nb_report['weighted avg'][metric] for metric in metrics]

# Comparaison graphique
labels = ['Precision', 'Recall', 'F1-Score']
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots()
#ax.bar(x - width/2, svm_scores, width, label="SVM")
#ax.bar(x + width/2, nb_scores, width, label="Naive Bayes")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
st.pyplot(fig)
