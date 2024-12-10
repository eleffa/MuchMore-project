import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Configuration de la page principale
st.set_page_config(
    page_title="Scientific Medical Abstracts",
    page_icon="📊",
    layout="wide"
)

# Ajout de la barre de navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Accueil", "Dashboard", "Exploration Interactive"]
)

# Charger les données (exemple)
@st.cache
#def load_data():
    # Remplacez ceci par le chargement réel de vos données
#    return pd.DataFrame({
#        "abstract": [
#            "Heart disease treatment",
#            "Diabetes prevention study",
#            "Cancer immunotherapy",
#            "Ophthalmology and vision",
#            "Cardiology and blood pressure",
#        ],
#        "category": ["Cardiology", "Endocrinology", "Oncology", "Ophthalmology", "Cardiology"],
#        "length": [120, 85, 200, 110, 95]
#    })

@st.cache
def load_data_from_github(url: str):
    """
    Load data from a CSV file hosted on GitHub.

    Parameters:
    - url (str): The raw URL of the CSV file in the GitHub repository.

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    """
    try:
        data = pd.read_csv(url)
        return data
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if an error occurs



df = load_data_from_github("https://raw.githubusercontent.com/eleffa/MuchMore-project/main/MuchMoreData_clean.csv")

# Accueil
if page == "Accueil":
    st.title("Welcome to MuchMore Project! 👋")
    st.sidebar.success("Vous êtes sur la page d'accueil.")

    st.markdown(
        """
        ### MuchMore-project
        This dataset consists of abstracts from medical scientific publications, 
        covering various fields such as Cardiology, Ophthalmology, etc. Therefore,
        we are dealing with a multiclass classification problem (assigning a single 
        possible class to a document). 
        
        **👈 Select a page from the sidebar** to see the dashboard
        or to play with the interactive exploration of data!
        """
    )

# Dashboard
elif page == "Dashboard":
    st.title("Classification des Abstracts Médicaux 📊")
    st.sidebar.success("Vous êtes sur la page Dashboard.")

    # Section 1 : Vue d'ensemble
    st.header("Vue d'ensemble")
    st.write("Résumé des données")
    # Exemple : ajouter des statistiques clés
    st.metric("Total Abstracts", 7823)
    st.metric("Nombre de Classes", 39)

    # Ajouter un diagramme de distribution des classes (importer une image générée)
    st.image("https://raw.githubusercontent.com/eleffa/MuchMore-project/main/dashboard/distribution_categories.png",
    caption="Distribution des Classes", use_column_width=True)

    # Section 2 : Analyse exploratoire
    st.header("Analyse exploratoire")
    st.subheader("Histogramme des longueurs")
    st.image("https://raw.githubusercontent.com/eleffa/MuchMore-project/main/dashboard/distribution_longueur.png", 
             caption="Distribution des longueurs des abstracts")
    

    st.subheader("Nuage de mots")
    st.image("https://raw.githubusercontent.com/eleffa/MuchMore-project/main/dashboard/nuage_de_mots.png", caption="Nuage de mots")

    st.subheader("Heatmap de similarité")
    st.image("https://raw.githubusercontent.com/eleffa/MuchMore-project/main/dashboard/heatmap_similarite.png", caption="Similarité entre catégories")

    # Section 3 : Modélisation
    st.header("Modélisation")

    st.subheader("Graphique radar")
    st.image("https://raw.githubusercontent.com/eleffa/MuchMore-project/main/dashboard/radar.png", caption="Graphique radar")

    st.subheader("Comparaison des Scores Moyens par Modèle")
    st.image("https://raw.githubusercontent.com/eleffa/MuchMore-project/main/dashboard/comparaison_des_scores.png", caption="Comparaison des Scores Moyens par Modèle")


    st.subheader("Matrice de confusion - SVM")
    st.image("https://raw.githubusercontent.com/eleffa/MuchMore-project/main/dashboard/confusion_svm.png", caption="Matrice de confusion - SVM")

    st.subheader("Matrice de confusion - Naive Bayes")
    st.image("https://raw.githubusercontent.com/eleffa/MuchMore-project/main/dashboard/confusion_naive.png", caption="Matrice de confusion - Naive Bayes")

    
    st.subheader("Courbes ROC/AUC")
    st.image("https://raw.githubusercontent.com/eleffa/MuchMore-project/main/dashboard/courbe_roc.png", caption="Courbes ROC - SVM vs Naive Bayes")
    

# Exploration Interactive
elif page == "Exploration Interactive":
    st.title("Exploration Interactive 🕵️")
    st.sidebar.success("Vous êtes sur la page Exploration Interactive.")

    # Prétraitement des données
    tfidf = TfidfVectorizer(max_features=500)
    X = tfidf.fit_transform(df['clean_content'])
    y = df['clean_category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

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
    filter_language = st.selectbox("Filtrer par langue", options=['Toutes'] + list(df['lang'].unique()))

    filtered_data = df.copy()
    if filter_category != 'Toutes':
        filtered_data = filtered_data[filtered_data['clean_category'] == filter_category]
    if filter_language != 'Toutes':
        filtered_data = filtered_data[filtered_data['lang'] == filter_language]

    st.write("Données filtrées :", filtered_data)

    # Ajuster les paramètres des modèles
    st.sidebar.header("Paramètres des modèles")
    C_value = st.sidebar.slider("Coefficient de régularisation (C) - SVM", 0.1, 10.0, 1.0, step=0.1)
    alpha_value = st.sidebar.slider("Smoothing Parameter (alpha) - Naive Bayes", 0.1, 10.0, 1.0, step=0.1)

    # Entraînement des modèles avec paramètres ajustés
    svm, nb, y_pred_svm, y_pred_nb, svm_report, nb_report = train_models(C=C_value, alpha=alpha_value)

    # Exploration des prédictions
    st.subheader("Exploration des prédictions")
    model_choice = st.radio("Choisissez un modèle pour voir les prédictions", ['SVM', 'Naive Bayes'])
    if model_choice == 'SVM':
        st.write("Prédictions du modèle SVM :")
        y_pred = y_pred_svm
    elif model_choice == 'Naive Bayes':
        st.write("Prédictions du modèle Naive Bayes :")
        y_pred = y_pred_nb

    # Documents mal classés
    misclassified = X_test[np.where(y_pred != y_test)]
    st.write("Exemples de documents mal classés :")
    for i, doc in enumerate(misclassified[:5]):
        st.write(f"**Document {i+1} :** {tfidf.inverse_transform(doc)}")
        st.write(f"**Document {i+1} : **{df['content'].iloc[i]}")
        st.write(f"**Classe réelle :** {y_test.iloc[i]} | **Classe prédite :** {y_pred[i]}")

    # Visualisation des performances
    st.subheader("Performances des modèles")
    metrics = ['precision', 'recall', 'f1-score']
    svm_scores = [svm_report['weighted avg'][metric] for metric in metrics]
    nb_scores = [nb_report['weighted avg'][metric] for metric in metrics]

    # Comparaison graphique
    labels = ['Precision', 'Recall', 'F1-Score']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, svm_scores, width, label="SVM")
    ax.bar(x + width/2, nb_scores, width, label="Naive Bayes")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    st.pyplot(fig)

    
