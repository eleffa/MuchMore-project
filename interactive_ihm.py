import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

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



df = load_data_from_github("https://github.com/eleffa/MuchMore-project/blob/main/MuchMoreData_clean.csv")

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
    st.title("Dashboard - Classification des Abstracts Médicaux 📊")
    st.sidebar.success("Vous êtes sur la page Dashboard.")

    # Section 1 : Vue d'ensemble
    st.header("Vue d'ensemble")
    st.write("Résumé des données")
    # Exemple : ajouter des statistiques clés
    st.metric("Total Abstracts", 7823)
    st.metric("Nombre de Classes", 39)

# Exploration Interactive
elif page == "Exploration Interactive":
    st.title("Exploration Interactive 🕵️")
    st.sidebar.success("Vous êtes sur la page Exploration Interactive.")

    # Tableau interactif
    st.subheader("Tableau interactif des données")
    filter_category = st.selectbox("Filtrer par catégorie", options=["Toutes"] + df['category'].unique().tolist())
    if filter_category != "Toutes":
        filtered_df = df[df['category'] == filter_category]
    else:
        filtered_df = df

    st.write("Données filtrées :", filtered_df)

    # Analyse de prédictions (exemple fictif)
    st.subheader("Exploration des prédictions mal classées")
    st.markdown("**Exemple de document mal classé**")
    st.write("Document : Heart disease treatment")
    st.write("Classe réelle : Cardiology | Classe prédite : Oncology")
