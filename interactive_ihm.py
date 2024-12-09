import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="scientific-medical-abstracts",
        page_icon="👋",
    )

    st.write("# Welcome to MuchMore-project! 👋")

    st.sidebar.success("Select a page above.")

    st.markdown(
        """
        ### MuchMore-project
        This dataset consists of abstracts from medical scientific publications, 
        covering various fields such as Cardiology, Ophthalmology, etc. Therefore,
        we are dealing with a multiclass classification problem (assigning a single 
        possible class to a document). The category to which a document belongs 
        corresponds to the first part of its name.

        **👈 Select a page from the sidebar** to see the dashboard
        or to play with the interactive exploration of data !
        
    """
    )


if __name__ == "__main__":
    run()
