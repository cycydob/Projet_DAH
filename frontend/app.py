import streamlit as st
import requests
from PIL import Image
import io
import json
import time

# Configuration de la page
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de l'API 
import os


try:
    API_URL = st.secrets.get("API_URL", "http://localhost:8000")
except:
    # En développement local
    API_URL = os.environ.get("API_URL", "http://localhost:8000")

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .healthy {
        background-color: #C8E6C9;
        border-left: 5px solid #4CAF50;
    }
    .diseased {
        background-color: #FFCCBC;
        border-left: 5px solid #FF5722;
    }
    .confidence-bar {
        background-color: #E0E0E0;
        border-radius: 10px;
        height: 30px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre
st.markdown('<div class="main-header">Plant Disease Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Détectez les maladies de vos plantes avec l\'IA</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/leaf.png", width=100)
    st.title("À propos")
    st.info("""
    Cette application utilise l'intelligence artificielle pour détecter les maladies des plantes à partir de photos de feuilles.
    
    **Comment l'utiliser:**
    1. Uploadez une photo de feuille
    2. Attendez l'analyse
    3. Consultez les résultats
    
    **Précision:** ~95% sur 38 classes de maladies
    """)
    
    st.divider()
    
    # Vérifier la connexion à l'API
    st.subheader("État de l'API")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("API connectée")
        else:
            st.error("API non disponible")
    except:
        st.error("Impossible de se connecter à l'API")
    
    st.divider()
    
    # Statistiques
    if st.button("Afficher les classes"):
        try:
            response = requests.get(f"{API_URL}/classes")
            if response.status_code == 200:
                data = response.json()
                st.metric("Classes disponibles", data['total_classes'])
        except:
            st.error("Erreur lors du chargement des classes")

# Interface principale
tabs = st.tabs(["Analyse Simple", "Analyse Multiple", "Informations"])

# TAB 1: Analyse Simple
with tabs[0]:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload de l'image")
        uploaded_file = st.file_uploader(
            "Choisissez une image de feuille",
            type=['jpg', 'jpeg', 'png'],
            help="Formats supportés: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Afficher l'image
            image = Image.open(uploaded_file)
            st.image(image, caption="Image uploadée", use_column_width=True)
            
            # Bouton d'analyse
            if st.button("Analyser l'image", type="primary", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    try:
                        # Préparer les données
                        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        
                        # Appeler l'API
                        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state['result'] = result
                            st.success("Analyse terminée!")
                        else:
                            st.error(f"Erreur: {response.text}")
                    
                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse: {str(e)}")
    
    with col2:
        st.subheader("Résultats")
        
        if 'result' in st.session_state and st.session_state['result']:
            result = st.session_state['result']
            prediction = result['prediction']
            
            # Box principale avec résultat
            box_class = "healthy" if prediction['is_healthy'] else "diseased"
            
            st.markdown(f"""
            <div class="result-box {box_class}">
                <h2>{"Plante Saine" if prediction['is_healthy'] else "Maladie Détectée"}</h2>
                <h3>Plante: {prediction['plant']}</h3>
                <h3>État: {prediction['disease']}</h3>
                <h3>Confiance: {prediction['confidence']}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Barre de confiance
            st.progress(prediction['confidence'] / 100)
            
            # Recommandation
            st.info(f"**Recommandation:** {result['recommendation']}")
            
            # Top 5 prédictions
            with st.expander("Voir les 5 meilleures prédictions"):
                for i, pred in enumerate(result['top_5_predictions'], 1):
                    status = "Saine" if pred['is_healthy'] else "Malade"
                    st.write(f"{i}. {status} **{pred['plant']}** - {pred['disease']} ({pred['confidence']}%)")
                    st.progress(pred['confidence'] / 100)
        else:
            st.info("Uploadez une image pour commencer l'analyse")

# TAB 2: Analyse Multiple
with tabs[1]:
    st.subheader("Analyse de plusieurs images")
    st.write("Uploadez jusqu'à 10 images pour une analyse groupée")
    
    uploaded_files = st.file_uploader(
        "Choisissez plusieurs images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if len(uploaded_files) > 10:
            st.error("Maximum 10 images autorisées")
        else:
            # Afficher les images
            cols = st.columns(min(len(uploaded_files), 3))
            for idx, file in enumerate(uploaded_files):
                with cols[idx % 3]:
                    image = Image.open(file)
                    st.image(image, caption=file.name, use_column_width=True)
            
            if st.button("Analyser toutes les images", type="primary"):
                progress_bar = st.progress(0)
                results_container = st.container()
                
                with results_container:
                    for idx, file in enumerate(uploaded_files):
                        with st.spinner(f"Analyse de {file.name}..."):
                            try:
                                files = {'file': (file.name, file.getvalue(), file.type)}
                                response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    pred = result['prediction']
                                    
                                    col1, col2 = st.columns([1, 3])
                                    with col1:
                                        st.image(Image.open(file), use_column_width=True)
                                    with col2:
                                        status = "Saine" if pred['is_healthy'] else "Malade"
                                        st.write(f"**{file.name}**")
                                        st.write(f"{status} {pred['plant']} - {pred['disease']}")
                                        st.progress(pred['confidence'] / 100)
                                        st.caption(f"Confiance: {pred['confidence']}%")
                                    
                                    st.divider()
                                else:
                                    st.error(f"Erreur pour {file.name}")
                            
                            except Exception as e:
                                st.error(f"Erreur pour {file.name}: {str(e)}")
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                
                st.success("Toutes les analyses sont terminées!")

# TAB 3: Informations
with tabs[2]:
    st.subheader("Comment ça marche ?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Technologie
        - **Modèle**: CNN avec Transfer Learning (MobileNetV2)
        - **Dataset**: 38 classes de maladies
        - **Précision**: ~95% sur le set de validation
        - **Backend**: FastAPI
        - **Frontend**: Streamlit
        
        ### Plantes supportées
        - Tomate
        - Pomme de terre
        - Poivron
        - Raisin
        - Maïs
        - Pêcher
        - Et bien d'autres...
        """)
    
    with col2:
        st.markdown("""
        ### Conseils pour de meilleurs résultats
        1. **Éclairage**: Prenez la photo en lumière naturelle
        2. **Angle**: Photographiez la feuille à plat
        3. **Distance**: La feuille doit occuper la majorité de l'image
        4. **Focus**: Assurez-vous que l'image est nette
        5. **Fond**: Un fond uni améliore la détection
        
        ### Limitations
        - L'outil est une aide au diagnostic, pas un remplacement d'expert
        - Consultez un agronome pour un diagnostic définitif
        - La précision dépend de la qualité de l'image
        """)
    
    st.divider()
    
    # Exemples
    st.subheader("Exemples de maladies détectables")
    st.write("Voici quelques exemples de maladies que le modèle peut identifier:")
    
    diseases = [
        {"name": "Mildiou de la tomate", "severity": "Élevée"},
        {"name": "Rouille du maïs", "severity": "Moyenne"},
        {"name": "Oïdium du raisin", "severity": "Élevée"},
        {"name": "Tache foliaire de la pomme de terre", "severity": "Moyenne"},
    ]
    
    cols = st.columns(4)
    for idx, disease in enumerate(diseases):
        with cols[idx]:
            st.info(f"**{disease['name']}**\nSévérité: {disease['severity']}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Développé par Dobé Aumounoh Nancy Prisca | Powered by TensorFlow & Streamlit</p>
  
</div>
""", unsafe_allow_html=True)
