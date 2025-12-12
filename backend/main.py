from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import json
import os
from typing import Dict, List
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application
app = FastAPI(
    title="Plant Disease Classification API",
    description="API pour la détection de maladies des plantes",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
IMG_SIZE = 224
MODEL_PATH = "plant_disease_model.h5"
CLASS_NAMES_PATH = "class_names.json"

# Variables globales
model = None
class_names = None

# Charger le modèle et les noms de classes
@app.on_event("startup")
async def load_model():
    """Charge le modèle et les classes au démarrage"""
    global model, class_names
    
    try:
        logger.info("Chargement du modèle...")
        
        # Vérifier si le modèle existe
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Modèle non trouvé: {MODEL_PATH}")
            raise FileNotFoundError(f"Le fichier {MODEL_PATH} n'existe pas")
        
        # Créer le dossier models si nécessaire
        os.makedirs("models", exist_ok=True)
        
        # différentes stratégies pour le chargement du modèle
        try:
            # Stratégie 1: Chargement avec compile=False
            logger.info("Tentative de chargement avec compile=False...")
            model = keras.models.load_model(MODEL_PATH, compile=False)
            logger.info("✓ Modèle chargé avec compile=False")
            
        except Exception as e1:
            logger.warning(f"Échec stratégie 1: {e1}")
            
            try:
                # Stratégie 2: Chargement avec safe_mode=False
                logger.info("Tentative avec safe_mode=False...")
                model = keras.models.load_model(MODEL_PATH, safe_mode=False, compile=False)
                logger.info("✓ Modèle chargé avec safe_mode=False")
                
            except Exception as e2:
                logger.warning(f"Échec stratégie 2: {e2}")
                
                # Stratégie 3: Chargement avec TensorFlow directement
                logger.info("Tentative avec tf.keras...")
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                logger.info("✓ Modèle chargé avec tf.keras")
        
        # Recompiler le modèle pour l'inférence
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(" Modèle chargé et compilé avec succès")
        logger.info(f"Architecture du modèle: {len(model.layers)} couches")
        
        # Charger les noms de classes
        logger.info("Chargement des noms de classes...")
        if not os.path.exists(CLASS_NAMES_PATH):
            logger.error(f"Fichier classes non trouvé: {CLASS_NAMES_PATH}")
            raise FileNotFoundError(f"Le fichier {CLASS_NAMES_PATH} n'existe pas")
            
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        logger.info(f"✓ {len(class_names)} classes chargées")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement: {str(e)}")
        logger.error(f"Type d'erreur: {type(e).__name__}")
        raise

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prétraite l'image pour la prédiction"""
    try:
        # Convertion en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionnement
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convertion  en array et normaliser
        img_array = np.array(image) / 255.0
        
        # Ajout de la dimension batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement: {str(e)}")
        raise

def format_disease_name(class_name: str) -> Dict[str, str]:
    """Formate le nom de la classe en informations structurées"""
    parts = class_name.split('___')
    
    if len(parts) == 2:
        plant = parts[0].replace('_', ' ').title()
        disease = parts[1].replace('_', ' ').title()
        
        # Déterminons si c'est sain ou malade
        is_healthy = 'healthy' in disease.lower()
        
        return {
            'plant': plant,
            'disease': disease,
            'is_healthy': is_healthy,
            'full_name': f"{plant} - {disease}"
        }
    else:
        return {
            'plant': 'Unknown',
            'disease': class_name.replace('_', ' ').title(),
            'is_healthy': False,
            'full_name': class_name.replace('_', ' ').title()
        }

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "Plant Disease Classification API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image for prediction",
            "/health": "GET - Check API health",
            "/classes": "GET - Get list of all classes",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Vérifie la santé de l'API"""
    if model is None or class_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "num_classes": len(class_names) if class_names else 0
    }

@app.get("/classes")
async def get_classes():
    """Retourne la liste de toutes les classes"""
    if class_names is None:
        raise HTTPException(status_code=503, detail="Classes not loaded")
    
    formatted_classes = []
    for idx, name in class_names.items():
        formatted_classes.append({
            'id': int(idx),
            'name': name,
            'formatted': format_disease_name(name)
        })
    
    return {
        "total_classes": len(class_names),
        "classes": formatted_classes
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Effectue une prédiction sur une image uploadée
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        JSON avec les prédictions top-5
    """
    if model is None or class_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Vérifier le type de fichier
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"File must be an image. Received: {file.content_type}"
        )
    
    try:
        # Lire et prétraiter l'image
        logger.info(f"Traitement de l'image: {file.filename}")
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        processed_image = preprocess_image(image)
        
        # Prédiction
        logger.info("Prédiction en cours...")
        predictions = model.predict(processed_image, verbose=0)
        predictions = predictions[0]
        
        # Top 5 prédictions
        top_5_indices = np.argsort(predictions)[-5:][::-1]
        
        results = []
        for idx in top_5_indices:
            confidence = float(predictions[idx])
            class_name = class_names[str(idx)]
            formatted = format_disease_name(class_name)
            
            results.append({
                'class_id': int(idx),
                'class_name': class_name,
                'plant': formatted['plant'],
                'disease': formatted['disease'],
                'is_healthy': formatted['is_healthy'],
                'confidence': round(confidence * 100, 2)
            })
        
        # Meilleure prédiction
        best_prediction = results[0]
        
        logger.info(
            f"✓ Prédiction: {best_prediction['plant']} - "
            f"{best_prediction['disease']} ({best_prediction['confidence']}%)"
        )
        
        return {
            "success": True,
            "filename": file.filename,
            "prediction": best_prediction,
            "top_5_predictions": results,
            "recommendation": get_recommendation(best_prediction)
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )

def get_recommendation(prediction: Dict) -> str:
    """Génère une recommandation basée sur la prédiction"""
    if prediction['is_healthy']:
        return "Plante en bonne santé ! Continuez les soins réguliers."
    else:
        confidence = prediction['confidence']
        if confidence > 90:
            return (
                f" Maladie détectée avec haute confiance ({confidence}%). "
                "Traitement recommandé immédiatement."
            )
        elif confidence > 70:
            return (
                f" Maladie probable ({confidence}%). "
                "Surveillance recommandée et consultation si symptômes persistent."
            )
        else:
            return (
                f" Maladie possible ({confidence}%). "
                "Consultation d'un expert recommandée pour confirmation."
            )

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Prédictions sur plusieurs images
    
    Args:
        files: Liste d'images (maximum 10)
    
    Returns:
        Liste des prédictions pour chaque image
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 10 images per batch"
        )
    
    results = []
    for idx, file in enumerate(files):
        try:
            logger.info(f"Traitement image {idx + 1}/{len(files)}: {file.filename}")
            
            # Vérifier le type
            if not file.content_type.startswith('image/'):
                results.append({
                    "success": False,
                    "filename": file.filename,
                    "error": "File is not an image"
                })
                continue
            
            # Lire et prétraiter
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            processed_image = preprocess_image(image)
            
            # Prédiction
            predictions = model.predict(processed_image, verbose=0)[0]
            
            # Top 5
            top_5_indices = np.argsort(predictions)[-5:][::-1]
            
            top_predictions = []
            for pred_idx in top_5_indices:
                confidence = float(predictions[pred_idx])
                class_name = class_names[str(pred_idx)]
                formatted = format_disease_name(class_name)
                
                top_predictions.append({
                    'class_id': int(pred_idx),
                    'class_name': class_name,
                    'plant': formatted['plant'],
                    'disease': formatted['disease'],
                    'is_healthy': formatted['is_healthy'],
                    'confidence': round(confidence * 100, 2)
                })
            
            best_pred = top_predictions[0]
            
            results.append({
                "success": True,
                "filename": file.filename,
                "prediction": best_pred,
                "top_5_predictions": top_predictions,
                "recommendation": get_recommendation(best_pred)
            })
            
        except Exception as e:
            logger.error(f"Erreur pour {file.filename}: {str(e)}")
            results.append({
                "success": False,
                "filename": file.filename,
                "error": str(e)
            })
    
    # Statistiques
    successful = sum(1 for r in results if r.get("success", False))
    
    return {
        "total_images": len(files),
        "successful": successful,
        "failed": len(files) - successful,
        "results": results
    }

# Point d'entrée pour exécution locale
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
