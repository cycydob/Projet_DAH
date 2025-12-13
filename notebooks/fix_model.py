"""
Script pour recréer le modèle depuis les poids sans problème de compatibilité
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import h5py
import json



# Charger le fichier class_names pour connaître le nombre de classes
try:
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"✓ Nombre de classes: {num_classes}")
except:
    print("class_names.json non trouvé, utilisation de 38 classes par défaut")
    num_classes = 38

# Recréer l'architecture du modèle (identique au training)
print("\nRecréation de l'architecture du modèle...")

IMG_SIZE = 224

# Charger MobileNetV2 pré-entraîné
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Geler les couches de base
base_model.trainable = False

# Création du modèle complet
model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),  # Utilisation de Input au lieu de batch_shape
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

print("✓ Architecture créée")

# Compiler le modèle
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Modèle compilé")

#  chargement des poids de l'ancien modèle
print("\nTentative de chargement des poids de l'ancien modèle...")
try:
    # Ouverture du fichier H5
    with h5py.File('plant_disease_model.h5', 'r') as f:
        # Vérification de la structure
        if 'model_weights' in f.keys():
            print(" Poids trouvés dans le fichier")
            # Charger les poids
            model.load_weights('plant_disease_model.h5')
            print(" Poids chargés avec succès!")
        else:
            print(" Structure de poids non reconnue")
            print("Le modèle sera sauvegardé avec des poids aléatoires (nécessite réentraînement)")
except Exception as e:
    print(f" Impossible de charger les poids: {e}")
    print("Le modèle sera sauvegardé avec l'architecture correcte mais nécessite réentraînement")

# Sauvegarder le nouveau modèle dans backend/
import os
os.makedirs('../backend', exist_ok=True)
os.makedirs('../backend/models', exist_ok=True)

print("\nSauvegarde du modèle corrigé...")

# Sauvegarder en H5 sans optimizer
model.save('../backend/plant_disease_model.h5', 
           save_format='h5',
           include_optimizer=False,
           save_traces=False)
print(" Modèle sauvegardé: ../backend/plant_disease_model.h5")

# Copie dans models/
model.save('../backend/models/best_model.h5',
           save_format='h5',
           include_optimizer=False,
           save_traces=False)
print(" Copie sauvegardée: ../backend/models/best_model.h5")

# Copier class_names.json
try:
    import shutil
    shutil.copy('class_names.json', '../backend/class_names.json')
    print(" class_names.json copié")
except:
    print(" class_names.json non copié")
print("\nCorrection terminée.")
