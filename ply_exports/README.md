# Exports PLY de segmentation PointNet++

## Fichiers générés

### Échantillon 1
- `sample_01_ground_truth.ply`: Vérité terrain
- `sample_01_predictions.ply`: Prédictions du modèle (avec confiance)
- `sample_01_comparison.ply`: Comparaison (erreurs en rouge)

### Échantillon 2
- `sample_02_ground_truth.ply`: Vérité terrain
- `sample_02_predictions.ply`: Prédictions du modèle (avec confiance)
- `sample_02_comparison.ply`: Comparaison (erreurs en rouge)

### Échantillon 3
- `sample_03_ground_truth.ply`: Vérité terrain
- `sample_03_predictions.ply`: Prédictions du modèle (avec confiance)
- `sample_03_comparison.ply`: Comparaison (erreurs en rouge)

### Échantillon 4
- `sample_04_ground_truth.ply`: Vérité terrain
- `sample_04_predictions.ply`: Prédictions du modèle (avec confiance)
- `sample_04_comparison.ply`: Comparaison (erreurs en rouge)

### Échantillon 5
- `sample_05_ground_truth.ply`: Vérité terrain
- `sample_05_predictions.ply`: Prédictions du modèle (avec confiance)
- `sample_05_comparison.ply`: Comparaison (erreurs en rouge)

## Visualisation

### Blender
1. Ouvrir Blender
2. Supprimer la scène par défaut (A + X + Confirmer)
3. Exécuter le script `import_ply_blender.py` dans l'éditeur de texte
4. Ou importer manuellement: File > Import > PLY

### CloudCompare
1. Suivre les instructions dans `CloudCompare_Instructions.md`
2. Ouvrir les fichiers PLY directement

## Légende des classes
- **0**: 100000000
- **1**: 303020000
- **2**: 303030204
- **3**: 303050500
- **4**: 304000000
- **5**: 304020000

## Codes couleur
- Chaque classe a une couleur unique générée automatiquement
- Les erreurs de prédiction sont affichées en rouge dans les fichiers de comparaison
- La propriété 'confidence' indique la certitude du modèle (0-1)

## Métriques
- Accuracy globale calculée par échantillon
- Confidence moyenne pour évaluer la certitude du modèle
- Fichiers de comparaison pour identifier visuellement les erreurs
