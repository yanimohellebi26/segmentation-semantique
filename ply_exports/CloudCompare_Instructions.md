# Visualisation de la segmentation PointNet++ dans CloudCompare

## Fichier à ouvrir
```
ply_exports/sample_01_predictions.ply
```

## Instructions d'ouverture

1. **Lancer CloudCompare**
   - Ouvrir CloudCompare
   - File > Open > Sélectionner le fichier PLY ci-dessus

2. **Configuration de l'affichage**
   - Clic droit sur l'objet dans la liste DB Tree
   - Properties > Colors > RGB
   - Ajuster la taille des points: Properties > Point Size (recommandé: 2-4)

3. **Visualisation par classes**
   - Dans la barre d'outils: cliquer sur l'icône "Color scale"
   - Sélectionner "Scalar Field" > "class"
   - Choisir une palette de couleurs appropriée

4. **Navigation**
   - Molette souris: Zoom
   - Clic gauche + glisser: Rotation
   - Clic droit + glisser: Translation
   - F: Vue de face, G: Vue de gauche, T: Vue du dessus

## Légende des classes
- **0**: 100000000
- **1**: 303020000
- **2**: 303030204
- **3**: 303050500
- **4**: 304000000
- **5**: 304020000

## Fonctionnalités avancées

1. **Analyse statistique**
   - Tools > Statistics > Compute stats on 'class' field
   - Affiche la répartition des points par classe

2. **Filtrage par classe**
   - Sélectionner l'objet
   - Edit > Scalar fields > Filter by value
   - Choisir la plage de classes à afficher

3. **Export d'images**
   - Display > Render to file
   - Choisir la résolution et le format

4. **Mesures**
   - Tools > Point picking: pour sélectionner des points spécifiques
   - Tools > Point pair registration: pour aligner avec d'autres nuages

## Analyse de la qualité de segmentation

- Zoom sur différentes zones pour évaluer la cohérence
- Vérifier les transitions entre classes
- Identifier les zones d'erreur potentielles
