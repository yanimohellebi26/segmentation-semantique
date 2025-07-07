#  Segmentation 3D de Scènes Urbaines avec PointNet et PointNet++

Ce projet traite de la segmentation sémantique de scènes urbaines 3D à partir de fichiers `.ply` annotés, en utilisant deux réseaux de neurones : **PointNet** et **PointNet++**. Il inclut la préparation des données, l’entraînement, l’inférence, la visualisation et l’analyse.

## Organisation du projet

```
.
├── data/
│   ├── training/                           # Données .ply d'origine (Paris, Lille1, Lille2)
│   ├── blocks_dataset/                    # Blocs extraits pour entraînement
│   ├── test/                              # Données de test/inférence
├── objects_all/                           # Objets extraits par classe
├── objects_10classes/                     # Objets regroupés en 10 classes
├── reconstructed_scenes/                 # Scènes reconstruites par ville
├── final_dataset/                         # Données splitées en train/val/test
├── visualizations_real/                  # Résultats de visualisation
├── best_model_pointnet.pth               # Modèle PointNet entraîné
├── best_model_pointnet2.pth              # Modèle PointNet++ entraîné
└── *.py                                   # Scripts du pipeline
```

##  Étapes du pipeline

### 1.  Préparation des données

#### a. Extraction des objets par classe
- **Script :** `extract.py`
- **Entrée :** `.ply` + `.txt` d’annotations
- **Sortie :** `objects_all/*.npy` (objets individuels par label)

```bash
python extract.py
```

#### b. Regroupement en 10 classes sémantiques
- **Script :** `parse.py`
- **Entrée :** `objects_all/`
- **Sortie :** `objects_10classes/` avec classes unifiées

```bash
python parse.py
```

#### c. Reconstruction des scènes globales
- **Script :** `rebuild_scene.py`
- **Entrée :** `objects_all/`
- **Sortie :** `reconstructed_scenes/*.npz`

```bash
python rebuild_scene.py
```

#### d. Découpage en blocs et génération du dataset
- **Script :** `slice_blocks.py`
- **Entrée :** `reconstructed_scenes/`
- **Sortie :** `blocks_dataset/*.npz`

#### e. Split train / val / test
- **Script :** `prepare_final_dataset.py`
- **Entrée :** `blocks_dataset/`
- **Sortie :** `final_dataset/train|val|test/`

### 2.  Entraînement des modèles

#### a. Entraînement avec PointNet
- **Script :** `entrainement.py`
- **Modèle :** `best_model_pointnet.pth`

```bash
python entrainement.py
```

#### b. Entraînement avec PointNet++
- **Script :** `entrainement2.py`
- **Modèle :** `best_model_pointnet2.pth`

```bash
python entrainement2.py
```

### 3.  Inférence sur des fichiers .ply

- **Script :** `ply_inference.py`
- **Entrée :** fichier `.ply`
- **Sortie :** `.npy` des prédictions + `.ply` coloré

```bash
python ply_inference.py lhd.ply --model best_model_pointnet2.pth --knn
```

### 4. Visualisation et export des résultats

- **Script :** `visualize_pointnet2.py`
- **Entrée :** `lhd_infer.npz`
- **Sortie :** `.png`, `.ply`

```bash
python visualize_pointnet2.py
```

### 5.  Normalisation des fichiers PLY

- **Script :** `normalize_ply_files.py`
- **Entrée :** fichiers `.ply`
- **Sortie :** versions centrées et normalisées

```bash
python normalize_ply_files.py
```

##  Modèles utilisés

###  PointNet
- Pas de hiérarchie spatiale
- Plus rapide, moins précis

###  PointNet++
- Structure hiérarchique
- Plus précis pour scènes complexes

##  Fichiers principaux

| Étape                        | Script utilisé                 |
|-----------------------------|--------------------------------|
| Extraction objets           | `extract.py`                   |
| Regroupement classes        | `parse.py`                     |
| Reconstruction scènes       | `rebuild_scene.py`             |
| Génération blocs            | `slice_blocks.py`              |
| Split final train/val/test  | `prepare_final_dataset.py`     |
| Entraînement PointNet       | `entrainement.py`              |
| Entraînement PointNet++     | `entrainement2.py`             |
| Inference PLY               | `ply_inference.py`             |
| Visualisation résultats     | `visualize_pointnet2.py`       |
| Normalisation PLY           | `normalize_ply_files.py`       |

