# Script Blender pour importer et visualiser la segmentation PointNet++
import bpy
import bmesh
import os

def clear_scene():
    """Nettoie la scène Blender"""
    # Suppression de tous les objets
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Suppression des matériaux orphelins
    for material in bpy.data.materials:
        if material.users == 0:
            bpy.data.materials.remove(material)

def import_ply_file():
    """Importe le fichier PLY avec la segmentation"""
    ply_filepath = r"ply_exports/sample_01_predictions.ply"
    
    if not os.path.exists(ply_filepath):
        print(f"Erreur: Fichier PLY non trouvé: {ply_filepath}")
        return None
    
    # Import du PLY
    bpy.ops.import_mesh.ply(filepath=ply_filepath)
    
    # Récupération de l'objet importé
    imported_obj = bpy.context.active_object
    if imported_obj:
        imported_obj.name = "PointNet2_Segmentation"
        print(f"PLY importé avec succès: {len(imported_obj.data.vertices)} points")
    
    return imported_obj

def setup_visualization():
    """Configure la visualisation optimale"""
    # Configuration de la vue 3D
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    # Mode solide avec couleurs vertex
                    space.shading.type = 'SOLID'
                    space.shading.color_type = 'VERTEX'
                    
                    # Configuration de l'affichage des points
                    space.overlay.show_wireframes = True
                    space.overlay.wireframe_threshold = 1.0
    
    # Ajout d'un éclairage
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 3.0

def create_class_legend():
    """Crée une légende des classes"""
    class_mapping = {0: np.int64(100000000), 1: np.int64(303020000), 2: np.int64(303030204), 3: np.int64(303050500), 4: np.int64(304000000), 5: np.int64(304020000)}
    
    # Création d'un objet texte pour la légende
    bpy.ops.object.text_add(location=(0, 0, 5))
    text_obj = bpy.context.active_object
    text_obj.name = "Class_Legend"
    
    # Construction du texte de légende
    legend_text = "Segmentation PointNet++\nClasses:\n"
    for class_id, class_name in class_mapping.items():
        legend_text += f"{class_id}: {class_name}\n"
    
    text_obj.data.body = legend_text
    text_obj.data.size = 0.5

def main():
    """Fonction principale d'import et configuration"""
    print("Import de la segmentation PointNet++ dans Blender")
    
    # Nettoyage de la scène
    clear_scene()
    
    # Import du PLY
    obj = import_ply_file()
    if obj is None:
        print("Échec de l'import")
        return
    
    # Configuration de la visualisation
    setup_visualization()
    
    # Création de la légende
    create_class_legend()
    
    # Centrage de la vue sur l'objet
    bpy.ops.view3d.view_selected()
    
    print("Import et configuration terminés")
    print("Utilisez les contrôles de la souris pour naviguer:")
    print("- Molette: Zoom")
    print("- Molette + glisser: Rotation")
    print("- Shift + Molette + glisser: Translation")

# Exécution du script
if __name__ == "__main__":
    main()
