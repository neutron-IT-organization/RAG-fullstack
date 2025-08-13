# download_models.py
# Script utilitaire pour télécharger les modèles IA depuis MinIO s'ils n'existent pas localement.

import os

os.environ["AWS_S3_ENDPOINT"] = "192.168.0.150:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "EdKvYHNxP5IhbeVjhmQb"
os.environ["AWS_SECRET_ACCESS_KEY"] = "NnLOf0hFDPUWQ6ez2JajPLD75mFsGqdO0LrwhlcM"

import sys
from minio import Minio
from minio.error import S3Error

# --- Configuration ---

# Chemin de base où le PVC est monté dans le conteneur.
MODELS_BASE_PATH = "/app/models"

# Lire les informations de connexion depuis les variables d'environnement injectées par le Secret
MINIO_BUCKET_NAME = "reda-rag"
S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Dictionnaire décrivant les modèles à télécharger
# LES CHEMINS SONT MAINTENANT ABSOLUS ET POINTENT VERS LE PVC
MODELS_TO_DOWNLOAD = {
    "Meta-Llama-3.2-3B-Instruct/": os.path.join(MODELS_BASE_PATH, "Meta-Llama-3.2-3B-Instruct"),
    "all-MiniLM-L6-v2/": os.path.join(MODELS_BASE_PATH, "all-MiniLM-L6-v2")
}

# --- Logique de Téléchargement ---

def download_files_from_minio(client, bucket, prefix, local_path):
    """
    Télécharge tous les fichiers d'un préfixe MinIO vers un dossier local.
    Ignore le téléchargement si le dossier local existe déjà et n'est pas vide.
    """
    if os.path.exists(local_path) and os.listdir(local_path):
        print(f" Le dossier '{local_path}' existe et n'est pas vide. Téléchargement ignoré.")
        return

    print(f"-> Le dossier '{local_path}' est manquant ou vide. Début du téléchargement depuis MinIO (prefix: '{prefix}')...")
    os.makedirs(local_path, exist_ok=True)
    
    try:
        objects = client.list_objects(bucket, prefix=prefix, recursive=True)
        files_to_download = [obj for obj in objects if not obj.object_name.endswith('/')]

        if not files_to_download:
            print(f"   AVERTISSEMENT : Aucun fichier trouvé sur MinIO avec le préfixe '{prefix}'.")
            return

        for obj in files_to_download:
            relative_path = os.path.relpath(obj.object_name, prefix)
            local_file_path = os.path.join(local_path, relative_path)
            
            if not os.path.exists(os.path.dirname(local_file_path)):
                os.makedirs(os.path.dirname(local_file_path))
            
            print(f"   Téléchargement de '{obj.object_name}'...")
            client.fget_object(bucket, obj.object_name, local_file_path)
        
        print(f"  Téléchargement pour '{local_path}' terminé.")

    except S3Error as exc:
        print(f"  Une erreur S3 est survenue pour {prefix}: {exc}")
        raise

# --- Exécution du Script (CORRIGÉ) ---
# La fonction main() et l'appel sont maintenant au niveau principal du script,
# avec la bonne indentation.

def main():
    """
    Fonction principale qui orchestre la connexion et le téléchargement.
    """
    print("--- Démarrage du script de préparation des modèles ---")
    
    if not all([S3_ENDPOINT, ACCESS_KEY, SECRET_KEY]):
        print("ERREUR: Variables d'environnement MinIO (AWS_S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) sont manquantes.")
        sys.exit(1)
    
    try:
        print(f"Connexion à MinIO sur l'endpoint: {S3_ENDPOINT}")
        minio_client = Minio(S3_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)

        for prefix, local_path in MODELS_TO_DOWNLOAD.items():
            download_files_from_minio(minio_client, MINIO_BUCKET_NAME, prefix, local_path)

        print("--- Tous les modèles sont prêts. ---")

    except Exception as exc:
        print(f"Une erreur inattendue est survenue : {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()