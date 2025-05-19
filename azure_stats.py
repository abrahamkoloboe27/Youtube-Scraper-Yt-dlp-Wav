import os
import logging
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
from dotenv import load_dotenv
from datetime import datetime
import urllib.parse

load_dotenv(".env")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/azure_stats.log'),
        logging.StreamHandler()
    ]
)

def get_azure_client():
    account_url = os.getenv("AZURE_ACCOUNT_URL")
    sas_token = os.getenv("AZURE_SAS_TOKEN")
    
    if not account_url or not sas_token:
        raise ValueError("AZURE_ACCOUNT_URL et AZURE_SAS_TOKEN doivent être définis dans le fichier .env")
    
    # Vérifier le format de l'URL
    if not account_url.startswith('https://'):
        account_url = f'https://{account_url}'
    
    # Vérifier le format du SAS token
    if not sas_token.startswith('?'):
        sas_token = f'?{sas_token}'
    
    logging.info(f"Tentative de connexion à Azure Blob Storage: {account_url}")
    return BlobServiceClient(account_url=account_url, credential=sas_token)

def format_size(size_bytes):
    """Convertit les octets en format lisible"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def get_container_stats():
    try:
        blob_service_client = get_azure_client()
        container_name = os.getenv("AZURE_CONTAINER_NAME", "audios")
        
        logging.info(f"Vérification du conteneur: {container_name}")
        container_client = blob_service_client.get_container_client(container_name)

        # Vérifier si le conteneur existe
        try:
            container_properties = container_client.get_container_properties()
            logging.info(f"Conteneur trouvé. Créé le: {container_properties.creation_time}")
        except ResourceNotFoundError:
            logging.error(f"Le conteneur {container_name} n'existe pas")
            return
        except HttpResponseError as e:
            if e.status_code == 403:
                logging.error("Erreur d'autorisation. Vérifiez que votre SAS token a les permissions suivantes:")
                logging.error("- Read (r)")
                logging.error("- List (l)")
                logging.error("Le SAS token doit être généré avec ces permissions pour le conteneur")
            else:
                logging.error(f"Erreur lors de l'accès au conteneur: {e}")
            return

        # Récupérer les statistiques du conteneur
        total_size = 0
        file_count = 0
        files_by_type = {}

        # Lister tous les blobs
        try:
            blobs = container_client.list_blobs()
            for blob in blobs:
                file_count += 1
                total_size += blob.size

                # Compter les fichiers par type
                file_extension = os.path.splitext(blob.name)[1].lower()
                files_by_type[file_extension] = files_by_type.get(file_extension, 0) + 1

                # Afficher les détails de chaque fichier
                last_modified = blob.last_modified.strftime("%Y-%m-%d %H:%M:%S")
                logging.info(f"Fichier: {blob.name}")
                logging.info(f"  Taille: {format_size(blob.size)}")
                logging.info(f"  Dernière modification: {last_modified}")
                logging.info(f"  Type: {blob.content_type}")
                logging.info("---")

            # Afficher les statistiques globales
            logging.info("\nStatistiques du conteneur:")
            logging.info(f"Nombre total de fichiers: {file_count}")
            logging.info(f"Espace total utilisé: {format_size(total_size)}")
            
            # Afficher la répartition par type de fichier
            logging.info("\nRépartition par type de fichier:")
            for ext, count in files_by_type.items():
                logging.info(f"{ext or 'sans extension'}: {count} fichiers")

        except HttpResponseError as e:
            if e.status_code == 403:
                logging.error("Erreur d'autorisation lors de la lecture des fichiers. Vérifiez les permissions du SAS token.")
            else:
                logging.error(f"Erreur lors de la lecture des fichiers: {e}")

    except Exception as e:
        logging.error(f"Erreur lors de la récupération des statistiques: {e}")

if __name__ == "__main__":
    get_container_stats() 