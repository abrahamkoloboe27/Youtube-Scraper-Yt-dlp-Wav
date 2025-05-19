import logging
import os
import time
from pathlib import Path
from mongo_utils import get_failed_downloads, log_download, log_failed_download
from minio_utils import upload_file, get_minio_client, delete_file
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import yt_dlp

load_dotenv()

# Constantes
MAX_COOKIE_FAILURES = 100
COOKIE_TIMEOUT = 300  # 5 minutes en secondes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/retry.log'),
        logging.StreamHandler()
    ]
)

def get_azure_client():
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    return BlobServiceClient.from_connection_string(connection_string)

def upload_to_azure(file_path, container_name):
    try:
        blob_service_client = get_azure_client()
        container_client = blob_service_client.get_container_client(container_name)
        blob_name = os.path.basename(file_path)
        with open(file_path, "rb") as data:
            container_client.upload_blob(name=blob_name, data=data, overwrite=True)
        return True
    except Exception as e:
        logging.error(f"Erreur lors de l'upload vers Azure: {e}")
        return False

def retry_download_and_upload(url, output_dir, bucket, playlist_url=None):
    try:
        if not os.path.exists('cookies.txt'):
            logging.error("Le fichier cookies.txt est introuvable. Pause de 5 minutes pour intervention manuelle.")
            time.sleep(COOKIE_TIMEOUT)
            return False

        output_path = Path(output_dir) / f"{url.split('=')[-1]}.mp3"
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': str(output_path.with_suffix('')),
            'cookiefile': 'cookies.txt',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logging.info(f"Téléchargement de la vidéo: {url}")
            info = ydl.extract_info(url, download=True)
            
            # Récupérer les métadonnées
            duration = info.get('duration', 0)
            filesize = os.path.getsize(output_path)
            
            # Upload vers MinIO
            upload_file(str(output_path), bucket, output_path.name)
            
            # Upload vers Azure
            azure_container = os.getenv("AZURE_CONTAINER_NAME", "audios")
            if upload_to_azure(str(output_path), azure_container):
                # Supprimer le fichier local
                os.remove(output_path)
                # Supprimer de MinIO
                delete_file(bucket, output_path.name)
                
                # Enregistrer dans MongoDB
                log_download({
                    "url": url,
                    "file": output_path.name,
                    "status": "success",
                    "duration": duration,
                    "filesize": filesize,
                    "playlist_url": playlist_url,
                    "azure_path": f"{azure_container}/{output_path.name}"
                })
                logging.info(f"Audio traité avec succès: {output_path.name}")
                return True
            else:
                return False

    except Exception as e:
        if "cookies" in str(e).lower():
            log_failed_download({"url": url, "error": str(e), "type": "cookie_error"})
        else:
            log_failed_download({"url": url, "error": str(e)})
        logging.error(f"Erreur lors du traitement de {url}: {e}")
        return False

def main():
    output_dir = "audios"
    bucket = os.getenv("MINIO_BUCKET", "audios")
    Path(output_dir).mkdir(exist_ok=True)
    
    cookie_failures = 0
    failed = get_failed_downloads()
    
    for entry in failed:
        if cookie_failures >= MAX_COOKIE_FAILURES:
            logging.error(f"Nombre maximum d'échecs liés aux cookies atteint ({MAX_COOKIE_FAILURES}). Arrêt du script.")
            break
            
        url = entry.get("url")
        playlist_url = entry.get("playlist_url")
        
        if url:
            logging.info(f"Retry téléchargement: {url}")
            success = retry_download_and_upload(url, output_dir, bucket, playlist_url)
            if not success:
                cookie_failures += 1

if __name__ == "__main__":
    main()
