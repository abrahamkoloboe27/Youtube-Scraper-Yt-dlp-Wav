"""
Script principal de scraping et de traitement de playlists YouTube.

Ce script lit une liste de playlists YouTube, extrait les vidéos, télécharge l'audio de chaque vidéo,
et téléverse les fichiers audio sur MinIO et Azure Blob Storage. Il journalise les succès et échecs dans MongoDB.
Le téléchargement et le traitement sont parallélisés pour accélérer le processus.

Fonctions principales :
- get_azure_client : Retourne un client Azure Blob Storage.
- upload_to_azure : Téléverse un fichier sur Azure Blob Storage.
- extract_playlist_id : Extrait l'ID d'une playlist à partir de son URL.
- get_videos_from_playlist : Récupère toutes les URLs vidéos d'une playlist YouTube.
- is_in_minio : Vérifie si un objet est déjà dans MinIO.
- is_in_mongo : Vérifie si une vidéo est déjà journalisée dans MongoDB.
- download_and_upload : Télécharge l'audio d'une vidéo et l'upload vers les stockages, journalise le résultat.
- main : Parcourt les playlists et orchestre le traitement parallèle.

Usage :
Exécuter ce script pour traiter automatiquement une ou plusieurs playlists YouTube listées dans playlist.txt.
"""
import os
import logging
from dotenv import load_dotenv
from mongo_utils import log_download, log_failed_download, get_db
from minio_utils import upload_file, get_minio_client, delete_file
import yt_dlp
import requests
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from azure.storage.blob import BlobServiceClient
import json

load_dotenv()
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper.log'),
        logging.StreamHandler()
    ]
)

# Constantes
MAX_COOKIE_FAILURES = 100
COOKIE_TIMEOUT = 300  # 5 minutes en secondes

def get_azure_client():
    """
    Retourne un client Azure Blob Storage à partir de la variable d'environnement AZURE_STORAGE_CONNECTION_STRING.

    Entrées :
        Néant (utilise la variable d'environnement AZURE_STORAGE_CONNECTION_STRING)
    Sorties :
        BlobServiceClient : client Azure prêt à l'emploi
    """
    account_url = os.getenv("AZURE_ACCOUNT_URL")
    sas_token = os.getenv("AZURE_SAS_TOKEN")
    return BlobServiceClient(account_url=account_url, credential=sas_token)


def upload_to_azure(file_path: str, container_name: str) -> bool:
    """
    Téléverse un fichier local sur Azure Blob Storage dans le conteneur spécifié.

    Entrées :
        file_path (str) : chemin du fichier local à téléverser
        container_name (str) : nom du conteneur Azure cible
    Sorties :
        bool : True si succès, False sinon
    """
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

def extract_playlist_id(playlist_url: str) -> str:
    """
    Extrait l'identifiant d'une playlist YouTube à partir de son URL.

    Entrées :
        playlist_url (str) : URL de la playlist YouTube
    Sorties :
        str : ID de la playlist (ou None si non trouvé)
    """
    import urllib.parse as urlparse
    parsed = urlparse.urlparse(playlist_url)
    query = urlparse.parse_qs(parsed.query)
    return query.get('list', [None])[0]

def get_videos_from_playlist(playlist_id: str) -> list:
    """
    Récupère toutes les URLs des vidéos d'une playlist YouTube via l'API YouTube Data.

    Entrées :
        playlist_id (str) : identifiant de la playlist YouTube
    Sorties :
        list : liste des URLs des vidéos de la playlist
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    base_url = "https://www.googleapis.com/youtube/v3/playlistItems"
    video_urls = []
    page_token = ""
    while True:
        params = {
            "part": "snippet",
            "playlistId": playlist_id,
            "maxResults": 50,
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        resp = requests.get(base_url, params=params)
        data = resp.json()
        for item in data.get("items", []):
            video_id = item["snippet"]["resourceId"]["videoId"]
            video_urls.append(f"https://www.youtube.com/watch?v={video_id}")
        page_token = data.get("nextPageToken", "")
        if not page_token:
            break
    return video_urls

def is_in_minio(bucket: str, object_name: str) -> bool:
    """
    Vérifie si un objet existe déjà dans un bucket MinIO.

    Entrées :
        bucket (str) : nom du bucket
        object_name (str) : nom de l'objet à vérifier
    Sorties :
        bool : True si l'objet existe, False sinon
    """
    client = get_minio_client()
    try:
        client.stat_object(bucket, object_name)
        return True
    except Exception:
        return False

def is_in_mongo(url: str) -> bool:
    """
    Vérifie si une vidéo a déjà été téléchargée (présence dans MongoDB).

    Entrées :
        url (str) : URL de la vidéo
    Sorties :
        bool : True si la vidéo est déjà journalisée, False sinon
    """
    db = get_db()
    return db.downloads.find_one({"url": url}) is not None

def download_and_upload(url: str, output_dir: str, bucket: str, playlist_url: str) -> bool:
    """
    Télécharge l'audio d'une vidéo YouTube, l'upload sur MinIO et Azure, puis journalise le résultat.
    Supprime le fichier local et de MinIO si l'upload Azure réussit.

    Entrées :
        url (str) : URL de la vidéo à traiter
        output_dir (str) : dossier de sortie pour l'audio
        bucket (str) : nom du bucket MinIO
        playlist_url (str) : URL de la playlist d'origine
    Sorties :
        bool : True si tout s'est bien passé, False sinon
    """
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

def main() -> None:
    """
    Fonction principale : lit les playlists, extrait les vidéos, et orchestre le téléchargement/traitement en parallèle.

    Entrées :
        Néant (utilise playlist.txt et les variables d'environnement)
    Sorties :
        None
    """
    output_dir = "audios"
    bucket = os.getenv("MINIO_BUCKET", "audios")
    Path(output_dir).mkdir(exist_ok=True)
    
    cookie_failures = 0
    
    with open("playlist.txt") as f:
        playlists = [line.strip() for line in f if line.strip()]
    
    for playlist_url in playlists:
        if cookie_failures >= MAX_COOKIE_FAILURES:
            logging.error(f"Nombre maximum d'échecs liés aux cookies atteint ({MAX_COOKIE_FAILURES}). Arrêt du script.")
            break
            
        playlist_id = extract_playlist_id(playlist_url)
        if not playlist_id:
            logging.error(f"Impossible d'extraire l'ID de la playlist: {playlist_url}")
            continue
            
        video_urls = get_videos_from_playlist(playlist_id)
        tasks = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            for url in video_urls:
                object_name = f"{url.split('=')[-1]}.mp3"
                if is_in_minio(bucket, object_name) and is_in_mongo(url):
                    logging.info(f"Vidéo déjà traitée (présente dans MinIO et MongoDB): {url}")
                    continue
                    
                tasks.append(executor.submit(download_and_upload, url, output_dir, bucket, playlist_url))
                
            for future in as_completed(tasks):
                try:
                    success = future.result()
                    if not success:
                        cookie_failures += 1
                except Exception as e:
                    logging.error(f"Erreur dans un téléchargement parallèle: {e}")
                    cookie_failures += 1

if __name__ == "__main__":
    main()
