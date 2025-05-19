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

def extract_playlist_id(playlist_url):
    import urllib.parse as urlparse
    parsed = urlparse.urlparse(playlist_url)
    query = urlparse.parse_qs(parsed.query)
    return query.get('list', [None])[0]

def get_videos_from_playlist(playlist_id):
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

def is_in_minio(bucket, object_name):
    client = get_minio_client()
    try:
        client.stat_object(bucket, object_name)
        return True
    except Exception:
        return False

def is_in_mongo(url):
    db = get_db()
    return db.downloads.find_one({"url": url}) is not None

def download_and_upload(url, output_dir, bucket, playlist_url):
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
