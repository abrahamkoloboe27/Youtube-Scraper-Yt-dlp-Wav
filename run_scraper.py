#!/usr/bin/env python3
"""
Script principal de scraping et traitement de playlists YouTube.

Ce script orchestre le processus complet de scraping, téléchargement, et upload
des vidéos YouTube vers MinIO et Azure Blob Storage, en utilisant les modules modulaires.

Usage:
    python run_scraper.py [--playlist_file PLAYLIST_FILE] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Import des modules
from scraper.youtube_client import YouTubeClient
from scraper.video_downloader import VideoDownloader
from scraper.playlist_scraper import PlaylistScraper
from minio_uploader.minio_client import MinioClient
from minio_uploader.minio_uploader import MinioUploader
from azure_uploader.azure_client import AzureClient
from azure_uploader.azure_uploader import AzureUploader
from storage_cleaner.storage_cleaner import StorageCleaner

# Import des utilitaires MongoDB
from mongo_utils import log_download, log_failed_download, get_db

# Configuration du logging
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

def parse_args():
    """
    Parse les arguments de la ligne de commande.
    
    Sorties :
        argparse.Namespace : Arguments parsés
    """
    parser = argparse.ArgumentParser(description="Scraping et traitement de playlists YouTube")
    
    parser.add_argument(
        "--playlist_file", 
        type=str, 
        default="playlist.txt",
        help="Fichier contenant les URLs des playlists YouTube (une par ligne)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="audios",
        help="Répertoire de sortie pour les fichiers audio"
    )
    
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=4,
        help="Nombre maximum de workers pour le téléchargement parallèle"
    )
    
    parser.add_argument(
        "--audio_format", 
        type=str, 
        default="wav",
        help="Format audio souhaité (wav, mp3, etc.)"
    )
    
    parser.add_argument(
        "--clean_after_upload", 
        action="store_true",
        help="Nettoyer les fichiers locaux et MinIO après upload vers Azure"
    )
    
    return parser.parse_args()

def check_exists(url: str) -> bool:
    """
    Vérifie si une vidéo a déjà été téléchargée (présence dans MongoDB et MinIO).
    
    Entrées :
        url (str) : URL de la vidéo
    Sorties :
        bool : True si la vidéo est déjà traitée, False sinon
    """
    # Vérifier dans MongoDB
    db = get_db()
    video_id = url.split('=')[-1]
    result = db.downloads.find_one({"url": url})
    
    if result:
        logging.info(f"Vidéo déjà traitée (présente dans MongoDB): {url}")
        return True
    
    # Vérifier dans MinIO
    minio_client = MinioClient()
    bucket = os.getenv("MINIO_BUCKET", "audios")
    object_name = f"{video_id}.wav"
    
    if minio_client.object_exists(bucket, object_name):
        logging.info(f"Vidéo déjà traitée (présente dans MinIO): {url}")
        return True
    
    return False

def process_video_result(result: Dict[str, Any]) -> None:
    """
    Traite le résultat du téléchargement d'une vidéo.
    
    Cette fonction gère l'upload vers MinIO et Azure, ainsi que le nettoyage
    et la journalisation dans MongoDB.
    
    Entrées :
        result (Dict[str, Any]) : Résultat du téléchargement
    """
    if not result["success"]:
        # Journaliser l'échec
        log_failed_download({
            "url": result["url"],
            "error": result.get("error", "Erreur inconnue"),
            "error_type": result.get("error_type", "download_error")
        })
        return
    
    # Récupérer les informations
    file_path = result["file_path"]
    metadata = result["metadata"]
    url = metadata["url"]
    playlist_url = result.get("playlist_url", "")
    
    # Upload vers MinIO
    minio_uploader = MinioUploader()
    bucket = os.getenv("MINIO_BUCKET", "audios")
    minio_result = minio_uploader.upload_file(file_path, bucket)
    
    if not minio_result["success"]:
        log_failed_download({
            "url": url,
            "error": f"Échec de l'upload vers MinIO: {minio_result.get('error', 'Erreur inconnue')}",
            "error_type": "minio_upload_error"
        })
        return
    
    # Upload vers Azure
    azure_uploader = AzureUploader()
    azure_container = os.getenv("AZURE_CONTAINER_NAME", "audios")
    azure_result = azure_uploader.upload_file(file_path, azure_container)
    
    if not azure_result["success"]:
        log_failed_download({
            "url": url,
            "error": f"Échec de l'upload vers Azure: {azure_result.get('error', 'Erreur inconnue')}",
            "error_type": "azure_upload_error"
        })
        return
    
    # Nettoyer les fichiers si demandé
    if args.clean_after_upload:
        storage_cleaner = StorageCleaner()
        storage_cleaner.clean_after_upload(file_path)
    else:
        # Supprimer uniquement le fichier local
        try:
            os.remove(file_path)
            logging.info(f"Fichier local supprimé: {file_path}")
        except Exception as e:
            logging.error(f"Erreur lors de la suppression du fichier local {file_path}: {e}")
    
    # Journaliser le succès
    log_download({
        "url": url,
        "file": os.path.basename(file_path),
        "status": "success",
        "duration": metadata.get("duration", 0),
        "filesize": metadata.get("filesize", 0),
        "playlist_url": playlist_url,
        "azure_path": f"{azure_container}/{os.path.basename(file_path)}"
    })
    
    logging.info(f"Audio traité avec succès: {file_path}")

def main():
    """
    Fonction principale du script.
    """
    # Charger les variables d'environnement
    load_dotenv()
    
    # Vérifier que le fichier de playlists existe
    playlist_file = args.playlist_file
    if not os.path.exists(playlist_file):
        logging.error(f"Le fichier de playlists n'existe pas: {playlist_file}")
        sys.exit(1)
    
    # Créer le répertoire de sortie
    output_dir = args.output_dir
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialiser les clients
    youtube_client = YouTubeClient()
    video_downloader = VideoDownloader(
        output_dir=output_dir,
        audio_format=args.audio_format
    )
    
    # Initialiser le scraper
    playlist_scraper = PlaylistScraper(
        youtube_client=youtube_client,
        video_downloader=video_downloader,
        max_workers=args.max_workers,
        max_cookie_failures=MAX_COOKIE_FAILURES,
        check_exists_callback=check_exists
    )
    
    # Traiter les playlists
    stats = playlist_scraper.process_playlists_from_file(
        playlist_file=playlist_file,
        callback=process_video_result
    )
    
    # Afficher les statistiques
    logging.info("Traitement terminé")
    logging.info(f"Statistiques finales: {stats}")

if __name__ == "__main__":
    # Parser les arguments
    args = parse_args()
    
    # Exécuter la fonction principale
    main()
