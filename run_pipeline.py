#!/usr/bin/env python3
"""
Script principal de pipeline complet pour le traitement des audios Fongbè.

Ce script orchestre l'ensemble du processus :
1. Téléchargement des vidéos YouTube à partir de playlists
2. Upload vers MinIO et Azure
3. Traitement audio (normalisation, segmentation, nettoyage, etc.)
4. Upload vers Hugging Face
5. Nettoyage des espaces de stockage

Usage:
    python run_pipeline.py [--playlist_file PLAYLIST_FILE] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Import des modules de scraping
from scraper.youtube_client import YouTubeClient
from scraper.video_downloader import VideoDownloader
from scraper.playlist_scraper import PlaylistScraper

# Import des modules d'upload
from minio_uploader.minio_client import MinioClient
from minio_uploader.minio_uploader import MinioUploader
from azure_uploader.azure_client import AzureClient
from azure_uploader.azure_uploader import AzureUploader
from upload_hf.upload_manager import UploadManager
from upload_hf.mongo_logger import HFMongoLogger

# Import des modules de traitement audio
from processing.audio_loader import AudioLoader
from processing.loudness_normalizer import LoudnessNormalizer
from processing.silence_remover import SilenceRemover
from processing.diarization import Diarization
from processing.segmentation import Segmentation
from processing.audio_cleaner import AudioCleaner
from processing.metadata_manager import MetadataManager
from processing.data_augmentation import DataAugmentation
from processing.quality_checker import QualityChecker
from processing.pipeline import AudioPipeline

# Import du module de nettoyage
from storage_cleaner.storage_cleaner import StorageCleaner

# Import des utilitaires MongoDB
from mongo_utils import log_download, log_failed_download, get_db

# Configuration du logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ││ %(levelname)s ││ %(name)s ││ %(message)s ",
    datefmt="%Y-%m-%d %H:%M:%S",

    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)


from transformers import logging as hf_logging
hf_logging.set_verbosity_info()       

# Constantes
MAX_COOKIE_FAILURES = 100
MAX_WORKERS = 4

def parse_args():
    """
    Parse les arguments de la ligne de commande.
    
    Sorties :
        argparse.Namespace : Arguments parsés
    """
    parser = argparse.ArgumentParser(description="Pipeline complet pour le traitement des audios Fongbè")
    
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
        help="Répertoire de sortie pour les fichiers audio bruts"
    )
    
    parser.add_argument(
        "--processed_dir", 
        type=str, 
        default="processed",
        help="Répertoire pour les fichiers audio traités"
    )
    
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=4,
        help="Nombre maximum de workers pour le traitement parallèle"
    )
    
    parser.add_argument(
        "--skip_download", 
        action="store_true",
        help="Sauter l'étape de téléchargement (utiliser les fichiers existants)"
    )
    
    parser.add_argument(
        "--skip_processing", 
        action="store_true",
        help="Sauter l'étape de traitement audio"
    )
    
    parser.add_argument(
        "--skip_hf_upload", 
        action="store_true",
        help="Sauter l'étape d'upload vers Hugging Face"
    )
    
    parser.add_argument(
        "--skip_cleaning", 
        action="store_true",
        help="Sauter l'étape de nettoyage des espaces de stockage"
    )
    
    parser.add_argument(
        "--hf_repo_id", 
        type=str, 
        default=None,
        help="ID du dépôt Hugging Face (format: 'username/repo_name')"
    )
    
    return parser.parse_args()

def check_exists(url: str) -> Dict[str, Any]:
    """
    Vérifie si une vidéo a déjà été téléchargée (présence dans MongoDB et MinIO).
    
    Entrées :
        url (str) : URL de la vidéo
    Sorties :
        Dict[str, Any] : Informations sur l'existence de la vidéo
        {
            "exists": bool,  # True si la vidéo existe déjà
            "file_path": str,  # Chemin du fichier si trouvé
            "video_id": str,  # ID de la vidéo YouTube
            "source": str,  # Source de l'information ("mongodb", "minio", ou None)
            "metadata": dict  # Métadonnées si disponibles
        }
    """
    video_id = url.split('=')[-1]
    result = {
        "exists": False,
        "file_path": None,
        "video_id": video_id,
        "source": None,
        "metadata": {}
    }
    
    # Vérifier dans MongoDB
    db = get_db()
    mongo_result = db.downloads.find_one({"url": url})
    
    if mongo_result:
        logging.info(f"Vidéo trouvée dans MongoDB: {url}")
        result["exists"] = True
        result["source"] = "mongodb"
        result["file_path"] = mongo_result.get("file_path")
        result["metadata"] = mongo_result
        return result
    
    # Vérifier dans MinIO
    minio_client = MinioClient()
    bucket = os.getenv("MINIO_BUCKET", "audios")
    object_name = f"{video_id}.wav"
    
    if minio_client.object_exists(bucket, object_name):
        logging.info(f"Vidéo trouvée dans MinIO: {url}")
        result["exists"] = True
        result["source"] = "minio"
        result["file_path"] = f"audios/{video_id}.wav"  # Chemin local présumé
        return result
    
    return result

def download_and_upload(url: str, output_dir: str, playlist_url: str) -> Dict[str, Any]:
    """
    Télécharge une vidéo YouTube et l'upload vers MinIO et Azure.
    
    Entrées :
        url (str) : URL de la vidéo YouTube
        output_dir (str) : Répertoire de sortie
        playlist_url (str) : URL de la playlist d'origine
    Sorties :
        Dict[str, Any] : Résultat du téléchargement et de l'upload
    """
    # Initialiser les clients
    video_downloader = VideoDownloader(output_dir=output_dir, cookie_file="cookies.txt" )
    minio_uploader = MinioUploader()
    azure_uploader = AzureUploader()
    
    # Télécharger la vidéo
    download_result = video_downloader.download_audio(url)
    
    if not download_result["success"]:
        # Journaliser l'échec
        log_failed_download({
            "url": url,
            "error": download_result.get("error", "Erreur inconnue"),
            "error_type": download_result.get("error_type", "download_error"),
            "playlist_url": playlist_url
        })
        return download_result
    
    # Récupérer les informations
    file_path = download_result["file_path"]
    metadata = download_result["metadata"]
    
    # Upload vers MinIO
    minio_result = minio_uploader.upload_file(file_path)
    
    if not minio_result["success"]:
        log_failed_download({
            "url": url,
            "error": f"Échec de l'upload vers MinIO: {minio_result.get('error', 'Erreur inconnue')}",
            "error_type": "minio_upload_error",
            "playlist_url": playlist_url
        })
        return {**download_result, "minio_result": minio_result, "success": False}
    
    # Upload vers Azure (ne pas échouer si l'upload Azure échoue)
    azure_container = os.getenv("AZURE_CONTAINER_NAME", "audios")
    azure_result = azure_uploader.upload_file(file_path, azure_container)
    
    # Journaliser le succès
    log_download({
        "url": url,
        "file": os.path.basename(file_path),
        "status": "success",
        "title": metadata.get("title", ""),  # Ajouter le titre de la vidéo
        "duration": metadata.get("duration", 0),
        "filesize": metadata.get("filesize", 0),
        "playlist_url": playlist_url,
        "azure_path": f"{azure_container}/{os.path.basename(file_path)}" if azure_result["success"] else None,
        "minio_path": f"{minio_result['bucket']}/{minio_result['object_name']}",
        "metadata": metadata  # Inclure toutes les métadonnées originales
    })
    
    logging.info(f"Audio téléchargé et uploadé avec succès: {file_path}")
    
    return {
        **download_result,
        "minio_result": minio_result,
        "azure_result": azure_result,
        "success": True
    }

def process_audio(file_path: str, processed_dir: str) -> Dict[str, Any]:
    """
    Traite un fichier audio avec le pipeline de traitement.
    
    Entrées :
        file_path (str) : Chemin du fichier audio à traiter
        processed_dir (str) : Répertoire pour les fichiers traités
    Sorties :
        Dict[str, Any] : Résultat du traitement
    """
    try:
        # Créer le répertoire de sortie
        Path(processed_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialiser les composants du pipeline
        audio_loader = AudioLoader()
        loudness_normalizer = LoudnessNormalizer()
        silence_remover = SilenceRemover()
        diarizer = Diarization()
        segmenter = Segmentation()
        audio_cleaner = AudioCleaner()
        metadata_manager = MetadataManager(output_dir=processed_dir)
        data_augmenter = DataAugmentation()
        quality_checker = QualityChecker()
        
        # Initialiser le pipeline avec le répertoire de sortie
        pipeline = AudioPipeline(
            base_output_dir=processed_dir
        )
        
        # Note: Les composants sont initialisés automatiquement dans la classe AudioPipeline
        
        # Traiter le fichier audio
        result = pipeline.process_file(file_path)
        
        if result["success"]:
            logging.info(f"Audio traité avec succès: {file_path}")
        else:
            logging.error(f"Échec du traitement audio: {file_path} - {result.get('error', 'Erreur inconnue')}")
        
        return result
        
    except Exception as e:
        logging.error(f"Erreur lors du traitement audio: {file_path} - {e}")
        return {"success": False, "error": str(e), "file_path": file_path}

def upload_to_huggingface(processed_dir: str, repo_id: str) -> Dict[str, Any]:
    """
    Upload les fichiers traités vers Hugging Face.
    
    Entrées :
        processed_dir (str) : Répertoire contenant les fichiers traités
        repo_id (str) : ID du dépôt Hugging Face
    Sorties :
        Dict[str, Any] : Résultat de l'upload
    """
    try:
        # Initialiser le logger MongoDB
        mongo_logger = HFMongoLogger()
        
        # Initialiser le gestionnaire d'upload
        upload_manager = UploadManager(
            repo_id=repo_id,
            dataset_name="fongbe_audio",
            dataset_language="fon",
            local_dir=processed_dir,
            mongo_logger=mongo_logger
        )
        
        # Chemins des fichiers
        metadata_file = Path(processed_dir) / "metadata" / "metadata_all.csv"
        audio_dir = Path(processed_dir) / "final"
        
        # Vérifier que les fichiers existent
        if not metadata_file.exists():
            raise FileNotFoundError(f"Fichier de métadonnées non trouvé: {metadata_file}")
        
        if not audio_dir.exists() or not audio_dir.is_dir():
            raise FileNotFoundError(f"Répertoire audio non trouvé: {audio_dir}")
        
        # Upload vers Hugging Face
        result = upload_manager.run_full_upload(
            metadata_file=str(metadata_file),
            audio_dir=str(audio_dir),
            format_metadata=True,
            prepare_local=True,
            incremental=True,
            batch_size=50,
            include_transcription=False,
            dataset_description="Dataset audio Fongbè",
            dataset_citation="Collecté et traité automatiquement à partir de YouTube"
        )
        
        if result["success"]:
            logging.info(f"Upload vers Hugging Face réussi: {repo_id}")
        else:
            logging.error(f"Échec de l'upload vers Hugging Face: {repo_id} - {result.get('error', 'Erreur inconnue')}")
        
        return result
        
    except Exception as e:
        logging.error(f"Erreur lors de l'upload vers Hugging Face: {e}")
        return {"success": False, "error": str(e)}

def clean_storage(file_paths: List[str]) -> Dict[str, Any]:
    """
    Nettoie les espaces de stockage après traitement complet.
    
    Entrées :
        file_paths (List[str]) : Liste des chemins de fichiers à nettoyer
    Sorties :
        Dict[str, Any] : Résultat du nettoyage
    """
    try:
        # Initialiser le nettoyeur
        storage_cleaner = StorageCleaner()
        
        # Statistiques
        stats = {
            "total_files": len(file_paths),
            "cleaned_local": 0,
            "cleaned_minio": 0,
            "failed": 0,
            "errors": []
        }
        
        # Nettoyer chaque fichier
        for file_path in file_paths:
            result = storage_cleaner.clean_after_upload(
                file_path=file_path,
                verify_azure=False  # Ne pas vérifier Azure car l'upload peut avoir échoué
            )
            
            if result["success"]:
                if result["local_cleaned"]:
                    stats["cleaned_local"] += 1
                if result["minio_cleaned"]:
                    stats["cleaned_minio"] += 1
            else:
                stats["failed"] += 1
                stats["errors"].append(result["error"])
        
        stats["success"] = stats["failed"] == 0
        
        logging.info(f"Nettoyage terminé: {stats['cleaned_local']}/{stats['total_files']} fichiers locaux nettoyés, "
                    f"{stats['cleaned_minio']}/{stats['total_files']} fichiers MinIO nettoyés")
        
        return stats
        
    except Exception as e:
        logging.error(f"Erreur lors du nettoyage des espaces de stockage: {e}")
        return {"success": False, "error": str(e)}

def process_video_result(result: Dict[str, Any], processed_dir: str, hf_repo_id: str, skip_processing: bool, skip_hf_upload: bool, skip_cleaning: bool) -> Dict[str, Any]:
    """
    Traite le résultat du téléchargement d'une vidéo.
    
    Cette fonction orchestre le traitement audio, l'upload vers Hugging Face
    et le nettoyage des espaces de stockage.
    
    Entrées :
        result (Dict[str, Any]) : Résultat du téléchargement
        processed_dir (str) : Répertoire pour les fichiers traités
        hf_repo_id (str) : ID du dépôt Hugging Face
        skip_processing (bool) : Sauter l'étape de traitement audio
        skip_hf_upload (bool) : Sauter l'étape d'upload vers Hugging Face
        skip_cleaning (bool) : Sauter l'étape de nettoyage
    Sorties :
        Dict[str, Any] : Résultat global du traitement
    """
    if not result["success"]:
        return result
    
    file_path = result["file_path"]
    pipeline_result = {"success": True}
    
    # Traitement audio
    if not skip_processing:
        pipeline_result = process_audio(file_path, processed_dir)
        
        if not pipeline_result["success"]:
            logging.error(f"Échec du traitement audio: {file_path}")
            return {**result, **pipeline_result, "success": False}
    
    # Upload vers Hugging Face
    hf_result = {"success": True}
    if not skip_hf_upload and hf_repo_id:
        hf_result = upload_to_huggingface(processed_dir, hf_repo_id)
        
        if not hf_result["success"]:
            logging.error(f"Échec de l'upload vers Hugging Face: {hf_repo_id}")
            return {**result, **pipeline_result, **hf_result, "success": False}
    
    # Nettoyage des espaces de stockage
    if not skip_cleaning:
        clean_result = clean_storage([file_path])
        
        if not clean_result["success"]:
            logging.warning(f"Échec du nettoyage des espaces de stockage: {file_path}")
            # Ne pas échouer le traitement global si le nettoyage échoue
    
    return {
        **result,
        **pipeline_result,
        **hf_result,
        "success": True
    }

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
    
    # Créer les répertoires
    output_dir = args.output_dir
    processed_dir = args.processed_dir
    Path(output_dir).mkdir(exist_ok=True)
    Path(processed_dir).mkdir(exist_ok=True)
    
    # Étape 1: Téléchargement et upload vers MinIO/Azure
    if not args.skip_download:
        logging.info("Étape 1: Téléchargement et upload vers MinIO/Azure")
        
        # Initialiser le scraper
        youtube_client = YouTubeClient()
        video_downloader = VideoDownloader(output_dir=output_dir)
        
        # Lire les playlists
        with open(playlist_file, 'r') as f:
            playlist_urls = [line.strip() for line in f if line.strip()]
        
        if not playlist_urls:
            logging.warning(f"Aucune playlist trouvée dans le fichier: {playlist_file}")
            sys.exit(1)
        
        # Traiter chaque playlist
        download_results = []
        for playlist_url in playlist_urls:
            # Récupérer les vidéos de la playlist
            video_urls = youtube_client.get_videos_from_playlist_url(playlist_url)
            
            if not video_urls:
                logging.warning(f"Aucune vidéo trouvée dans la playlist: {playlist_url}")
                continue
            
            logging.info(f"Traitement de {len(video_urls)} vidéos de la playlist: {playlist_url}")
            
            # Télécharger et uploader chaque vidéo
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = []
                existing_videos = []
                
                for url in video_urls:
                    # Vérifier si la vidéo existe déjà
                    existence_info = check_exists(url)
                    
                    if existence_info["exists"]:
                        logging.info(f"Vidéo déjà téléchargée: {url}, continuant avec le traitement")
                        # Ajouter aux vidéos existantes pour traitement ultérieur
                        if existence_info["file_path"]:
                            existing_videos.append({
                                "success": True,
                                "file_path": existence_info["file_path"],
                                "metadata": existence_info["metadata"],
                                "video_id": existence_info["video_id"],
                                "from_cache": True,
                                "source": existence_info["source"]
                            })
                        continue
                    
                    # Soumettre la tâche de téléchargement pour les nouvelles vidéos
                    future = executor.submit(download_and_upload, url, output_dir, playlist_url)
                    futures.append(future)
                
                # Traiter les résultats des téléchargements
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        download_results.append(result)
                    except Exception as e:
                        logging.error(f"Erreur lors du téléchargement parallèle: {e}")
                
                # Ajouter les vidéos existantes aux résultats
                download_results.extend(existing_videos)
        
        # Vérifier si des vidéos ont été téléchargées ou trouvées dans la base de données
        successful_downloads = [r for r in download_results if r["success"]]
        # Récupérer les vidéos existantes (trouvées dans MongoDB)
        existing_videos_list = []
        for playlist_url in playlist_urls:
            video_urls = youtube_client.get_videos_from_playlist_url(playlist_url)
            for url in video_urls:
                existence_info = check_exists(url)
                if existence_info["exists"] and existence_info["file_path"]:
                    existing_videos_list.append({
                        "success": True,
                        "file_path": existence_info["file_path"],
                        "metadata": existence_info["metadata"],
                        "video_id": existence_info["video_id"],
                        "from_cache": True,
                        "source": existence_info["source"]
                    })
        
        if not successful_downloads and not existing_videos_list:
            logging.warning("Aucune vidéo téléchargée avec succès et aucune vidéo existante trouvée")
            # Ne pas quitter le programme, continuer avec le traitement des fichiers existants
            # si skip_processing et skip_hf_upload sont tous les deux True, alors il n'y a rien à faire
            if args.skip_processing and args.skip_hf_upload:
                sys.exit(0)
        elif existing_videos_list:
            logging.info(f"Trouvé {len(existing_videos_list)} vidéos existantes dans la base de données")
            # Ajouter les vidéos existantes aux résultats
            successful_downloads.extend(existing_videos_list)
    else:
        logging.info("Étape de téléchargement ignorée")
        successful_downloads = []
    
    # Étape 2: Traitement audio, upload vers Hugging Face et nettoyage
    # Vérifier directement les fichiers audio existants dans le répertoire output_dir
    audio_files = list(Path(output_dir).glob("*.wav"))
    if audio_files:
        logging.info(f"Trouvé {len(audio_files)} fichiers audio existants dans le répertoire {output_dir}")
    
    # Continuer même si aucune nouvelle vidéo n'a été téléchargée, car nous pourrions avoir des vidéos existantes
    if successful_downloads or args.skip_download or existing_videos_list or audio_files:
        # Si on a sauté le téléchargement ou si aucun nouveau téléchargement n'a été effectué,
        # utiliser les fichiers audio existants dans le répertoire de sortie
        if args.skip_download or not successful_downloads:
            audio_files = list(Path(output_dir).glob("*.wav"))
            # Ajouter les fichiers audio existants à la liste des téléchargements réussis
            # pour qu'ils soient traités par la suite
            if audio_files:
                logging.info(f"Utilisation de {len(audio_files)} fichiers audio existants pour le traitement")
                successful_downloads = [{"success": True, "file_path": str(f)} for f in audio_files]
            
        # Ajouter les vidéos existantes (trouvées dans MongoDB) aux téléchargements réussis
        if existing_videos:
            logging.info(f"Traitement de {len(existing_videos)} vidéos existantes trouvées dans la base de données")
            # Ajouter les vidéos existantes à successful_downloads s'ils ne sont pas déjà présents
            existing_paths = [r["file_path"] for r in successful_downloads]
            for video in existing_videos:
                if video["file_path"] not in existing_paths:
                    successful_downloads.append(video)
        
        # Traiter chaque fichier téléchargé
        for result in successful_downloads:
            final_result = process_video_result(
                result=result,
                processed_dir=processed_dir,
                hf_repo_id=args.hf_repo_id,
                skip_processing=args.skip_processing,
                skip_hf_upload=args.skip_hf_upload,
                skip_cleaning=args.skip_cleaning
            )
            
            if final_result["success"]:
                logging.info(f"Traitement complet réussi pour: {result['file_path']}")
            else:
                logging.error(f"Échec du traitement complet pour: {result['file_path']}")
    
    logging.info("Pipeline terminé")

if __name__ == "__main__":
    # Parser les arguments
    args = parse_args()
    
    # Exécuter la fonction principale
    main()
