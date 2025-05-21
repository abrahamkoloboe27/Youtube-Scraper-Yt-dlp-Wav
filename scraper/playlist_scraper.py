"""
Module de scraping de playlists YouTube.

Ce module fournit une classe pour orchestrer le scraping complet de playlists YouTube,
en gérant le téléchargement parallèle des vidéos et la vérification des doublons.
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from .youtube_client import YouTubeClient
from .video_downloader import VideoDownloader

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ││ %(levelname)s ││ %(name)s ││ %(message)s ",
    datefmt="%Y-%m-%d %H:%M:%S",

    handlers=[
        logging.FileHandler('logs/scraper.log'),
        logging.StreamHandler()
    ]
)

class PlaylistScraper:
    """
    Scraper de playlists YouTube.
    
    Cette classe orchestre le processus complet de scraping de playlists YouTube,
    en gérant le téléchargement parallèle des vidéos et la vérification des doublons.
    """
    
    def __init__(self, 
                youtube_client: Optional[YouTubeClient] = None,
                video_downloader: Optional[VideoDownloader] = None,
                max_workers: int = 4,
                max_cookie_failures: int = 100,
                check_exists_callback: Optional[Callable[[str], bool]] = None):
        """
        Initialise le scraper de playlists.
        
        Entrées :
            youtube_client (Optional[YouTubeClient]) : Client YouTube à utiliser
            video_downloader (Optional[VideoDownloader]) : Téléchargeur de vidéos à utiliser
            max_workers (int) : Nombre maximum de workers pour le téléchargement parallèle
            max_cookie_failures (int) : Nombre maximum d'échecs liés aux cookies avant d'arrêter
            check_exists_callback (Optional[Callable]) : Fonction pour vérifier si une vidéo existe déjà
        """
        load_dotenv()
        self.youtube_client = youtube_client or YouTubeClient()
        self.video_downloader = video_downloader or VideoDownloader()
        self.max_workers = max_workers
        self.max_cookie_failures = max_cookie_failures
        self.check_exists_callback = check_exists_callback
        
        # Statistiques
        self.stats = {
            "total_videos": 0,
            "downloaded_videos": 0,
            "failed_videos": 0,
            "skipped_videos": 0,
            "cookie_failures": 0
        }
    
    def process_video(self, url: str, playlist_url: str) -> Dict[str, Any]:
        """
        Traite une vidéo YouTube (téléchargement).
        
        Entrées :
            url (str) : URL de la vidéo YouTube
            playlist_url (str) : URL de la playlist d'origine
        Sorties :
            Dict[str, Any] : Résultat du traitement
        """
        # Vérifier si la vidéo existe déjà
        if self.check_exists_callback and self.check_exists_callback(url):
            logging.info(f"Vidéo déjà traitée: {url}")
            self.stats["skipped_videos"] += 1
            return {"success": True, "skipped": True, "url": url}
        
        # Télécharger la vidéo
        result = self.video_downloader.download_audio(url)
        
        # Ajouter des informations supplémentaires
        result["playlist_url"] = playlist_url
        
        # Mettre à jour les statistiques
        if result["success"]:
            self.stats["downloaded_videos"] += 1
        else:
            self.stats["failed_videos"] += 1
            if result.get("error_type") == "cookie_error":
                self.stats["cookie_failures"] += 1
        
        return result
    
    def process_playlist(self, playlist_url: str, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Traite une playlist YouTube complète.
        
        Entrées :
            playlist_url (str) : URL de la playlist YouTube
            callback (Optional[Callable]) : Fonction à appeler pour chaque vidéo traitée
        Sorties :
            Dict[str, Any] : Statistiques du traitement
        """
        video_urls = self.youtube_client.get_videos_from_playlist_url(playlist_url)
        self.stats["total_videos"] += len(video_urls)
        
        if not video_urls:
            logging.warning(f"Aucune vidéo trouvée dans la playlist: {playlist_url}")
            return self.stats
        
        logging.info(f"Traitement de {len(video_urls)} vidéos de la playlist: {playlist_url}")
        
        tasks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for url in video_urls:
                # Vérifier si on a atteint le nombre maximum d'échecs liés aux cookies
                if self.stats["cookie_failures"] >= self.max_cookie_failures:
                    logging.error(f"Nombre maximum d'échecs liés aux cookies atteint ({self.max_cookie_failures}). Arrêt du traitement.")
                    break
                
                # Soumettre la tâche
                tasks.append(executor.submit(self.process_video, url, playlist_url))
            
            # Traiter les résultats au fur et à mesure
            for future in as_completed(tasks):
                try:
                    result = future.result()
                    
                    # Appeler le callback si fourni
                    if callback and not result.get("skipped", False):
                        callback(result)
                        
                except Exception as e:
                    logging.error(f"Erreur lors du traitement parallèle: {e}")
                    self.stats["failed_videos"] += 1
        
        logging.info(f"Traitement de la playlist terminé: {playlist_url}")
        logging.info(f"Statistiques: {self.stats}")
        
        return self.stats
    
    def process_playlists(self, playlist_urls: List[str], callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Traite plusieurs playlists YouTube.
        
        Entrées :
            playlist_urls (List[str]) : Liste des URLs des playlists YouTube
            callback (Optional[Callable]) : Fonction à appeler pour chaque vidéo traitée
        Sorties :
            Dict[str, Any] : Statistiques du traitement
        """
        for playlist_url in playlist_urls:
            # Vérifier si on a atteint le nombre maximum d'échecs liés aux cookies
            if self.stats["cookie_failures"] >= self.max_cookie_failures:
                logging.error(f"Nombre maximum d'échecs liés aux cookies atteint ({self.max_cookie_failures}). Arrêt du traitement.")
                break
            
            self.process_playlist(playlist_url, callback)
        
        logging.info(f"Traitement de toutes les playlists terminé")
        logging.info(f"Statistiques finales: {self.stats}")
        
        return self.stats
    
    def process_playlists_from_file(self, playlist_file: str, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Traite les playlists YouTube listées dans un fichier.
        
        Entrées :
            playlist_file (str) : Chemin vers le fichier contenant les URLs des playlists
            callback (Optional[Callable]) : Fonction à appeler pour chaque vidéo traitée
        Sorties :
            Dict[str, Any] : Statistiques du traitement
        """
        try:
            with open(playlist_file, 'r') as f:
                playlist_urls = [line.strip() for line in f if line.strip()]
            
            if not playlist_urls:
                logging.warning(f"Aucune playlist trouvée dans le fichier: {playlist_file}")
                return self.stats
            
            logging.info(f"Traitement de {len(playlist_urls)} playlists depuis le fichier: {playlist_file}")
            return self.process_playlists(playlist_urls, callback)
            
        except Exception as e:
            logging.error(f"Erreur lors de la lecture du fichier de playlists: {e}")
            return self.stats
