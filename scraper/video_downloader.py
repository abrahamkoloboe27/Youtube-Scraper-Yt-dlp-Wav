"""
Module de téléchargement de vidéos YouTube.

Ce module fournit une classe pour télécharger l'audio des vidéos YouTube
en utilisant yt-dlp, avec gestion des erreurs et des retries.
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yt_dlp
from dotenv import load_dotenv

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

class VideoDownloader:
    """
    Téléchargeur d'audio pour les vidéos YouTube.
    
    Cette classe permet de télécharger l'audio des vidéos YouTube
    en gérant les erreurs et les retries.
    """
    
    def __init__(self, 
                output_dir: str = "audios", 
                audio_format: str = "wav",
                max_retries: int = 3,
                retry_delay: int = 5,
                cookie_file: Optional[str] = "cookies.txt"):
        """
        Initialise le téléchargeur de vidéos.
        
        Entrées :
            output_dir (str) : Répertoire de sortie pour les fichiers audio
            audio_format (str) : Format audio souhaité (wav, mp3, etc.)
            max_retries (int) : Nombre maximum de tentatives en cas d'échec
            retry_delay (int) : Délai entre les tentatives en secondes
            cookie_file (Optional[str]) : Chemin vers le fichier de cookies (None pour désactiver)
        """
        load_dotenv()
        self.output_dir = Path(output_dir)
        self.audio_format = audio_format
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cookie_file = cookie_file
        
        # Créer le répertoire de sortie s'il n'existe pas
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_output_path(self, url: str) -> Path:
        """
        Génère le chemin de sortie pour une vidéo.
        
        Entrées :
            url (str) : URL de la vidéo YouTube
        Sorties :
            Path : Chemin du fichier de sortie
        """
        video_id = url.split('=')[-1]
        return self.output_dir / f"{video_id}.{self.audio_format}"
    
    def download_audio(self, url: str) -> Dict[str, Any]:
        """
        Télécharge l'audio d'une vidéo YouTube.
        
        Entrées :
            url (str) : URL de la vidéo YouTube
        Sorties :
            Dict[str, Any] : Résultat du téléchargement avec métadonnées
            {
                "success": bool,
                "file_path": str,
                "error": str (si échec),
                "metadata": dict (si succès)
            }
        """
        output_path = self.get_output_path(url)
        
        # Options pour yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': self.audio_format,
            }],
            'outtmpl': str(output_path.with_suffix('')),
            'ignoreerrors': True,
            'no_warnings': True,
            'quiet': False,
            'verbose': True,
            'geo_bypass': True,
            'geo_bypass_country': 'US',
            'nocheckcertificate': True,
            'extractor_retries': 5,
            'socket_timeout': 30,
            'external_downloader_args': ['-loglevel', 'debug'],
            'skip_download': False,
            'noplaylist': True,
            'extract_flat': False,
            'force_generic_extractor': False,
            'youtube_include_dash_manifest': False,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'referer': 'https://www.youtube.com/',
        }
        
        # Ajouter le fichier de cookies si spécifié
        if self.cookie_file:
            cookie_path = Path(self.cookie_file)
            if cookie_path.exists():
                logging.info(f"Utilisation du fichier de cookies: {cookie_path.absolute()}")
                ydl_opts['cookiefile'] = str(cookie_path.absolute())
            else:
                logging.warning(f"Fichier de cookies non trouvé: {cookie_path.absolute()}")
        
        # Tentatives de téléchargement
        for attempt in range(self.max_retries):
            try:
                logging.info(f"Téléchargement de la vidéo: {url} (tentative {attempt+1}/{self.max_retries})")
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    
                    # Vérifier que le fichier a bien été créé
                    if not output_path.exists():
                        raise FileNotFoundError(f"Le fichier {output_path} n'a pas été créé")
                    
                    # Récupérer les métadonnées
                    metadata = {
                        "title": info.get('title', ''),
                        "duration": info.get('duration', 0),
                        "filesize": output_path.stat().st_size,
                        "url": url,
                        "file": output_path.name,
                        "format": self.audio_format
                    }
                    
                    logging.info(f"Téléchargement réussi: {output_path}")
                    
                    return {
                        "success": True,
                        "file_path": str(output_path),
                        "metadata": metadata
                    }
                    
            except Exception as e:
                error_msg = str(e)
                logging.warning(f"Échec du téléchargement (tentative {attempt+1}/{self.max_retries}): {error_msg}")
                
                # Gestion spécifique de l'erreur 403 Forbidden
                if "HTTP Error 403: Forbidden" in error_msg:
                    logging.warning("Erreur 403 Forbidden détectée. Tentative avec des options alternatives.")
                    # Essayer avec des options alternatives pour l'erreur 403
                    try:
                        alt_opts = ydl_opts.copy()
                        alt_opts['format'] = 'worstaudio/worst'
                        alt_opts['force_generic_extractor'] = True
                        
                        with yt_dlp.YoutubeDL(alt_opts) as ydl:
                            info = ydl.extract_info(url, download=True)
                            
                            # Vérifier que le fichier a bien été créé
                            if output_path.exists():
                                metadata = {
                                    "title": info.get('title', ''),
                                    "duration": info.get('duration', 0),
                                    "filesize": output_path.stat().st_size,
                                    "url": url,
                                    "file": output_path.name,
                                    "format": self.audio_format,
                                    "recovery_method": "alternative_format"
                                }
                                
                                logging.info(f"Téléchargement réussi avec options alternatives: {output_path}")
                                
                                return {
                                    "success": True,
                                    "file_path": str(output_path),
                                    "metadata": metadata
                                }
                    except Exception as alt_e:
                        logging.warning(f"L'approche alternative a également échoué: {str(alt_e)}")
                
                # Si ce n'est pas la dernière tentative, attendre avant de réessayer
                if attempt < self.max_retries - 1:
                    # Augmenter progressivement le délai entre les tentatives
                    retry_wait = self.retry_delay * (attempt + 1)
                    logging.info(f"Attente de {retry_wait} secondes avant la prochaine tentative...")
                    time.sleep(retry_wait)
        
        # Si toutes les tentatives ont échoué
        error_type = "cookie_error" if self.cookie_file and "cookies" in error_msg.lower() else "download_error"
        
        return {
            "success": False,
            "error": error_msg,
            "error_type": error_type,
            "url": url
        }
