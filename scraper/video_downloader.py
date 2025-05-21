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
    format='%(asctime)s - %(levelname)s - %(message)s',
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
        }
        
        # Ajouter le fichier de cookies si spécifié
        if self.cookie_file:
            ydl_opts['cookiefile'] = self.cookie_file
        
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
                
                # Si ce n'est pas la dernière tentative, attendre avant de réessayer
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        # Si toutes les tentatives ont échoué
        error_type = "cookie_error" if self.cookie_file and "cookies" in error_msg.lower() else "download_error"
        
        return {
            "success": False,
            "error": error_msg,
            "error_type": error_type,
            "url": url
        }
