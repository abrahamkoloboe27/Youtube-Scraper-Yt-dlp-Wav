#!/usr/bin/env python3
"""
Script de test pour télécharger une vidéo YouTube.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.video_downloader import VideoDownloader

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ││ %(levelname)s ││ %(name)s ││ %(message)s ",
    datefmt="%Y-%m-%d %H:%M:%S",

    handlers=[
        logging.StreamHandler()
    ]
)

def main():
    """Fonction principale pour tester le téléchargement d'une vidéo YouTube."""
    # URL de test (vidéo YouTube "Me at the zoo", la première vidéo YouTube)
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    
    # Créer un répertoire de sortie
    output_dir = Path("test_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialiser le téléchargeur avec le fichier de cookies
    downloader = VideoDownloader(
        output_dir=str(output_dir),
        audio_format="wav",
        max_retries=3,
        retry_delay=5,
        cookie_file="cookies.txt"
    )
    
    # Télécharger la vidéo
    logging.info(f"Tentative de téléchargement de la vidéo: {url}")
    result = downloader.download_audio(url)
    
    # Afficher le résultat
    if result["success"]:
        logging.info(f"Téléchargement réussi: {result['file_path']}")
        logging.info(f"Métadonnées: {result['metadata']}")
    else:
        logging.error(f"Échec du téléchargement: {result.get('error', 'Erreur inconnue')}")
    
    return 0 if result["success"] else 1

if __name__ == "__main__":
    sys.exit(main())
