#!/usr/bin/env python3
"""
Script de test pour vérifier le bon fonctionnement de chaque module du pipeline.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def test_scraper():
    """Teste le module de scraping YouTube."""
    from scraper.youtube_client import YouTubeClient
    from scraper.video_downloader import VideoDownloader
    from scraper.playlist_scraper import PlaylistScraper
    
    logging.info("=== Test du module scraper ===")
    
    # Tester YouTubeClient
    try:
        client = YouTubeClient()
        logging.info("YouTubeClient initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de YouTubeClient: {e}")
        return False
    
    # Tester VideoDownloader
    try:
        downloader = VideoDownloader(output_dir="test_output")
        logging.info("VideoDownloader initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de VideoDownloader: {e}")
        return False
    
    # Tester PlaylistScraper
    try:
        scraper = PlaylistScraper(output_dir="test_output")
        logging.info("PlaylistScraper initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de PlaylistScraper: {e}")
        return False
    
    logging.info("Module scraper testé avec succès")
    return True

def test_minio_uploader():
    """Teste le module d'upload vers MinIO."""
    from minio_uploader.minio_client import MinioClient
    from minio_uploader.minio_uploader import MinioUploader
    
    logging.info("=== Test du module minio_uploader ===")
    
    # Tester MinioClient
    try:
        client = MinioClient()
        logging.info("MinioClient initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de MinioClient: {e}")
        return False
    
    # Tester MinioUploader
    try:
        uploader = MinioUploader()
        logging.info("MinioUploader initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de MinioUploader: {e}")
        return False
    
    logging.info("Module minio_uploader testé avec succès")
    return True

def test_azure_uploader():
    """Teste le module d'upload vers Azure."""
    from azure_uploader.azure_client import AzureClient
    from azure_uploader.azure_uploader import AzureUploader
    
    logging.info("=== Test du module azure_uploader ===")
    
    # Tester AzureClient
    try:
        client = AzureClient()
        logging.info("AzureClient initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de AzureClient: {e}")
        return False
    
    # Tester AzureUploader
    try:
        uploader = AzureUploader()
        logging.info("AzureUploader initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de AzureUploader: {e}")
        return False
    
    logging.info("Module azure_uploader testé avec succès")
    return True

def test_storage_cleaner():
    """Teste le module de nettoyage de stockage."""
    from storage_cleaner.storage_cleaner import StorageCleaner
    
    logging.info("=== Test du module storage_cleaner ===")
    
    # Tester StorageCleaner
    try:
        cleaner = StorageCleaner()
        logging.info("StorageCleaner initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de StorageCleaner: {e}")
        return False
    
    logging.info("Module storage_cleaner testé avec succès")
    return True

def test_processing():
    """Teste les modules de traitement audio."""
    logging.info("=== Test des modules de traitement audio ===")
    
    # Tester AudioLoader
    try:
        from processing.audio_loader import AudioLoader
        loader = AudioLoader()
        logging.info("AudioLoader initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de AudioLoader: {e}")
        return False
    
    # Tester LoudnessNormalizer
    try:
        from processing.loudness_normalizer import LoudnessNormalizer
        normalizer = LoudnessNormalizer()
        logging.info("LoudnessNormalizer initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de LoudnessNormalizer: {e}")
        return False
    
    # Tester SilenceRemover
    try:
        from processing.silence_remover import SilenceRemover
        remover = SilenceRemover()
        logging.info("SilenceRemover initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de SilenceRemover: {e}")
        return False
    
    # Tester Diarization
    try:
        from processing.diarization import Diarization
        diarizer = Diarization()
        logging.info("Diarization initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de Diarization: {e}")
        return False
    
    # Tester Segmentation
    try:
        from processing.segmentation import Segmentation
        segmenter = Segmentation()
        logging.info("Segmentation initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de Segmentation: {e}")
        return False
    
    # Tester AudioCleaner
    try:
        from processing.audio_cleaner import AudioCleaner
        cleaner = AudioCleaner()
        logging.info("AudioCleaner initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de AudioCleaner: {e}")
        return False
    
    # Tester MetadataManager
    try:
        from processing.metadata_manager import MetadataManager
        manager = MetadataManager()
        logging.info("MetadataManager initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de MetadataManager: {e}")
        return False
    
    # Tester DataAugmentation
    try:
        from processing.data_augmentation import DataAugmentation
        augmenter = DataAugmentation()
        logging.info("DataAugmentation initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de DataAugmentation: {e}")
        return False
    
    # Tester QualityChecker
    try:
        from processing.quality_checker import QualityChecker
        checker = QualityChecker()
        logging.info("QualityChecker initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de QualityChecker: {e}")
        return False
    
    logging.info("Modules de traitement audio testés avec succès")
    return True

def test_upload_hf():
    """Teste le module d'upload vers Hugging Face."""
    logging.info("=== Test du module upload_hf ===")
    
    # Tester HFClient
    try:
        from upload_hf.hf_client import HFClient
        client = HFClient()
        logging.info("HFClient initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de HFClient: {e}")
        return False
    
    # Tester HFUploader
    try:
        from upload_hf.hf_uploader import HFUploader
        uploader = HFUploader()
        logging.info("HFUploader initialisé avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de HFUploader: {e}")
        return False
    
    logging.info("Module upload_hf testé avec succès")
    return True

def main():
    """Fonction principale pour tester tous les modules."""
    # Créer un répertoire de test
    os.makedirs("test_output", exist_ok=True)
    
    # Tester chaque module
    results = {
        "scraper": test_scraper(),
        "minio_uploader": test_minio_uploader(),
        "azure_uploader": test_azure_uploader(),
        "storage_cleaner": test_storage_cleaner(),
        "processing": test_processing(),
        "upload_hf": test_upload_hf()
    }
    
    # Afficher un résumé
    logging.info("=== Résumé des tests ===")
    for module, success in results.items():
        status = "✅ OK" if success else "❌ ÉCHEC"
        logging.info(f"{module}: {status}")
    
    # Vérifier si tous les tests ont réussi
    if all(results.values()):
        logging.info("Tous les modules ont été testés avec succès !")
        return 0
    else:
        logging.error("Certains modules ont échoué aux tests.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
