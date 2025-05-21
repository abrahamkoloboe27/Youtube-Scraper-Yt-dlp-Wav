"""
Module d'upload vers Hugging Face pour les données audio Fongbè.

Ce module permet de téléverser les fichiers audio traités et leurs métadonnées
vers un dataset Hugging Face, facilitant ainsi le partage et l'utilisation des données.

Modules principaux :
- hf_client : Client pour l'API Hugging Face
- dataset_creator : Création et configuration du dataset
- metadata_formatter : Formatage des métadonnées pour Hugging Face
- audio_uploader : Upload des fichiers audio
- dataset_uploader : Upload du dataset complet
- upload_manager : Gestion globale du processus d'upload
- mongo_logger : Journalisation dans MongoDB
"""

from .hf_client import HFClient
from .dataset_creator import DatasetCreator
from .metadata_formatter import MetadataFormatter
from .audio_uploader import AudioUploader
from .dataset_uploader import DatasetUploader
from .upload_manager import UploadManager
from .mongo_logger import HFMongoLogger

__all__ = [
    'HFClient',
    'DatasetCreator',
    'MetadataFormatter',
    'AudioUploader',
    'DatasetUploader',
    'UploadManager',
    'HFMongoLogger'
]
