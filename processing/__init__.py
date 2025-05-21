"""
Module de traitement audio pour le Fongbè.

Ce module implémente un pipeline complet de préparation de données audio Fongbè
pour l'entraînement de modèles de transcription. Il automatise les étapes de prétraitement,
segmentation, nettoyage, annotation et augmentation des fichiers .wav.

Modules principaux :
- audio_loader : Chargement et uniformisation des fichiers audio
- loudness_normalizer : Normalisation du volume
- silence_remover : Suppression des silences
- diarization : Isolation des locuteurs
- segmentation : Segmentation fine des fichiers audio
- audio_cleaner : Nettoyage sonore et amélioration de la qualité
- metadata_manager : Étiquetage et gestion des métadonnées
- data_augmentation : Augmentation de données
- quality_checker : Vérification finale et exports
- mongo_logger : Centralisation des logs dans MongoDB
- pipeline : Exécution du pipeline complet de traitement
"""

from .audio_loader import AudioLoader
from .loudness_normalizer import LoudnessNormalizer
from .silence_remover import SilenceRemover
from .diarization import Diarization
from .segmentation import Segmentation
from .audio_cleaner import AudioCleaner
from .metadata_manager import MetadataManager
from .data_augmentation import DataAugmentation
from .quality_checker import QualityChecker
from .mongo_logger import MongoLogger
from .pipeline import AudioPipeline

__all__ = [
    'AudioLoader',
    'LoudnessNormalizer',
    'SilenceRemover',
    'Diarization',
    'Segmentation',
    'AudioCleaner',
    'MetadataManager',
    'DataAugmentation',
    'QualityChecker',
    'MongoLogger',
    'AudioPipeline'
]
