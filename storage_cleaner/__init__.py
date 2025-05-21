"""
Module de nettoyage des espaces de stockage.

Ce module permet de libérer l'espace disque et de nettoyer les fichiers dans MinIO
après leur upload réussi vers Azure Blob Storage.

Classes principales :
- StorageCleaner : Gère le nettoyage des fichiers locaux et dans MinIO
"""

from .storage_cleaner import StorageCleaner

__all__ = [
    'StorageCleaner'
]
