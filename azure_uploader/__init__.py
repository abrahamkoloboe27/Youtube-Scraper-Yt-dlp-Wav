"""
Module d'upload de fichiers vers Azure Blob Storage.

Ce module permet de téléverser des fichiers audio vers Azure Blob Storage,
en gérant la connexion et les conteneurs.

Classes principales :
- AzureClient : Gère la connexion et l'interaction avec Azure Blob Storage
- AzureUploader : Téléverse des fichiers vers Azure Blob Storage
"""

from .azure_client import AzureClient
from .azure_uploader import AzureUploader

__all__ = [
    'AzureClient',
    'AzureUploader'
]
