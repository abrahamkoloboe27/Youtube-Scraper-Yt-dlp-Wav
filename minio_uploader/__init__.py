"""
Module d'upload de fichiers vers MinIO.

Ce module permet de téléverser des fichiers audio vers un stockage objet MinIO,
en gérant la création de buckets et la vérification d'existence des objets.

Classes principales :
- MinioClient : Gère la connexion et l'interaction avec MinIO
- MinioUploader : Téléverse des fichiers vers MinIO
"""

from .minio_client import MinioClient
from .minio_uploader import MinioUploader

__all__ = [
    'MinioClient',
    'MinioUploader'
]
