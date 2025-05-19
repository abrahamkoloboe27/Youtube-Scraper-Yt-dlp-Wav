"""
Utilitaires pour l'interaction avec le stockage objet MinIO.

Ce script fournit des fonctions pour obtenir un client MinIO et pour téléverser des fichiers dans un bucket.

Fonctions principales :
- get_minio_client : Retourne un client MinIO configuré à partir des variables d'environnement.
- upload_file : Téléverse un fichier local dans un bucket MinIO (créé le bucket s'il n'existe pas).
"""
import os
from minio import Minio

def get_minio_client():
    """
    Crée et retourne un client MinIO à partir des variables d'environnement.

    Entrées :
        Néant (utilise les variables d'environnement MINIO_ENDPOINT, MINIO_ROOT_USER, MINIO_ROOT_PASSWORD)
    Sorties :
        Minio : client MinIO prêt à l'emploi
    """
    return Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        secure=False
    )

def upload_file(local_path: str, bucket: str, object_name: str) -> None:
    """
    Téléverse un fichier local dans un bucket MinIO. Crée le bucket s'il n'existe pas.

    Entrées :
        local_path (str) : chemin du fichier local à téléverser
        bucket (str) : nom du bucket cible
        object_name (str) : nom du fichier dans le bucket
    Sorties :
        None
    """
    client = get_minio_client()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    client.fput_object(bucket, object_name, local_path)
