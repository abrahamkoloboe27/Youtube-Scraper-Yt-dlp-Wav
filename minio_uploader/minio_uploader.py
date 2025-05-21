"""
Module d'upload vers MinIO.

Ce module fournit une classe pour téléverser des fichiers vers MinIO,
en gérant la création de buckets et la vérification d'existence des objets.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv

from .minio_client import MinioClient

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/minio.log'),
        logging.StreamHandler()
    ]
)

class MinioUploader:
    """
    Uploader pour MinIO.
    
    Cette classe permet de téléverser des fichiers vers MinIO,
    en gérant la création de buckets et la vérification d'existence des objets.
    """
    
    def __init__(self, 
                minio_client: Optional[MinioClient] = None,
                default_bucket: str = "audios"):
        """
        Initialise l'uploader MinIO.
        
        Entrées :
            minio_client (Optional[MinioClient]) : Client MinIO à utiliser
            default_bucket (str) : Nom du bucket par défaut
        """
        load_dotenv()
        
        self.minio_client = minio_client or MinioClient()
        self.default_bucket = default_bucket or os.getenv("MINIO_BUCKET", "audios")
        
        # Créer le bucket par défaut s'il n'existe pas
        self.minio_client.create_bucket(self.default_bucket)
    
    def upload_file(self, 
                   file_path: Union[str, Path], 
                   bucket: Optional[str] = None, 
                   object_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Téléverse un fichier vers MinIO.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier à téléverser
            bucket (Optional[str]) : Nom du bucket (utilise le bucket par défaut si None)
            object_name (Optional[str]) : Nom de l'objet dans MinIO (utilise le nom du fichier si None)
        Sorties :
            Dict[str, Any] : Résultat de l'upload
            {
                "success": bool,
                "bucket": str,
                "object_name": str,
                "error": str (si échec)
            }
        """
        file_path = Path(file_path)
        bucket = bucket or self.default_bucket
        object_name = object_name or file_path.name
        
        # Vérifier que le fichier existe
        if not file_path.exists():
            error_msg = f"Le fichier {file_path} n'existe pas"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            # Créer le bucket s'il n'existe pas
            if not self.minio_client.bucket_exists(bucket):
                self.minio_client.create_bucket(bucket)
            
            # Téléverser le fichier
            self.minio_client.client.fput_object(bucket, object_name, str(file_path))
            
            logging.info(f"Fichier téléversé avec succès: {file_path} -> {bucket}/{object_name}")
            
            return {
                "success": True,
                "bucket": bucket,
                "object_name": object_name,
                "size": file_path.stat().st_size
            }
            
        except Exception as e:
            error_msg = f"Erreur lors du téléversement de {file_path}: {e}"
            logging.error(error_msg)
            
            return {
                "success": False,
                "bucket": bucket,
                "object_name": object_name,
                "error": str(e)
            }
    
    def upload_directory(self, 
                        dir_path: Union[str, Path], 
                        bucket: Optional[str] = None,
                        recursive: bool = True,
                        file_pattern: str = "*.*") -> Dict[str, Any]:
        """
        Téléverse tous les fichiers d'un répertoire vers MinIO.
        
        Entrées :
            dir_path (Union[str, Path]) : Chemin du répertoire à téléverser
            bucket (Optional[str]) : Nom du bucket (utilise le bucket par défaut si None)
            recursive (bool) : Recherche récursive si True
            file_pattern (str) : Motif de fichier à rechercher
        Sorties :
            Dict[str, Any] : Résultat de l'upload
            {
                "success": bool,
                "total_files": int,
                "uploaded_files": int,
                "failed_files": int,
                "errors": List[str] (si échecs)
            }
        """
        dir_path = Path(dir_path)
        bucket = bucket or self.default_bucket
        
        # Vérifier que le répertoire existe
        if not dir_path.exists() or not dir_path.is_dir():
            error_msg = f"Le répertoire {dir_path} n'existe pas"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Trouver tous les fichiers
        if recursive:
            files = list(dir_path.glob(f"**/{file_pattern}"))
        else:
            files = list(dir_path.glob(file_pattern))
        
        if not files:
            logging.warning(f"Aucun fichier trouvé dans {dir_path} avec le motif {file_pattern}")
            return {"success": True, "total_files": 0, "uploaded_files": 0, "failed_files": 0}
        
        # Statistiques
        stats = {
            "total_files": len(files),
            "uploaded_files": 0,
            "failed_files": 0,
            "errors": []
        }
        
        # Téléverser chaque fichier
        for file_path in files:
            result = self.upload_file(file_path, bucket)
            
            if result["success"]:
                stats["uploaded_files"] += 1
            else:
                stats["failed_files"] += 1
                stats["errors"].append(result["error"])
        
        stats["success"] = stats["failed_files"] == 0
        
        logging.info(f"Téléversement du répertoire terminé: {stats['uploaded_files']}/{stats['total_files']} fichiers téléversés")
        
        return stats
    
    def file_exists(self, object_name: str, bucket: Optional[str] = None) -> bool:
        """
        Vérifie si un fichier existe dans MinIO.
        
        Entrées :
            object_name (str) : Nom de l'objet à vérifier
            bucket (Optional[str]) : Nom du bucket (utilise le bucket par défaut si None)
        Sorties :
            bool : True si le fichier existe, False sinon
        """
        bucket = bucket or self.default_bucket
        return self.minio_client.object_exists(bucket, object_name)
    
    def delete_file(self, object_name: str, bucket: Optional[str] = None) -> bool:
        """
        Supprime un fichier de MinIO.
        
        Entrées :
            object_name (str) : Nom de l'objet à supprimer
            bucket (Optional[str]) : Nom du bucket (utilise le bucket par défaut si None)
        Sorties :
            bool : True si le fichier a été supprimé, False sinon
        """
        bucket = bucket or self.default_bucket
        
        try:
            # Vérifier que l'objet existe
            if not self.minio_client.object_exists(bucket, object_name):
                logging.warning(f"L'objet {object_name} n'existe pas dans le bucket {bucket}")
                return False
            
            # Supprimer l'objet
            self.minio_client.client.remove_object(bucket, object_name)
            
            logging.info(f"Objet supprimé avec succès: {bucket}/{object_name}")
            return True
            
        except Exception as e:
            logging.error(f"Erreur lors de la suppression de l'objet {object_name} dans le bucket {bucket}: {e}")
            return False
