"""
Module de nettoyage des espaces de stockage.

Ce module fournit une classe pour libérer l'espace disque et nettoyer les fichiers dans MinIO
après leur upload réussi vers Azure Blob Storage.
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dotenv import load_dotenv

# Import des modules d'upload
try:
    from minio_uploader.minio_client import MinioClient
    from minio_uploader.minio_uploader import MinioUploader
    from azure_uploader.azure_client import AzureClient
    from azure_uploader.azure_uploader import AzureUploader
except ImportError:
    # Fallback pour les imports relatifs si les modules ne sont pas installés
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from minio_uploader.minio_client import MinioClient
    from minio_uploader.minio_uploader import MinioUploader
    from azure_uploader.azure_client import AzureClient
    from azure_uploader.azure_uploader import AzureUploader

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/storage_cleaner.log'),
        logging.StreamHandler()
    ]
)

class StorageCleaner:
    """
    Nettoyeur d'espaces de stockage.
    
    Cette classe permet de libérer l'espace disque et de nettoyer les fichiers dans MinIO
    après leur upload réussi vers Azure Blob Storage.
    """
    
    def __init__(self, 
                minio_client: Optional[MinioClient] = None,
                azure_client: Optional[AzureClient] = None,
                minio_bucket: str = "audios",
                azure_container: str = "audios",
                local_dir: Union[str, Path] = "audios"):
        """
        Initialise le nettoyeur d'espaces de stockage.
        
        Entrées :
            minio_client (Optional[MinioClient]) : Client MinIO à utiliser
            azure_client (Optional[AzureClient]) : Client Azure à utiliser
            minio_bucket (str) : Nom du bucket MinIO par défaut
            azure_container (str) : Nom du conteneur Azure par défaut
            local_dir (Union[str, Path]) : Répertoire local par défaut
        """
        load_dotenv()
        
        self.minio_client = minio_client or MinioClient()
        self.azure_client = azure_client or AzureClient()
        
        self.minio_bucket = minio_bucket or os.getenv("MINIO_BUCKET", "audios")
        self.azure_container = azure_container or os.getenv("AZURE_CONTAINER_NAME", "audios")
        self.local_dir = Path(local_dir)
        
        # Créer le répertoire local s'il n'existe pas
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        # Wrappers pour les opérations de nettoyage
        self.minio_uploader = MinioUploader(minio_client=self.minio_client, default_bucket=self.minio_bucket)
        self.azure_uploader = AzureUploader(azure_client=self.azure_client, default_container=self.azure_container)
    
    def clean_local_file(self, file_path: Union[str, Path]) -> bool:
        """
        Supprime un fichier local.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier à supprimer
        Sorties :
            bool : True si le fichier a été supprimé, False sinon
        """
        file_path = Path(file_path)
        
        try:
            if file_path.exists():
                file_path.unlink()
                logging.info(f"Fichier local supprimé: {file_path}")
                return True
            else:
                logging.warning(f"Le fichier local n'existe pas: {file_path}")
                return False
        except Exception as e:
            logging.error(f"Erreur lors de la suppression du fichier local {file_path}: {e}")
            return False
    
    def clean_minio_file(self, object_name: str, bucket: Optional[str] = None) -> bool:
        """
        Supprime un fichier de MinIO.
        
        Entrées :
            object_name (str) : Nom de l'objet à supprimer
            bucket (Optional[str]) : Nom du bucket (utilise le bucket par défaut si None)
        Sorties :
            bool : True si le fichier a été supprimé, False sinon
        """
        bucket = bucket or self.minio_bucket
        return self.minio_uploader.delete_file(object_name, bucket)
    
    def clean_after_upload(self, 
                          file_path: Union[str, Path], 
                          object_name: Optional[str] = None,
                          blob_name: Optional[str] = None,
                          verify_azure: bool = True) -> Dict[str, Any]:
        """
        Nettoie les fichiers local et MinIO après un upload réussi vers Azure.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier local
            object_name (Optional[str]) : Nom de l'objet dans MinIO (utilise le nom du fichier si None)
            blob_name (Optional[str]) : Nom du blob dans Azure (utilise le nom du fichier si None)
            verify_azure (bool) : Vérifie que le fichier existe dans Azure avant de nettoyer
        Sorties :
            Dict[str, Any] : Résultat du nettoyage
            {
                "success": bool,
                "local_cleaned": bool,
                "minio_cleaned": bool,
                "error": str (si échec)
            }
        """
        file_path = Path(file_path)
        object_name = object_name or file_path.name
        blob_name = blob_name or file_path.name
        
        result = {
            "success": True,
            "local_cleaned": False,
            "minio_cleaned": False,
            "error": None
        }
        
        try:
            # Vérifier que le fichier existe dans Azure si demandé
            if verify_azure and not self.azure_uploader.blob_exists(blob_name):
                error_msg = f"Le fichier n'existe pas dans Azure: {self.azure_container}/{blob_name}"
                logging.warning(error_msg)
                result["success"] = False
                result["error"] = error_msg
                return result
            
            # Nettoyer le fichier local
            local_result = self.clean_local_file(file_path)
            result["local_cleaned"] = local_result
            
            # Nettoyer le fichier dans MinIO
            minio_result = self.clean_minio_file(object_name)
            result["minio_cleaned"] = minio_result
            
            # Vérifier le résultat global
            if not local_result and not minio_result:
                result["success"] = False
                result["error"] = "Échec du nettoyage local et MinIO"
            elif not local_result:
                result["success"] = False
                result["error"] = "Échec du nettoyage local"
            elif not minio_result:
                result["success"] = False
                result["error"] = "Échec du nettoyage MinIO"
            
            logging.info(f"Nettoyage après upload: {file_path} -> Local: {local_result}, MinIO: {minio_result}")
            
            return result
            
        except Exception as e:
            error_msg = f"Erreur lors du nettoyage après upload: {e}"
            logging.error(error_msg)
            
            result["success"] = False
            result["error"] = error_msg
            
            return result
    
    def clean_directory(self, 
                       dir_path: Union[str, Path],
                       verify_azure: bool = True,
                       file_pattern: str = "*.wav") -> Dict[str, Any]:
        """
        Nettoie tous les fichiers d'un répertoire local et de MinIO après un upload réussi vers Azure.
        
        Entrées :
            dir_path (Union[str, Path]) : Chemin du répertoire local
            verify_azure (bool) : Vérifie que les fichiers existent dans Azure avant de nettoyer
            file_pattern (str) : Motif de fichier à rechercher
        Sorties :
            Dict[str, Any] : Résultat du nettoyage
            {
                "success": bool,
                "total_files": int,
                "cleaned_local": int,
                "cleaned_minio": int,
                "failed": int,
                "errors": List[str] (si échecs)
            }
        """
        dir_path = Path(dir_path)
        
        # Vérifier que le répertoire existe
        if not dir_path.exists() or not dir_path.is_dir():
            error_msg = f"Le répertoire n'existe pas: {dir_path}"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Trouver tous les fichiers
        files = list(dir_path.glob(file_pattern))
        
        if not files:
            logging.warning(f"Aucun fichier trouvé dans {dir_path} avec le motif {file_pattern}")
            return {"success": True, "total_files": 0, "cleaned_local": 0, "cleaned_minio": 0, "failed": 0}
        
        # Statistiques
        stats = {
            "total_files": len(files),
            "cleaned_local": 0,
            "cleaned_minio": 0,
            "failed": 0,
            "errors": []
        }
        
        # Nettoyer chaque fichier
        for file_path in files:
            result = self.clean_after_upload(file_path, verify_azure=verify_azure)
            
            if result["success"]:
                if result["local_cleaned"]:
                    stats["cleaned_local"] += 1
                if result["minio_cleaned"]:
                    stats["cleaned_minio"] += 1
            else:
                stats["failed"] += 1
                stats["errors"].append(result["error"])
        
        stats["success"] = stats["failed"] == 0
        
        logging.info(f"Nettoyage du répertoire terminé: {stats['cleaned_local']}/{stats['total_files']} fichiers locaux nettoyés, "
                    f"{stats['cleaned_minio']}/{stats['total_files']} fichiers MinIO nettoyés")
        
        return stats
    
    def clean_by_prefix(self, 
                       prefix: str,
                       verify_azure: bool = True) -> Dict[str, Any]:
        """
        Nettoie tous les fichiers de MinIO avec un préfixe donné après un upload réussi vers Azure.
        
        Entrées :
            prefix (str) : Préfixe des objets à nettoyer
            verify_azure (bool) : Vérifie que les fichiers existent dans Azure avant de nettoyer
        Sorties :
            Dict[str, Any] : Résultat du nettoyage
            {
                "success": bool,
                "total_files": int,
                "cleaned_minio": int,
                "failed": int,
                "errors": List[str] (si échecs)
            }
        """
        try:
            # Lister tous les objets avec le préfixe
            objects = list(self.minio_client.list_objects(self.minio_bucket, prefix=prefix))
            
            if not objects:
                logging.warning(f"Aucun objet trouvé dans {self.minio_bucket} avec le préfixe {prefix}")
                return {"success": True, "total_files": 0, "cleaned_minio": 0, "failed": 0}
            
            # Statistiques
            stats = {
                "total_files": len(objects),
                "cleaned_minio": 0,
                "failed": 0,
                "errors": []
            }
            
            # Nettoyer chaque objet
            for obj in objects:
                object_name = obj.object_name
                blob_name = object_name
                
                # Vérifier que le fichier existe dans Azure si demandé
                if verify_azure and not self.azure_uploader.blob_exists(blob_name):
                    error_msg = f"Le fichier n'existe pas dans Azure: {self.azure_container}/{blob_name}"
                    logging.warning(error_msg)
                    stats["failed"] += 1
                    stats["errors"].append(error_msg)
                    continue
                
                # Nettoyer le fichier dans MinIO
                if self.clean_minio_file(object_name):
                    stats["cleaned_minio"] += 1
                else:
                    stats["failed"] += 1
                    stats["errors"].append(f"Échec du nettoyage MinIO: {object_name}")
            
            stats["success"] = stats["failed"] == 0
            
            logging.info(f"Nettoyage par préfixe terminé: {stats['cleaned_minio']}/{stats['total_files']} fichiers MinIO nettoyés")
            
            return stats
            
        except Exception as e:
            error_msg = f"Erreur lors du nettoyage par préfixe: {e}"
            logging.error(error_msg)
            
            return {
                "success": False,
                "total_files": 0,
                "cleaned_minio": 0,
                "failed": 0,
                "error": error_msg
            }
