"""
Module d'upload vers Azure Blob Storage.

Ce module fournit une classe pour téléverser des fichiers vers Azure Blob Storage,
en gérant la création de conteneurs et la vérification d'existence des blobs.
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv

from .azure_client import AzureClient

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/azure.log'),
        logging.StreamHandler()
    ]
)

class AzureUploader:
    """
    Uploader pour Azure Blob Storage.
    
    Cette classe permet de téléverser des fichiers vers Azure Blob Storage,
    en gérant la création de conteneurs et la vérification d'existence des blobs.
    """
    
    def __init__(self, 
                azure_client: Optional[AzureClient] = None,
                default_container: str = "audios",
                max_retries: int = 3,
                retry_delay: int = 5):
        """
        Initialise l'uploader Azure.
        
        Entrées :
            azure_client (Optional[AzureClient]) : Client Azure à utiliser
            default_container (str) : Nom du conteneur par défaut
            max_retries (int) : Nombre maximum de tentatives en cas d'échec
            retry_delay (int) : Délai entre les tentatives en secondes
        """
        load_dotenv()
        
        self.azure_client = azure_client or AzureClient()
        self.default_container = default_container or os.getenv("AZURE_CONTAINER_NAME", "audios")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Créer le conteneur par défaut s'il n'existe pas
        if self.azure_client.is_connected():
            self.azure_client.create_container(self.default_container)
    
    def upload_file(self, 
                   file_path: Union[str, Path], 
                   container: Optional[str] = None, 
                   blob_name: Optional[str] = None,
                   overwrite: bool = True) -> Dict[str, Any]:
        """
        Téléverse un fichier vers Azure Blob Storage.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier à téléverser
            container (Optional[str]) : Nom du conteneur (utilise le conteneur par défaut si None)
            blob_name (Optional[str]) : Nom du blob dans Azure (utilise le nom du fichier si None)
            overwrite (bool) : Écrase le blob s'il existe déjà
        Sorties :
            Dict[str, Any] : Résultat de l'upload
            {
                "success": bool,
                "container": str,
                "blob_name": str,
                "error": str (si échec)
            }
        """
        if not self.azure_client.is_connected():
            error_msg = "Client Azure non connecté"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}
        
        file_path = Path(file_path)
        container = container or self.default_container
        blob_name = blob_name or file_path.name
        
        # Vérifier que le fichier existe
        if not file_path.exists():
            error_msg = f"Le fichier {file_path} n'existe pas"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Tentatives d'upload
        for attempt in range(self.max_retries):
            try:
                # Créer le conteneur s'il n'existe pas
                if not self.azure_client.container_exists(container):
                    self.azure_client.create_container(container)
                
                # Récupérer le client de conteneur
                container_client = self.azure_client.get_container_client(container)
                if not container_client:
                    raise Exception(f"Impossible de récupérer le client pour le conteneur {container}")
                
                # Téléverser le fichier
                with open(file_path, "rb") as data:
                    container_client.upload_blob(name=blob_name, data=data, overwrite=overwrite)
                
                logging.info(f"Fichier téléversé avec succès: {file_path} -> {container}/{blob_name}")
                
                return {
                    "success": True,
                    "container": container,
                    "blob_name": blob_name,
                    "size": file_path.stat().st_size
                }
                
            except Exception as e:
                error_msg = f"Erreur lors du téléversement de {file_path} (tentative {attempt+1}/{self.max_retries}): {e}"
                logging.warning(error_msg)
                
                # Si ce n'est pas la dernière tentative, attendre avant de réessayer
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        # Si toutes les tentatives ont échoué
        error_msg = f"Échec du téléversement après {self.max_retries} tentatives: {file_path}"
        logging.error(error_msg)
        
        return {
            "success": False,
            "container": container,
            "blob_name": blob_name,
            "error": error_msg
        }
    
    def upload_directory(self, 
                        dir_path: Union[str, Path], 
                        container: Optional[str] = None,
                        recursive: bool = True,
                        file_pattern: str = "*.*",
                        overwrite: bool = True) -> Dict[str, Any]:
        """
        Téléverse tous les fichiers d'un répertoire vers Azure Blob Storage.
        
        Entrées :
            dir_path (Union[str, Path]) : Chemin du répertoire à téléverser
            container (Optional[str]) : Nom du conteneur (utilise le conteneur par défaut si None)
            recursive (bool) : Recherche récursive si True
            file_pattern (str) : Motif de fichier à rechercher
            overwrite (bool) : Écrase les blobs s'ils existent déjà
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
        if not self.azure_client.is_connected():
            error_msg = "Client Azure non connecté"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}
        
        dir_path = Path(dir_path)
        container = container or self.default_container
        
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
            # Déterminer le nom du blob (chemin relatif si récursif)
            if recursive:
                rel_path = file_path.relative_to(dir_path)
                blob_name = str(rel_path).replace("\\", "/")
            else:
                blob_name = file_path.name
            
            result = self.upload_file(file_path, container, blob_name, overwrite)
            
            if result["success"]:
                stats["uploaded_files"] += 1
            else:
                stats["failed_files"] += 1
                stats["errors"].append(result["error"])
        
        stats["success"] = stats["failed_files"] == 0
        
        logging.info(f"Téléversement du répertoire terminé: {stats['uploaded_files']}/{stats['total_files']} fichiers téléversés")
        
        return stats
    
    def blob_exists(self, blob_name: str, container: Optional[str] = None) -> bool:
        """
        Vérifie si un blob existe dans Azure Blob Storage.
        
        Entrées :
            blob_name (str) : Nom du blob à vérifier
            container (Optional[str]) : Nom du conteneur (utilise le conteneur par défaut si None)
        Sorties :
            bool : True si le blob existe, False sinon
        """
        if not self.azure_client.is_connected():
            return False
            
        container = container or self.default_container
        return self.azure_client.blob_exists(container, blob_name)
    
    def delete_blob(self, blob_name: str, container: Optional[str] = None) -> bool:
        """
        Supprime un blob d'Azure Blob Storage.
        
        Entrées :
            blob_name (str) : Nom du blob à supprimer
            container (Optional[str]) : Nom du conteneur (utilise le conteneur par défaut si None)
        Sorties :
            bool : True si le blob a été supprimé, False sinon
        """
        if not self.azure_client.is_connected():
            return False
            
        container = container or self.default_container
        
        try:
            # Vérifier que le blob existe
            if not self.azure_client.blob_exists(container, blob_name):
                logging.warning(f"Le blob {blob_name} n'existe pas dans le conteneur {container}")
                return False
            
            # Récupérer le client de blob
            blob_client = self.azure_client.get_blob_client(container, blob_name)
            if not blob_client:
                return False
            
            # Supprimer le blob
            blob_client.delete_blob()
            
            logging.info(f"Blob supprimé avec succès: {container}/{blob_name}")
            return True
            
        except Exception as e:
            logging.error(f"Erreur lors de la suppression du blob {blob_name} dans le conteneur {container}: {e}")
            return False
