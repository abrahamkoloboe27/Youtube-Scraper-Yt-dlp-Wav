"""
Module d'upload des fichiers audio vers Hugging Face.

Ce module permet de téléverser les fichiers audio traités vers un dataset Hugging Face,
en gérant les métadonnées et les configurations associées.

Classes principales :
- AudioUploader : Classe gérant l'upload des fichiers audio
"""

import os
import logging
import json
import time
from typing import Dict, Any, Optional, List, Union, Tuple, Set
from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm
from .hf_client import HFClient
from .mongo_logger import HFMongoLogger

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ││ %(levelname)s ││ %(name)s ││ %(message)s ",
    datefmt="%Y-%m-%d %H:%M:%S",

    handlers=[
        logging.FileHandler('logs/hf_upload.log'),
        logging.StreamHandler()
    ]
)

class AudioUploader:
    """
    Classe pour téléverser les fichiers audio vers Hugging Face.
    
    Cette classe permet de téléverser les fichiers audio traités vers un dataset Hugging Face,
    en gérant les métadonnées et les configurations associées.
    """
    
    def __init__(self, 
                hf_client: Optional[HFClient] = None,
                repo_id: Optional[str] = None,
                local_dir: Union[str, Path] = "hf_dataset",
                audio_dir: Optional[Union[str, Path]] = None,
                batch_size: int = 50,
                max_retries: int = 3,
                retry_delay: int = 5,
                mongo_logger: Optional[HFMongoLogger] = None,
                mongo_doc_id: Optional[str] = None):
        """
        Initialise l'uploader audio avec les paramètres spécifiés.
        
        Entrées :
            hf_client (Optional[HFClient]) : Client Hugging Face
            repo_id (Optional[str]) : Identifiant du dépôt Hugging Face
            local_dir (Union[str, Path]) : Répertoire local pour le dataset
            audio_dir (Optional[Union[str, Path]]) : Répertoire contenant les fichiers audio
            batch_size (int) : Nombre de fichiers à téléverser par lot
            max_retries (int) : Nombre maximum de tentatives en cas d'échec
            retry_delay (int) : Délai entre les tentatives en secondes
        """
        self.hf_client = hf_client or HFClient()
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.audio_dir = Path(audio_dir) if audio_dir else self.local_dir / "audio"
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Logger MongoDB
        self.mongo_logger = mongo_logger
        self.mongo_doc_id = mongo_doc_id
        
        # Créer les répertoires s'ils n'existent pas
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistiques d'upload
        self.upload_stats = {
            "total_files": 0,
            "uploaded_files": 0,
            "failed_files": 0,
            "total_size_mb": 0,
            "upload_time_sec": 0
        }
    
    def upload_audio_file(self, 
                         file_path: Union[str, Path], 
                         path_in_repo: Optional[str] = None) -> bool:
        """
        Téléverse un fichier audio vers le dépôt Hugging Face.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier audio
            path_in_repo (Optional[str]) : Chemin dans le dépôt où placer le fichier
        
        Sorties :
            bool : True si le téléversement a réussi, False sinon
        """
        if not self.repo_id:
            logging.error("Aucun dépôt spécifié pour l'upload")
            return False
        
        file_path = Path(file_path)
        
        # Définir le chemin dans le dépôt si non spécifié
        if path_in_repo is None:
            path_in_repo = f"audio/{file_path.name}"
        
        # Vérifier que le fichier existe
        if not file_path.exists():
            logging.error(f"Le fichier {file_path} n'existe pas")
            return False
        
        # Téléverser le fichier avec plusieurs tentatives
        for attempt in range(self.max_retries):
            try:
                success = self.hf_client.upload_file(
                    repo_id=self.repo_id,
                    file_path=file_path,
                    path_in_repo=path_in_repo
                )
                
                if success:
                    logging.info(f"Fichier téléversé avec succès: {file_path} -> {path_in_repo}")
                    
                    # Journaliser le succès dans MongoDB
                    if self.mongo_logger and self.mongo_doc_id and self.mongo_logger.is_connected():
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        self.mongo_logger.log_file_upload(
                            doc_id=self.mongo_doc_id,
                            file_path=str(file_path),
                            success=True,
                            details={
                                "path_in_repo": path_in_repo,
                                "size_mb": file_size_mb,
                                "attempts": attempt + 1
                            }
                        )
                    
                    return True
                else:
                    logging.warning(f"Échec du téléversement: {file_path} (tentative {attempt+1}/{self.max_retries})")
            except Exception as e:
                logging.error(f"Erreur lors du téléversement de {file_path}: {e} (tentative {attempt+1}/{self.max_retries})")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        
        logging.error(f"Échec du téléversement après {self.max_retries} tentatives: {file_path}")
        
        # Journaliser l'échec dans MongoDB
        if self.mongo_logger and self.mongo_doc_id and self.mongo_logger.is_connected():
            self.mongo_logger.log_file_upload(
                doc_id=self.mongo_doc_id,
                file_path=str(file_path),
                success=False,
                details={
                    "path_in_repo": path_in_repo,
                    "attempts": self.max_retries
                },
                error=f"Échec après {self.max_retries} tentatives"
            )
        
        return False
    
    def upload_audio_batch(self, 
                          file_paths: List[Union[str, Path]], 
                          base_path_in_repo: str = "audio",
                          batch_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Téléverse un lot de fichiers audio vers le dépôt Hugging Face.
        
        Entrées :
            file_paths (List[Union[str, Path]]) : Liste des chemins des fichiers audio
            base_path_in_repo (str) : Chemin de base dans le dépôt
        
        Sorties :
            Dict[str, Any] : Statistiques d'upload pour ce lot
        """
        if not self.repo_id:
            logging.error("Aucun dépôt spécifié pour l'upload")
            return {"success": False, "uploaded": 0, "failed": len(file_paths)}
        
        batch_stats = {
            "total": len(file_paths),
            "uploaded": 0,
            "failed": 0,
            "size_mb": 0
        }
        
        start_time = time.time()
        
        for file_path in tqdm(file_paths, desc="Upload de fichiers audio"):
            file_path = Path(file_path)
            
            # Vérifier que le fichier existe
            if not file_path.exists():
                logging.warning(f"Le fichier {file_path} n'existe pas")
                batch_stats["failed"] += 1
                continue
            
            # Calculer la taille du fichier
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            batch_stats["size_mb"] += file_size_mb
            
            # Définir le chemin dans le dépôt
            path_in_repo = f"{base_path_in_repo}/{file_path.name}"
            
            # Téléverser le fichier
            success = self.upload_audio_file(file_path, path_in_repo)
            
            if success:
                batch_stats["uploaded"] += 1
            else:
                batch_stats["failed"] += 1
        
        end_time = time.time()
        batch_stats["time_sec"] = end_time - start_time
        
        # Mettre à jour les statistiques globales
        self.upload_stats["total_files"] += batch_stats["total"]
        self.upload_stats["uploaded_files"] += batch_stats["uploaded"]
        self.upload_stats["failed_files"] += batch_stats["failed"]
        self.upload_stats["total_size_mb"] += batch_stats["size_mb"]
        self.upload_stats["upload_time_sec"] += batch_stats["time_sec"]
        
        logging.info(f"Lot téléversé: {batch_stats['uploaded']}/{batch_stats['total']} fichiers "
                    f"({batch_stats['size_mb']:.2f} MB) en {batch_stats['time_sec']:.2f}s")
        
        return batch_stats
    
    def upload_from_metadata(self, 
                           metadata_file: Union[str, Path],
                           audio_column: str = "segment_file",
                           audio_source_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Téléverse les fichiers audio listés dans un fichier de métadonnées.
        
        Entrées :
            metadata_file (Union[str, Path]) : Chemin du fichier de métadonnées
            audio_column (str) : Nom de la colonne contenant les chemins des fichiers audio
            audio_source_dir (Optional[Union[str, Path]]) : Répertoire source des fichiers audio
        
        Sorties :
            Dict[str, Any] : Statistiques d'upload
        """
        if not self.repo_id:
            logging.error("Aucun dépôt spécifié pour l'upload")
            return {"success": False}
        
        metadata_file = Path(metadata_file)
        
        try:
            # Charger les métadonnées
            if metadata_file.suffix.lower() == '.csv':
                df = pd.read_csv(metadata_file)
            elif metadata_file.suffix.lower() == '.parquet':
                df = pd.read_parquet(metadata_file)
            else:
                raise ValueError(f"Format de fichier non pris en charge: {metadata_file.suffix}")
            
            # Vérifier que la colonne audio existe
            if audio_column not in df.columns:
                raise ValueError(f"La colonne {audio_column} n'existe pas dans le fichier de métadonnées")
            
            # Récupérer la liste des fichiers audio
            audio_files = df[audio_column].unique().tolist()
            
            # Définir le répertoire source des fichiers audio
            if audio_source_dir:
                audio_source_dir = Path(audio_source_dir)
            else:
                audio_source_dir = self.audio_dir
            
            # Préparer les chemins complets des fichiers audio
            file_paths = [audio_source_dir / file for file in audio_files]
            
            # Vérifier l'existence des fichiers
            existing_files = [path for path in file_paths if path.exists()]
            missing_files = len(file_paths) - len(existing_files)
            
            if missing_files > 0:
                logging.warning(f"{missing_files} fichiers audio manquants")
            
            # Téléverser les fichiers par lots
            total_batches = (len(existing_files) + self.batch_size - 1) // self.batch_size
            
            logging.info(f"Téléversement de {len(existing_files)} fichiers audio en {total_batches} lots")
            
            for i in range(0, len(existing_files), self.batch_size):
                batch = existing_files[i:i+self.batch_size]
                self.upload_audio_batch(batch)
            
            # Téléverser également le fichier de métadonnées
            metadata_path_in_repo = f"metadata/{metadata_file.name}"
            self.hf_client.upload_file(
                repo_id=self.repo_id,
                file_path=metadata_file,
                path_in_repo=metadata_path_in_repo
            )
            
            logging.info(f"Téléversement terminé: {self.upload_stats['uploaded_files']}/{self.upload_stats['total_files']} "
                        f"fichiers ({self.upload_stats['total_size_mb']:.2f} MB) en {self.upload_stats['upload_time_sec']:.2f}s")
            
            return self.upload_stats
            
        except Exception as e:
            logging.error(f"Erreur lors du téléversement depuis les métadonnées: {e}")
            return {"success": False, "error": str(e)}
    
    def upload_directory(self, 
                        dir_path: Union[str, Path],
                        recursive: bool = True,
                        file_pattern: str = "*.wav",
                        base_path_in_repo: str = "audio") -> Dict[str, Any]:
        """
        Téléverse tous les fichiers audio d'un répertoire.
        
        Entrées :
            dir_path (Union[str, Path]) : Chemin du répertoire à téléverser
            recursive (bool) : Si True, recherche récursivement dans les sous-répertoires
            file_pattern (str) : Motif de fichier à rechercher
            base_path_in_repo (str) : Chemin de base dans le dépôt
        
        Sorties :
            Dict[str, Any] : Statistiques d'upload
        """
        if not self.repo_id:
            logging.error("Aucun dépôt spécifié pour l'upload")
            return {"success": False}
        
        dir_path = Path(dir_path)
        
        try:
            # Vérifier que le répertoire existe
            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"Le répertoire {dir_path} n'existe pas")
            
            # Trouver tous les fichiers audio
            if recursive:
                file_paths = list(dir_path.glob(f"**/{file_pattern}"))
            else:
                file_paths = list(dir_path.glob(file_pattern))
            
            if not file_paths:
                logging.warning(f"Aucun fichier trouvé dans {dir_path} avec le motif {file_pattern}")
                return {"success": True, "uploaded": 0, "total": 0}
            
            # Téléverser les fichiers par lots
            total_batches = (len(file_paths) + self.batch_size - 1) // self.batch_size
            
            logging.info(f"Téléversement de {len(file_paths)} fichiers audio en {total_batches} lots")
            
            for i in range(0, len(file_paths), self.batch_size):
                batch = file_paths[i:i+self.batch_size]
                self.upload_audio_batch(batch, base_path_in_repo)
            
            logging.info(f"Téléversement terminé: {self.upload_stats['uploaded_files']}/{self.upload_stats['total_files']} "
                        f"fichiers ({self.upload_stats['total_size_mb']:.2f} MB) en {self.upload_stats['upload_time_sec']:.2f}s")
            
            return self.upload_stats
            
        except Exception as e:
            logging.error(f"Erreur lors du téléversement du répertoire: {e}")
            return {"success": False, "error": str(e)}
