"""
Module d'upload de dataset complet vers Hugging Face.

Ce module permet de téléverser un dataset complet vers Hugging Face,
incluant les fichiers audio, les métadonnées et les configurations.

Classes principales :
- DatasetUploader : Classe gérant l'upload de datasets complets
"""

import os
import logging
import json
import time
import shutil
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, Repository
from .hf_client import HFClient
from .audio_uploader import AudioUploader

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hf_upload.log'),
        logging.StreamHandler()
    ]
)

class DatasetUploader:
    """
    Classe pour téléverser un dataset complet vers Hugging Face.
    
    Cette classe permet de téléverser un dataset complet vers Hugging Face,
    incluant les fichiers audio, les métadonnées et les configurations.
    """
    
    def __init__(self, 
                hf_client: Optional[HFClient] = None,
                repo_id: Optional[str] = None,
                local_dir: Union[str, Path] = "hf_dataset",
                create_repo_if_missing: bool = True,
                private: bool = False):
        """
        Initialise l'uploader de dataset avec les paramètres spécifiés.
        
        Entrées :
            hf_client (Optional[HFClient]) : Client Hugging Face
            repo_id (Optional[str]) : Identifiant du dépôt Hugging Face
            local_dir (Union[str, Path]) : Répertoire local contenant le dataset
            create_repo_if_missing (bool) : Si True, crée le dépôt s'il n'existe pas
            private (bool) : Si True, le dépôt sera privé
        """
        self.hf_client = hf_client or HFClient()
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.create_repo_if_missing = create_repo_if_missing
        self.private = private
        
        # Initialiser l'uploader audio
        self.audio_uploader = AudioUploader(
            hf_client=self.hf_client,
            repo_id=self.repo_id,
            local_dir=self.local_dir
        )
        
        # Statistiques d'upload
        self.upload_stats = {
            "start_time": None,
            "end_time": None,
            "duration_sec": 0,
            "total_files": 0,
            "uploaded_files": 0,
            "failed_files": 0,
            "total_size_mb": 0
        }
    
    def prepare_repo(self) -> bool:
        """
        Prépare le dépôt Hugging Face pour l'upload.
        
        Sorties :
            bool : True si la préparation a réussi, False sinon
        """
        if not self.repo_id:
            logging.error("Aucun dépôt spécifié pour l'upload")
            return False
        
        try:
            # Vérifier si le dépôt existe
            repo_exists = self.hf_client.check_repo_exists(self.repo_id, repo_type="dataset")
            
            if not repo_exists:
                if self.create_repo_if_missing:
                    # Créer le dépôt
                    success = self.hf_client.create_repo(
                        repo_id=self.repo_id,
                        private=self.private,
                        repo_type="dataset"
                    )
                    
                    if not success:
                        logging.error(f"Échec de la création du dépôt {self.repo_id}")
                        return False
                    
                    logging.info(f"Dépôt créé: {self.repo_id}")
                else:
                    logging.error(f"Le dépôt {self.repo_id} n'existe pas et create_repo_if_missing est False")
                    return False
            
            # Cloner le dépôt en local si nécessaire
            if not (self.local_dir / ".git").exists():
                repo = self.hf_client.clone_repo(
                    repo_id=self.repo_id,
                    local_dir=self.local_dir,
                    repo_type="dataset"
                )
                
                if not repo:
                    logging.error(f"Échec du clonage du dépôt {self.repo_id}")
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Erreur lors de la préparation du dépôt: {e}")
            return False
    
    def upload_dataset_structure(self) -> bool:
        """
        Téléverse la structure du dataset (README, configuration, etc.).
        
        Sorties :
            bool : True si l'upload a réussi, False sinon
        """
        if not self.repo_id:
            logging.error("Aucun dépôt spécifié pour l'upload")
            return False
        
        try:
            # Vérifier que le répertoire local existe
            if not self.local_dir.exists() or not self.local_dir.is_dir():
                logging.error(f"Le répertoire local {self.local_dir} n'existe pas")
                return False
            
            # Téléverser le README.md s'il existe
            readme_path = self.local_dir / "README.md"
            if readme_path.exists():
                success = self.hf_client.upload_file(
                    repo_id=self.repo_id,
                    file_path=readme_path,
                    path_in_repo="README.md"
                )
                
                if not success:
                    logging.warning("Échec du téléversement du README.md")
            
            # Téléverser la configuration du dataset s'il existe
            config_path = self.local_dir / "dataset_infos.json"
            if config_path.exists():
                success = self.hf_client.upload_file(
                    repo_id=self.repo_id,
                    file_path=config_path,
                    path_in_repo="dataset_infos.json"
                )
                
                if not success:
                    logging.warning("Échec du téléversement de dataset_infos.json")
            
            # Téléverser les métadonnées
            metadata_dir = self.local_dir / "metadata"
            if metadata_dir.exists() and metadata_dir.is_dir():
                success = self.hf_client.upload_folder(
                    repo_id=self.repo_id,
                    folder_path=metadata_dir,
                    path_in_repo="metadata"
                )
                
                if not success:
                    logging.warning("Échec du téléversement du dossier metadata")
            
            return True
            
        except Exception as e:
            logging.error(f"Erreur lors du téléversement de la structure du dataset: {e}")
            return False
    
    def upload_dataset(self, 
                      metadata_file: Optional[Union[str, Path]] = None,
                      audio_column: str = "segment_file",
                      audio_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Téléverse un dataset complet vers Hugging Face.
        
        Entrées :
            metadata_file (Optional[Union[str, Path]]) : Chemin du fichier de métadonnées
            audio_column (str) : Nom de la colonne contenant les chemins des fichiers audio
            audio_dir (Optional[Union[str, Path]]) : Répertoire contenant les fichiers audio
        
        Sorties :
            Dict[str, Any] : Statistiques d'upload
        """
        if not self.repo_id:
            logging.error("Aucun dépôt spécifié pour l'upload")
            return {"success": False}
        
        self.upload_stats["start_time"] = time.time()
        
        try:
            # Préparer le dépôt
            if not self.prepare_repo():
                return {"success": False, "error": "Échec de la préparation du dépôt"}
            
            # Téléverser la structure du dataset
            if not self.upload_dataset_structure():
                logging.warning("Échec du téléversement de la structure du dataset")
            
            # Téléverser les fichiers audio
            if metadata_file:
                # Téléverser les fichiers audio à partir des métadonnées
                metadata_file = Path(metadata_file)
                audio_stats = self.audio_uploader.upload_from_metadata(
                    metadata_file=metadata_file,
                    audio_column=audio_column,
                    audio_source_dir=audio_dir
                )
            else:
                # Téléverser tous les fichiers audio du répertoire
                audio_dir = audio_dir or self.local_dir / "audio"
                audio_stats = self.audio_uploader.upload_directory(
                    dir_path=audio_dir,
                    recursive=True,
                    file_pattern="*.wav"
                )
            
            # Mettre à jour les statistiques
            self.upload_stats.update(audio_stats)
            self.upload_stats["end_time"] = time.time()
            self.upload_stats["duration_sec"] = self.upload_stats["end_time"] - self.upload_stats["start_time"]
            
            logging.info(f"Téléversement du dataset terminé: {self.upload_stats['uploaded_files']}/{self.upload_stats['total_files']} "
                        f"fichiers ({self.upload_stats['total_size_mb']:.2f} MB) en {self.upload_stats['duration_sec']:.2f}s")
            
            return {"success": True, **self.upload_stats}
            
        except Exception as e:
            logging.error(f"Erreur lors du téléversement du dataset: {e}")
            self.upload_stats["end_time"] = time.time()
            self.upload_stats["duration_sec"] = self.upload_stats["end_time"] - self.upload_stats["start_time"]
            return {"success": False, "error": str(e), **self.upload_stats}
    
    def upload_dataset_incrementally(self, 
                                   metadata_file: Union[str, Path],
                                   audio_column: str = "segment_file",
                                   audio_dir: Optional[Union[str, Path]] = None,
                                   batch_size: int = 50) -> Dict[str, Any]:
        """
        Téléverse un dataset de manière incrémentale vers Hugging Face.
        
        Entrées :
            metadata_file (Union[str, Path]) : Chemin du fichier de métadonnées
            audio_column (str) : Nom de la colonne contenant les chemins des fichiers audio
            audio_dir (Optional[Union[str, Path]]) : Répertoire contenant les fichiers audio
            batch_size (int) : Nombre de fichiers à téléverser par lot
        
        Sorties :
            Dict[str, Any] : Statistiques d'upload
        """
        if not self.repo_id:
            logging.error("Aucun dépôt spécifié pour l'upload")
            return {"success": False}
        
        self.upload_stats["start_time"] = time.time()
        
        try:
            # Préparer le dépôt
            if not self.prepare_repo():
                return {"success": False, "error": "Échec de la préparation du dépôt"}
            
            # Téléverser la structure du dataset
            if not self.upload_dataset_structure():
                logging.warning("Échec du téléversement de la structure du dataset")
            
            # Charger les métadonnées
            metadata_file = Path(metadata_file)
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
            if audio_dir:
                audio_dir = Path(audio_dir)
            else:
                audio_dir = self.local_dir / "audio"
            
            # Préparer les chemins complets des fichiers audio
            file_paths = [audio_dir / file for file in audio_files]
            
            # Vérifier l'existence des fichiers
            existing_files = [path for path in file_paths if path.exists()]
            missing_files = len(file_paths) - len(existing_files)
            
            if missing_files > 0:
                logging.warning(f"{missing_files} fichiers audio manquants")
            
            # Mettre à jour le batch_size de l'uploader audio
            self.audio_uploader.batch_size = batch_size
            
            # Téléverser les fichiers par lots
            total_batches = (len(existing_files) + batch_size - 1) // batch_size
            
            logging.info(f"Téléversement incrémental de {len(existing_files)} fichiers audio en {total_batches} lots")
            
            for i in range(0, len(existing_files), batch_size):
                batch = existing_files[i:i+batch_size]
                self.audio_uploader.upload_audio_batch(batch)
                
                # Mettre à jour les statistiques intermédiaires
                self.upload_stats.update(self.audio_uploader.upload_stats)
                
                # Afficher la progression
                progress = (i + len(batch)) / len(existing_files) * 100
                logging.info(f"Progression: {progress:.2f}% ({i + len(batch)}/{len(existing_files)} fichiers)")
            
            # Téléverser également le fichier de métadonnées
            metadata_path_in_repo = f"metadata/{metadata_file.name}"
            self.hf_client.upload_file(
                repo_id=self.repo_id,
                file_path=metadata_file,
                path_in_repo=metadata_path_in_repo
            )
            
            # Mettre à jour les statistiques finales
            self.upload_stats["end_time"] = time.time()
            self.upload_stats["duration_sec"] = self.upload_stats["end_time"] - self.upload_stats["start_time"]
            
            logging.info(f"Téléversement incrémental terminé: {self.upload_stats['uploaded_files']}/{self.upload_stats['total_files']} "
                        f"fichiers ({self.upload_stats['total_size_mb']:.2f} MB) en {self.upload_stats['duration_sec']:.2f}s")
            
            return {"success": True, **self.upload_stats}
            
        except Exception as e:
            logging.error(f"Erreur lors du téléversement incrémental du dataset: {e}")
            self.upload_stats["end_time"] = time.time()
            self.upload_stats["duration_sec"] = self.upload_stats["end_time"] - self.upload_stats["start_time"]
            return {"success": False, "error": str(e), **self.upload_stats}
