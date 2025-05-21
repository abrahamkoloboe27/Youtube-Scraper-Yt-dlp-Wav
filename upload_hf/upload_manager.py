"""
Module de gestion globale du processus d'upload vers Hugging Face.

Ce module orchestre l'ensemble du processus d'upload des données audio Fongbè
vers Hugging Face, en coordonnant les différentes étapes et composants.

Classes principales :
- UploadManager : Classe gérant l'ensemble du processus d'upload
"""

import os
import logging
import json
import time
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from .hf_client import HFClient
from .dataset_creator import DatasetCreator
from .metadata_formatter import MetadataFormatter
from .audio_uploader import AudioUploader
from .dataset_uploader import DatasetUploader
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

class UploadManager:
    """
    Classe pour gérer l'ensemble du processus d'upload vers Hugging Face.
    
    Cette classe orchestre l'ensemble du processus d'upload des données audio Fongbè
    vers Hugging Face, en coordonnant les différentes étapes et composants.
    """
    
    def __init__(self, 
                config_path: Optional[Union[str, Path]] = None,
                hf_token: Optional[str] = None,
                repo_id: Optional[str] = None,
                dataset_name: str = "fongbe_audio",
                dataset_language: str = "fon",
                local_dir: Union[str, Path] = "hf_dataset",
                private: bool = False):
        """
        Initialise le gestionnaire d'upload avec les paramètres spécifiés.
        
        Entrées :
            config_path (Optional[Union[str, Path]]) : Chemin du fichier de configuration
            hf_token (Optional[str]) : Token d'authentification Hugging Face
            repo_id (Optional[str]) : Identifiant du dépôt Hugging Face
            dataset_name (str) : Nom du dataset
            dataset_language (str) : Code de langue du dataset (ISO 639-3)
            local_dir (Union[str, Path]) : Répertoire local pour le dataset
            private (bool) : Si True, le dépôt sera privé
        """
        load_dotenv()
        
        # Charger la configuration si spécifiée
        self.config = self._load_config(config_path) if config_path else {}
        
        # Récupérer les paramètres de la configuration ou utiliser les valeurs par défaut
        self.hf_token = hf_token or self.config.get("hf_token") or os.getenv("HF_TOKEN")
        self.repo_id = repo_id or self.config.get("repo_id")
        self.dataset_name = dataset_name or self.config.get("dataset_name", "fongbe_audio")
        self.dataset_language = dataset_language or self.config.get("dataset_language", "fon")
        self.local_dir = Path(local_dir or self.config.get("local_dir", "hf_dataset"))
        self.private = private or self.config.get("private", False)
        
        # Créer le répertoire local s'il n'existe pas
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialiser le client Hugging Face
        self.hf_client = HFClient(token=self.hf_token)
        
        # Initialiser les composants
        self.dataset_creator = DatasetCreator(
            hf_client=self.hf_client,
            dataset_name=self.dataset_name,
            dataset_language=self.dataset_language,
            local_dir=self.local_dir
        )
        
        self.metadata_formatter = MetadataFormatter(
            language_code=self.dataset_language,
            output_dir=self.local_dir / "metadata"
        )
        
        self.dataset_uploader = DatasetUploader(
            hf_client=self.hf_client,
            repo_id=self.repo_id,
            local_dir=self.local_dir,
            private=self.private
        )
        
        # Initialiser le logger MongoDB
        self.mongo_logger = HFMongoLogger(collection_name="hf_uploads")
        
        # Statistiques d'exécution
        self.execution_stats = {
            "start_time": None,
            "end_time": None,
            "duration_sec": 0,
            "stages": {}
        }
    
    def _load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Charge la configuration à partir d'un fichier JSON.
        
        Entrées :
            config_path (Union[str, Path]) : Chemin du fichier de configuration
        
        Sorties :
            Dict[str, Any] : Configuration chargée
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logging.info(f"Configuration chargée depuis {config_path}")
            return config
        except Exception as e:
            logging.error(f"Erreur lors du chargement de la configuration: {e}")
            return {}
    
    def _time_stage(self, stage_name: str, func, *args, **kwargs):
        """
        Exécute une étape du processus en mesurant le temps d'exécution.
        
        Entrées :
            stage_name (str) : Nom de l'étape
            func : Fonction à exécuter
            *args, **kwargs : Arguments à passer à la fonction
        
        Sorties :
            Any : Résultat de la fonction
        """
        logging.info(f"Début de l'étape: {stage_name}")
        start_time = time.time()
        
        # Récupérer l'ID du document MongoDB si disponible
        mongo_doc_id = self.execution_stats.get("mongo_doc_id")
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            duration = end_time - start_time
            
            stage_stats = {
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": duration,
                "success": True
            }
            
            self.execution_stats["stages"][stage_name] = stage_stats
            
            # Journaliser l'étape dans MongoDB si disponible
            if mongo_doc_id and self.mongo_logger.is_connected():
                self.mongo_logger.log_stage(
                    doc_id=mongo_doc_id,
                    stage_name=stage_name,
                    success=True,
                    details={"duration_sec": duration}
                )
            
            logging.info(f"Fin de l'étape: {stage_name} (durée: {duration:.2f}s)")
            return result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            stage_stats = {
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": duration,
                "success": False,
                "error": str(e)
            }
            
            self.execution_stats["stages"][stage_name] = stage_stats
            
            # Journaliser l'erreur dans MongoDB si disponible
            if mongo_doc_id and self.mongo_logger.is_connected():
                self.mongo_logger.log_stage(
                    doc_id=mongo_doc_id,
                    stage_name=stage_name,
                    success=False,
                    details={"duration_sec": duration},
                    error=str(e)
                )
            
            logging.error(f"Erreur lors de l'étape {stage_name}: {e} (durée: {duration:.2f}s)")
            raise
    
    def prepare_metadata(self, 
                        metadata_file: Union[str, Path],
                        audio_column: str = "segment_file",
                        split_column: Optional[str] = "split",
                        speaker_column: Optional[str] = "speaker_id",
                        duration_column: Optional[str] = "duration") -> str:
        """
        Prépare les métadonnées pour l'upload vers Hugging Face.
        
        Entrées :
            metadata_file (Union[str, Path]) : Chemin du fichier de métadonnées
            audio_column (str) : Nom de la colonne contenant les chemins des fichiers audio
            split_column (Optional[str]) : Nom de la colonne contenant les splits
            speaker_column (Optional[str]) : Nom de la colonne contenant les IDs de locuteurs
            duration_column (Optional[str]) : Nom de la colonne contenant les durées
        
        Sorties :
            str : Chemin du fichier de métadonnées formaté
        """
        return self._time_stage(
            "prepare_metadata",
            self.metadata_formatter.create_hf_metadata,
            metadata_file=metadata_file,
            audio_column=audio_column,
            split_column=split_column,
            speaker_column=speaker_column,
            duration_column=duration_column
        )
    
    def prepare_dataset(self, 
                       metadata_file: Union[str, Path],
                       audio_dir: Union[str, Path],
                       include_transcription: bool = False,
                       speaker_labels: Optional[List[str]] = None,
                       dataset_description: str = "",
                       dataset_citation: str = "") -> Path:
        """
        Prépare un dataset local pour l'upload vers Hugging Face.
        
        Entrées :
            metadata_file (Union[str, Path]) : Chemin du fichier de métadonnées
            audio_dir (Union[str, Path]) : Répertoire contenant les fichiers audio
            include_transcription (bool) : Si True, inclut un champ pour la transcription
            speaker_labels (Optional[List[str]]) : Liste des labels de locuteurs
            dataset_description (str) : Description du dataset
            dataset_citation (str) : Citation du dataset
        
        Sorties :
            Path : Chemin du répertoire local préparé
        """
        # Définir les features du dataset
        self._time_stage(
            "define_features",
            self.dataset_creator.define_features,
            include_transcription=include_transcription,
            speaker_labels=speaker_labels
        )
        
        # Préparer le dataset local
        local_dir = self._time_stage(
            "prepare_local_dataset",
            self.dataset_creator.prepare_local_dataset,
            metadata_file=metadata_file,
            audio_dir=audio_dir,
            copy_audio=True
        )
        
        # Créer la carte du dataset
        self._time_stage(
            "create_dataset_card",
            self.dataset_creator.create_dataset_card,
            dataset_description=dataset_description,
            dataset_citation=dataset_citation
        )
        
        # Créer la configuration du dataset
        self._time_stage(
            "create_dataset_config",
            self.dataset_creator.create_dataset_config
        )
        
        return local_dir
    
    def upload_dataset(self, 
                      metadata_file: Optional[Union[str, Path]] = None,
                      audio_dir: Optional[Union[str, Path]] = None,
                      incremental: bool = False,
                      batch_size: int = 50) -> Dict[str, Any]:
        """
        Téléverse un dataset vers Hugging Face.
        
        Entrées :
            metadata_file (Optional[Union[str, Path]]) : Chemin du fichier de métadonnées
            audio_dir (Optional[Union[str, Path]]) : Répertoire contenant les fichiers audio
            incremental (bool) : Si True, utilise l'upload incrémental
            batch_size (int) : Nombre de fichiers à téléverser par lot
        
        Sorties :
            Dict[str, Any] : Statistiques d'upload
        """
        if incremental and metadata_file:
            return self._time_stage(
                "upload_dataset_incrementally",
                self.dataset_uploader.upload_dataset_incrementally,
                metadata_file=metadata_file,
                audio_dir=audio_dir,
                batch_size=batch_size
            )
        else:
            return self._time_stage(
                "upload_dataset",
                self.dataset_uploader.upload_dataset,
                metadata_file=metadata_file,
                audio_dir=audio_dir
            )
    
    def run_full_upload(self, 
                       metadata_file: Union[str, Path],
                       audio_dir: Union[str, Path],
                       format_metadata: bool = True,
                       prepare_local: bool = True,
                       incremental: bool = False,
                       batch_size: int = 50,
                       include_transcription: bool = False,
                       dataset_description: str = "",
                       dataset_citation: str = "") -> Dict[str, Any]:
        """
        Exécute le processus complet d'upload vers Hugging Face.
        
        Entrées :
            metadata_file (Union[str, Path]) : Chemin du fichier de métadonnées
            audio_dir (Union[str, Path]) : Répertoire contenant les fichiers audio
            format_metadata (bool) : Si True, formate les métadonnées
            prepare_local (bool) : Si True, prépare un dataset local
            incremental (bool) : Si True, utilise l'upload incrémental
            batch_size (int) : Nombre de fichiers à téléverser par lot
            include_transcription (bool) : Si True, inclut un champ pour la transcription
            dataset_description (str) : Description du dataset
            dataset_citation (str) : Citation du dataset
        
        Sorties :
            Dict[str, Any] : Statistiques d'exécution
        """
        if not self.repo_id:
            logging.error("Aucun dépôt spécifié pour l'upload")
            return {"success": False, "error": "Aucun dépôt spécifié"}
        
        if not self.hf_client.is_authenticated:
            logging.error("Authentification Hugging Face requise")
            return {"success": False, "error": "Authentification requise"}
        
        self.execution_stats["start_time"] = time.time()
        
        # Journaliser le début de l'upload dans MongoDB
        mongo_doc_id = None
        if self.mongo_logger.is_connected():
            additional_info = {
                "repo_id": self.repo_id,
                "dataset_name": self.dataset_name,
                "dataset_language": self.dataset_language,
                "private": self.private,
                "incremental": incremental,
                "batch_size": batch_size,
                "include_transcription": include_transcription
            }
            
            mongo_doc_id = self.mongo_logger.log_upload_start(
                repo_id=self.repo_id,
                metadata_file=str(metadata_file),
                audio_dir=str(audio_dir),
                additional_info=additional_info
            )
            
            # Stocker l'ID du document pour les étapes suivantes
            if mongo_doc_id:
                self.execution_stats["mongo_doc_id"] = mongo_doc_id
        
        try:
            # 1. Formater les métadonnées si demandé
            formatted_metadata = metadata_file
            if format_metadata:
                formatted_metadata = self.prepare_metadata(
                    metadata_file=metadata_file,
                    audio_column="segment_file",
                    split_column="split",
                    speaker_column="speaker_id",
                    duration_column="duration"
                )
            
            # 2. Préparer le dataset local si demandé
            if prepare_local:
                self.prepare_dataset(
                    metadata_file=formatted_metadata,
                    audio_dir=audio_dir,
                    include_transcription=include_transcription,
                    dataset_description=dataset_description,
                    dataset_citation=dataset_citation
                )
            
            # 3. Téléverser le dataset
            upload_stats = self.upload_dataset(
                metadata_file=formatted_metadata,
                audio_dir=audio_dir,
                incremental=incremental,
                batch_size=batch_size
            )
            
            self.execution_stats["end_time"] = time.time()
            self.execution_stats["duration_sec"] = self.execution_stats["end_time"] - self.execution_stats["start_time"]
            self.execution_stats["upload_stats"] = upload_stats
            
            # Journaliser la fin de l'upload dans MongoDB
            if mongo_doc_id and self.mongo_logger.is_connected():
                self.mongo_logger.log_upload_complete(
                    doc_id=mongo_doc_id,
                    success=True,
                    stats=upload_stats
                )
            
            logging.info(f"Processus d'upload terminé en {self.execution_stats['duration_sec']:.2f}s")
            
            return {
                "success": True,
                "repo_id": self.repo_id,
                "dataset_name": self.dataset_name,
                "execution_stats": self.execution_stats,
                "mongo_doc_id": mongo_doc_id
            }
            
        except Exception as e:
            logging.error(f"Erreur lors du processus d'upload: {e}")
            self.execution_stats["end_time"] = time.time()
            self.execution_stats["duration_sec"] = self.execution_stats["end_time"] - self.execution_stats["start_time"]
            self.execution_stats["error"] = str(e)
            
            # Journaliser l'erreur dans MongoDB
            if mongo_doc_id and self.mongo_logger.is_connected():
                self.mongo_logger.log_upload_complete(
                    doc_id=mongo_doc_id,
                    success=False,
                    stats=self.execution_stats,
                    error=str(e)
                )
            
            return {
                "success": False,
                "error": str(e),
                "execution_stats": self.execution_stats,
                "mongo_doc_id": mongo_doc_id
            }
