#!/usr/bin/env python3
"""
Module pour l'upload des données vers Hugging Face.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import json
import shutil
import time
import random
from tqdm import tqdm
from .hf_client import HFClient
from processing.mongo_logger import MongoLogger

class HFUploader:
    """
    Classe pour l'upload des données vers Hugging Face.
    
    Cette classe permet de gérer l'upload des fichiers audio traités et des métadonnées
    vers un dépôt Hugging Face.
    """
    
    def __init__(self, 
                 repo_id: Optional[str] = None,
                 auth_token: Optional[str] = None,
                 temp_dir: str = "temp_hf"):
        """
        Initialise l'uploader Hugging Face avec les paramètres spécifiés.
        
        Entrées :
            repo_id (Optional[str]) : ID du dépôt Hugging Face (format: 'username/repo_name')
            auth_token (Optional[str]) : Token d'authentification Hugging Face
            temp_dir (str) : Répertoire temporaire pour la préparation des données
        """
        self.repo_id = repo_id or os.getenv("HUGGINGFACE_REPO_ID")
        self.auth_token = auth_token or os.getenv("HUGGINGFACE_TOKEN")
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MongoLogger()
        
        # Initialiser le client Hugging Face
        self.client = HFClient(auth_token=self.auth_token)
        
        logging.info(f"HFUploader initialisé pour le dépôt: {self.repo_id}")
    
    def prepare_dataset(self, 
                       processed_dir: Union[str, Path],
                       metadata_file: Optional[Union[str, Path]] = None) -> Path:
        """
        Prépare les données pour l'upload vers Hugging Face.
        
        Entrées :
            processed_dir (Union[str, Path]) : Répertoire contenant les fichiers traités
            metadata_file (Optional[Union[str, Path]]) : Fichier de métadonnées (optionnel)
        
        Sorties :
            Path : Chemin du répertoire temporaire contenant les données préparées
        """
        processed_dir = Path(processed_dir)
        
        # Créer un répertoire temporaire pour cette préparation
        temp_dataset_dir = self.temp_dir / f"dataset_{int(time.time())}"
        temp_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer les sous-répertoires nécessaires
        audio_dir = temp_dataset_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_dir = temp_dataset_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Copier les fichiers audio
        audio_files = list(processed_dir.glob("**/*.wav"))
        audio_metadata = []
        
        for audio_file in tqdm(audio_files, desc="Préparation des fichiers audio"):
            # Copier le fichier audio
            dest_file = audio_dir / audio_file.name
            shutil.copy2(audio_file, dest_file)
            
            # Récupérer les métadonnées du fichier depuis MongoDB
            metadata = self.logger.get_segment_metadata(audio_file.name)
            if metadata:
                audio_metadata.append({
                    "file_name": audio_file.name,
                    "path": f"audio/{audio_file.name}",
                    "metadata": metadata
                })
        
        # Créer le fichier de métadonnées global
        dataset_metadata = {
            "dataset_name": self.repo_id.split("/")[-1] if self.repo_id else "fongbe_dataset",
            "version": time.strftime("%Y%m%d_%H%M%S"),
            "description": "Dataset audio en langue Fongbè",
            "language": "fon",
            "license": "CC-BY-4.0",
            "files": audio_metadata
        }
        
        # Sauvegarder les métadonnées
        with open(metadata_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(dataset_metadata, f, ensure_ascii=False, indent=2)
        
        # Si un fichier de métadonnées externe est fourni, le copier également
        if metadata_file:
            metadata_file = Path(metadata_file)
            if metadata_file.exists():
                shutil.copy2(metadata_file, metadata_dir / metadata_file.name)
        
        logging.info(f"Préparation terminée: {len(audio_files)} fichiers audio prêts pour l'upload")
        return temp_dataset_dir
    
    def upload_dataset(self, 
                      dataset_dir: Union[str, Path],
                      commit_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload les données préparées vers Hugging Face.
        
        Entrées :
            dataset_dir (Union[str, Path]) : Répertoire contenant les données préparées
            commit_message (Optional[str]) : Message de commit
        
        Sorties :
            Dict[str, Any] : Résultats de l'upload
        """
        dataset_dir = Path(dataset_dir)
        
        if not self.repo_id:
            raise ValueError("Aucun ID de dépôt Hugging Face spécifié")
        
        # Préparer le message de commit
        if not commit_message:
            commit_message = f"Upload de données audio - {time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Récupérer la liste des fichiers à uploader
        files_to_upload = list(dataset_dir.glob("**/*.*"))
        
        # Initialiser les statistiques d'upload
        upload_stats = {
            "total_files": len(files_to_upload),
            "uploaded_files": 0,
            "failed_files": 0,
            "total_size_bytes": 0,
            "start_time": time.time(),
            "end_time": None,
            "duration_seconds": None
        }
        
        # Créer un document dans MongoDB pour suivre l'upload
        upload_doc_id = self.logger.create_hf_upload_log({
            "repo_id": self.repo_id,
            "dataset_dir": str(dataset_dir),
            "commit_message": commit_message,
            "status": "started",
            "stats": upload_stats,
            "timestamp": time.time()
        })
        
        try:
            # Upload chaque fichier
            for file_path in tqdm(files_to_upload, desc=f"Upload vers {self.repo_id}"):
                try:
                    # Calculer le chemin relatif pour l'upload
                    relative_path = file_path.relative_to(dataset_dir)
                    
                    # Upload le fichier
                    self.client.upload_file(
                        repo_id=self.repo_id,
                        local_path=str(file_path),
                        repo_path=str(relative_path)
                    )
                    
                    # Mettre à jour les statistiques
                    upload_stats["uploaded_files"] += 1
                    upload_stats["total_size_bytes"] += file_path.stat().st_size
                    
                    # Enregistrer le fichier uploadé dans MongoDB
                    self.logger.add_hf_uploaded_file(
                        upload_id=upload_doc_id,
                        file_path=str(file_path),
                        repo_path=str(relative_path),
                        success=True
                    )
                    
                except Exception as e:
                    logging.error(f"Erreur lors de l'upload de {file_path}: {e}")
                    upload_stats["failed_files"] += 1
                    
                    # Enregistrer l'échec dans MongoDB
                    self.logger.add_hf_uploaded_file(
                        upload_id=upload_doc_id,
                        file_path=str(file_path),
                        repo_path=str(relative_path),
                        success=False,
                        error=str(e)
                    )
            
            # Finaliser l'upload avec un commit
            self.client.commit_changes(
                repo_id=self.repo_id,
                commit_message=commit_message
            )
            
            # Mettre à jour les statistiques finales
            upload_stats["end_time"] = time.time()
            upload_stats["duration_seconds"] = upload_stats["end_time"] - upload_stats["start_time"]
            
            # Mettre à jour le document MongoDB
            self.logger.update_hf_upload_log(
                upload_id=upload_doc_id,
                status="completed",
                stats=upload_stats
            )
            
            logging.info(f"Upload terminé: {upload_stats['uploaded_files']}/{upload_stats['total_files']} "
                        f"fichiers uploadés en {upload_stats['duration_seconds']:.2f} secondes")
            
            return {
                "success": True,
                "repo_id": self.repo_id,
                "stats": upload_stats
            }
            
        except Exception as e:
            logging.error(f"Erreur lors de l'upload vers {self.repo_id}: {e}")
            
            # Mettre à jour le document MongoDB
            self.logger.update_hf_upload_log(
                upload_id=upload_doc_id,
                status="failed",
                error=str(e)
            )
            
            return {
                "success": False,
                "repo_id": self.repo_id,
                "error": str(e)
            }
    
    def upload_processed_directory(self, 
                                 processed_dir: Union[str, Path],
                                 metadata_file: Optional[Union[str, Path]] = None,
                                 commit_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Prépare et upload un répertoire de fichiers traités vers Hugging Face.
        
        Entrées :
            processed_dir (Union[str, Path]) : Répertoire contenant les fichiers traités
            metadata_file (Optional[Union[str, Path]]) : Fichier de métadonnées (optionnel)
            commit_message (Optional[str]) : Message de commit
        
        Sorties :
            Dict[str, Any] : Résultats de l'upload
        """
        # Préparer les données
        dataset_dir = self.prepare_dataset(processed_dir, metadata_file)
        
        # Upload les données
        result = self.upload_dataset(dataset_dir, commit_message)
        
        # Nettoyer le répertoire temporaire si l'upload a réussi
        if result["success"]:
            shutil.rmtree(dataset_dir, ignore_errors=True)
        
        return result
