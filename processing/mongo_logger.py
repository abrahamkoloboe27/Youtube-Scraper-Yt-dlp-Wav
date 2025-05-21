"""
Module de journalisation MongoDB pour le pipeline de traitement audio.

Ce module fournit des fonctionnalités pour journaliser les différentes étapes du traitement audio
dans MongoDB, permettant de suivre l'historique complet du traitement de chaque fichier audio.

Classes principales :
- MongoLogger : Classe gérant la journalisation dans MongoDB
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from pymongo import MongoClient


class MongoLogger:
    """
    Classe pour la journalisation des opérations de traitement audio dans MongoDB.
    
    Cette classe permet de créer et de mettre à jour des documents dans MongoDB
    pour suivre le traitement des fichiers audio à travers les différentes étapes
    du pipeline.
    """
    
    def __init__(self):
        """
        Initialise la connexion à MongoDB.
        """
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://mongodb:27017/")
        self.db_name = os.getenv("MONGO_DB", "scraper_db")
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db["audio_processing"]
    
    def create_audio_document(self, audio_file: str, original_metadata: Dict[str, Any]) -> str:
        """
        Crée un nouveau document dans la collection pour un fichier audio.
        
        Entrées :
            audio_file (str) : Chemin du fichier audio
            original_metadata (Dict[str, Any]) : Métadonnées originales du fichier
        
        Sorties :
            str : ID du document créé
        """
        file_name = Path(audio_file).name
        
        # Vérifier si le document existe déjà
        existing_doc = self.collection.find_one({"file": file_name})
        if existing_doc:
            return str(existing_doc["_id"])
        
        # Créer un nouveau document avec les informations de base
        doc = {
            "file": file_name,
            "original_metadata": original_metadata,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "processing_stages": {
                "loaded": False,
                "loudness_normalized": False,
                "silence_removed": False,
                "diarized": False,
                "segmented": False,
                "cleaned": False,
                "metadata_tagged": False,
                "augmented": False,
                "quality_checked": False,
                "exported": False
            },
            "processing_details": {},
            "segments": [],
            "augmentations": []
        }
        
        result = self.collection.insert_one(doc)
        return str(result.inserted_id)
    
    def update_stage(self, 
                     doc_id: str, 
                     stage_name: str, 
                     status: bool, 
                     details: Optional[Dict[str, Any]] = None) -> None:
        """
        Met à jour le statut d'une étape de traitement pour un fichier audio.
        
        Entrées :
            doc_id (str) : ID du document dans MongoDB
            stage_name (str) : Nom de l'étape de traitement
            status (bool) : Statut de l'étape (True pour réussi, False pour échec)
            details (Optional[Dict[str, Any]]) : Détails supplémentaires sur l'étape
        
        Sorties :
            None
        """
        update_data = {
            f"processing_stages.{stage_name}": status,
            "updated_at": datetime.now()
        }
        
        if details:
            update_data[f"processing_details.{stage_name}"] = details
        
        self.collection.update_one(
            {"_id": doc_id},
            {"$set": update_data}
        )
    
    def add_segment(self, 
                    doc_id: str, 
                    segment_file: str, 
                    speaker_id: str, 
                    start_time: float, 
                    end_time: float, 
                    metadata: Dict[str, Any]) -> None:
        """
        Ajoute un segment audio au document.
        
        Entrées :
            doc_id (str) : ID du document dans MongoDB
            segment_file (str) : Chemin du fichier segment
            speaker_id (str) : ID du locuteur
            start_time (float) : Temps de début du segment (en secondes)
            end_time (float) : Temps de fin du segment (en secondes)
            metadata (Dict[str, Any]) : Métadonnées du segment
        
        Sorties :
            None
        """
        segment = {
            "file": Path(segment_file).name,
            "speaker_id": speaker_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "metadata": metadata,
            "created_at": datetime.now()
        }
        
        self.collection.update_one(
            {"_id": doc_id},
            {
                "$push": {"segments": segment},
                "$set": {"updated_at": datetime.now()}
            }
        )
    
    def add_augmentation(self, 
                         doc_id: str, 
                         original_segment: str, 
                         augmented_file: str, 
                         augmentation_type: str, 
                         parameters: Dict[str, Any]) -> None:
        """
        Ajoute une augmentation de données au document.
        
        Entrées :
            doc_id (str) : ID du document dans MongoDB
            original_segment (str) : Chemin du fichier segment original
            augmented_file (str) : Chemin du fichier augmenté
            augmentation_type (str) : Type d'augmentation (speed, pitch, noise, etc.)
            parameters (Dict[str, Any]) : Paramètres de l'augmentation
        
        Sorties :
            None
        """
        augmentation = {
            "original_file": Path(original_segment).name,
            "augmented_file": Path(augmented_file).name,
            "type": augmentation_type,
            "parameters": parameters,
            "created_at": datetime.now()
        }
        
        self.collection.update_one(
            {"_id": doc_id},
            {
                "$push": {"augmentations": augmentation},
                "$set": {"updated_at": datetime.now()}
            }
        )
    
    def get_processing_status(self, file_name: str) -> Optional[Dict[str, Any]]:
        """
        Récupère le statut de traitement d'un fichier audio.
        
        Entrées :
            file_name (str) : Nom du fichier audio
        
        Sorties :
            Optional[Dict[str, Any]] : Document contenant le statut de traitement
        """
        return self.collection.find_one({"file": Path(file_name).name})
    
    def get_all_files_with_stage(self, stage_name: str, status: bool) -> List[Dict[str, Any]]:
        """
        Récupère tous les fichiers ayant un statut spécifique pour une étape donnée.
        
        Entrées :
            stage_name (str) : Nom de l'étape de traitement
            status (bool) : Statut recherché (True ou False)
        
        Sorties :
            List[Dict[str, Any]] : Liste des documents correspondant au critère
        """
        return list(self.collection.find({f"processing_stages.{stage_name}": status}))
