"""
Module de journalisation MongoDB pour le processus d'upload vers Hugging Face.

Ce module fournit des fonctionnalités pour journaliser les différentes étapes
du processus d'upload vers Hugging Face dans MongoDB.

Classes principales :
- HFMongoLogger : Classe gérant la journalisation dans MongoDB
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pymongo import MongoClient
from dotenv import load_dotenv

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

class HFMongoLogger:
    """
    Classe pour la journalisation des opérations d'upload vers Hugging Face dans MongoDB.
    
    Cette classe permet de créer et de mettre à jour des documents dans MongoDB
    pour suivre le processus d'upload des fichiers audio vers Hugging Face.
    """
    
    def __init__(self, collection_name: str = "hf_uploads"):
        """
        Initialise la connexion à MongoDB.
        
        Entrées :
            collection_name (str) : Nom de la collection MongoDB à utiliser
        """
        load_dotenv()
        # Liste des URI à essayer dans l'ordre
        mongo_uris = [
            "mongodb://localhost:27017/",  # URI locale (hors docker) - essayée en premier
            os.getenv("MONGO_URI", "mongodb://mongodb:27017/")  # URI du service (docker)
        ]
        self.db_name = os.getenv("MONGO_DB", "scraper_db")
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        
        # Essayer chaque URI jusqu'à ce qu'une connexion réussisse
        for uri in mongo_uris:
            try:
                self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                # Vérifier que la connexion fonctionne
                self.client.server_info()
                self.db = self.client[self.db_name]
                self.collection = self.db[self.collection_name]
                logging.info(f"Connexion à MongoDB réussie avec l'URI: {uri}, collection: {self.collection_name}")
                self.mongo_uri = uri
                return
            except Exception as e:
                logging.warning(f"Échec de connexion à MongoDB avec l'URI {uri}: {e}")
        
        # Si aucune connexion n'a réussi
        logging.error("Aucune connexion MongoDB n'a réussi.")
        self.client = None
        self.db = None
        self.collection = None
    
    def is_connected(self) -> bool:
        """
        Vérifie si la connexion à MongoDB est établie.
        
        Sorties :
            bool : True si la connexion est établie, False sinon
        """
        if self.client is None:
            return False
        
        try:
            # Vérifier la connexion
            self.client.admin.command('ping')
            return True
        except Exception:
            return False
    
    def log_upload_start(self, 
                       repo_id: str, 
                       metadata_file: str,
                       audio_dir: str,
                       additional_info: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Journalise le début d'un processus d'upload.
        
        Entrées :
            repo_id (str) : Identifiant du dépôt Hugging Face
            metadata_file (str) : Chemin du fichier de métadonnées
            audio_dir (str) : Répertoire contenant les fichiers audio
            additional_info (Optional[Dict[str, Any]]) : Informations supplémentaires
        
        Sorties :
            Optional[str] : ID du document créé, ou None en cas d'échec
        """
        if not self.is_connected():
            logging.warning("Impossible de journaliser le début de l'upload: connexion MongoDB non établie")
            return None
        
        try:
            # Créer le document
            doc = {
                "repo_id": repo_id,
                "metadata_file": metadata_file,
                "audio_dir": audio_dir,
                "start_time": datetime.now(),
                "status": "started",
                "files": {
                    "total": 0,
                    "uploaded": 0,
                    "failed": 0
                },
                "stages": {},
                "additional_info": additional_info or {}
            }
            
            # Insérer le document
            result = self.collection.insert_one(doc)
            
            logging.info(f"Upload journalisé: {repo_id}, ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logging.error(f"Erreur lors de la journalisation du début de l'upload: {e}")
            return None
    
    def log_upload_complete(self, 
                          doc_id: str, 
                          success: bool,
                          stats: Dict[str, Any],
                          error: Optional[str] = None) -> bool:
        """
        Journalise la fin d'un processus d'upload.
        
        Entrées :
            doc_id (str) : ID du document MongoDB
            success (bool) : True si l'upload a réussi, False sinon
            stats (Dict[str, Any]) : Statistiques d'upload
            error (Optional[str]) : Message d'erreur en cas d'échec
        
        Sorties :
            bool : True si la journalisation a réussi, False sinon
        """
        if not self.is_connected():
            logging.warning("Impossible de journaliser la fin de l'upload: connexion MongoDB non établie")
            return False
        
        try:
            # Mettre à jour le document
            update_data = {
                "end_time": datetime.now(),
                "status": "completed" if success else "failed",
                "stats": stats
            }
            
            if error:
                update_data["error"] = error
            
            self.collection.update_one(
                {"_id": doc_id},
                {"$set": update_data}
            )
            
            logging.info(f"Fin d'upload journalisée: {doc_id}, success: {success}")
            return True
            
        except Exception as e:
            logging.error(f"Erreur lors de la journalisation de la fin de l'upload: {e}")
            return False
    
    def log_stage(self, 
                doc_id: str, 
                stage_name: str, 
                success: bool,
                details: Optional[Dict[str, Any]] = None,
                error: Optional[str] = None) -> bool:
        """
        Journalise une étape du processus d'upload.
        
        Entrées :
            doc_id (str) : ID du document MongoDB
            stage_name (str) : Nom de l'étape
            success (bool) : True si l'étape a réussi, False sinon
            details (Optional[Dict[str, Any]]) : Détails de l'étape
            error (Optional[str]) : Message d'erreur en cas d'échec
        
        Sorties :
            bool : True si la journalisation a réussi, False sinon
        """
        if not self.is_connected():
            logging.warning(f"Impossible de journaliser l'étape {stage_name}: connexion MongoDB non établie")
            return False
        
        try:
            # Préparer les données de l'étape
            stage_data = {
                "success": success,
                "timestamp": datetime.now()
            }
            
            if details:
                stage_data["details"] = details
            
            if error:
                stage_data["error"] = error
            
            # Mettre à jour le document
            self.collection.update_one(
                {"_id": doc_id},
                {"$set": {f"stages.{stage_name}": stage_data}}
            )
            
            logging.info(f"Étape journalisée: {doc_id}, {stage_name}, success: {success}")
            return True
            
        except Exception as e:
            logging.error(f"Erreur lors de la journalisation de l'étape {stage_name}: {e}")
            return False
    
    def log_file_upload(self, 
                       doc_id: str, 
                       file_path: str, 
                       success: bool,
                       details: Optional[Dict[str, Any]] = None,
                       error: Optional[str] = None) -> bool:
        """
        Journalise l'upload d'un fichier.
        
        Entrées :
            doc_id (str) : ID du document MongoDB
            file_path (str) : Chemin du fichier
            success (bool) : True si l'upload a réussi, False sinon
            details (Optional[Dict[str, Any]]) : Détails de l'upload
            error (Optional[str]) : Message d'erreur en cas d'échec
        
        Sorties :
            bool : True si la journalisation a réussi, False sinon
        """
        if not self.is_connected():
            logging.warning(f"Impossible de journaliser l'upload du fichier {file_path}: connexion MongoDB non établie")
            return False
        
        try:
            # Préparer les données du fichier
            file_data = {
                "file_path": file_path,
                "success": success,
                "timestamp": datetime.now()
            }
            
            if details:
                file_data["details"] = details
            
            if error:
                file_data["error"] = error
            
            # Mettre à jour le document
            self.collection.update_one(
                {"_id": doc_id},
                {
                    "$push": {"files.details": file_data},
                    "$inc": {
                        "files.total": 1,
                        f"files.{'uploaded' if success else 'failed'}": 1
                    }
                }
            )
            
            logging.info(f"Upload de fichier journalisé: {doc_id}, {file_path}, success: {success}")
            return True
            
        except Exception as e:
            logging.error(f"Erreur lors de la journalisation de l'upload du fichier {file_path}: {e}")
            return False
    
    def get_upload_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère le statut d'un processus d'upload.
        
        Entrées :
            doc_id (str) : ID du document MongoDB
        
        Sorties :
            Optional[Dict[str, Any]] : Statut de l'upload, ou None en cas d'échec
        """
        if not self.is_connected():
            logging.warning(f"Impossible de récupérer le statut de l'upload {doc_id}: connexion MongoDB non établie")
            return None
        
        try:
            # Récupérer le document
            doc = self.collection.find_one({"_id": doc_id})
            
            if doc:
                return doc
            else:
                logging.warning(f"Document non trouvé: {doc_id}")
                return None
            
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du statut de l'upload {doc_id}: {e}")
            return None
