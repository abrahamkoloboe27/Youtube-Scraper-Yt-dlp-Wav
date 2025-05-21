"""
Module client pour l'API Hugging Face.

Ce module fournit une interface pour interagir avec l'API Hugging Face,
permettant l'authentification et les opérations de base sur les datasets.

Classes principales :
- HFClient : Classe gérant la connexion à l'API Hugging Face
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import requests
from huggingface_hub import HfApi, HfFolder, Repository
from huggingface_hub.utils import validate_repo_id, RepositoryNotFoundError
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

class HFClient:
    """
    Classe pour interagir avec l'API Hugging Face.
    
    Cette classe fournit des méthodes pour se connecter à l'API Hugging Face,
    créer et gérer des datasets, et effectuer des opérations de base.
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialise le client Hugging Face avec le token d'authentification.
        
        Entrées :
            token (Optional[str]) : Token d'authentification Hugging Face
                Si None, essaie de le récupérer depuis les variables d'environnement
        """
        load_dotenv()
        
        # Récupérer le token depuis les variables d'environnement si non fourni
        self.token = token or os.getenv("HF_TOKEN")
        
        if not self.token:
            logging.warning("Aucun token Hugging Face trouvé. Certaines fonctionnalités seront limitées.")
        
        # Initialiser l'API Hugging Face
        self.api = HfApi(token=self.token)
        
        # Vérifier l'authentification
        self.is_authenticated = self._check_authentication()
        
        if self.is_authenticated:
            logging.info("Authentification Hugging Face réussie")
        else:
            logging.warning("Échec de l'authentification Hugging Face")
    
    def _check_authentication(self) -> bool:
        """
        Vérifie si l'authentification à Hugging Face est valide.
        
        Sorties :
            bool : True si l'authentification est valide, False sinon
        """
        if not self.token:
            return False
        
        try:
            # Essayer de récupérer les informations du compte
            whoami = self.api.whoami()
            return whoami.get("name") is not None
        except Exception as e:
            logging.error(f"Erreur lors de la vérification de l'authentification: {e}")
            return False
    
    def create_repo(self, 
                   repo_id: str, 
                   private: bool = False, 
                   repo_type: str = "dataset") -> bool:
        """
        Crée un nouveau dépôt sur Hugging Face.
        
        Entrées :
            repo_id (str) : Identifiant du dépôt (format: 'username/repo_name')
            private (bool) : Si True, le dépôt sera privé
            repo_type (str) : Type de dépôt ('dataset' ou 'model')
        
        Sorties :
            bool : True si la création a réussi, False sinon
        """
        if not self.is_authenticated:
            logging.error("Authentification requise pour créer un dépôt")
            return False
        
        try:
            # Valider l'identifiant du dépôt
            validate_repo_id(repo_id)
            
            # Créer le dépôt
            self.api.create_repo(
                repo_id=repo_id,
                private=private,
                repo_type=repo_type,
                exist_ok=True
            )
            
            logging.info(f"Dépôt créé avec succès: {repo_id}")
            return True
            
        except Exception as e:
            logging.error(f"Erreur lors de la création du dépôt {repo_id}: {e}")
            return False
    
    def check_repo_exists(self, repo_id: str, repo_type: str = "dataset") -> bool:
        """
        Vérifie si un dépôt existe sur Hugging Face.
        
        Entrées :
            repo_id (str) : Identifiant du dépôt (format: 'username/repo_name')
            repo_type (str) : Type de dépôt ('dataset' ou 'model')
        
        Sorties :
            bool : True si le dépôt existe, False sinon
        """
        try:
            # Valider l'identifiant du dépôt
            validate_repo_id(repo_id)
            
            # Vérifier si le dépôt existe
            self.api.repo_info(repo_id=repo_id, repo_type=repo_type)
            return True
            
        except RepositoryNotFoundError:
            return False
            
        except Exception as e:
            logging.error(f"Erreur lors de la vérification du dépôt {repo_id}: {e}")
            return False
    
    def clone_repo(self, 
                  repo_id: str, 
                  local_dir: Union[str, Path], 
                  repo_type: str = "dataset") -> Optional[Repository]:
        """
        Clone un dépôt Hugging Face en local.
        
        Entrées :
            repo_id (str) : Identifiant du dépôt (format: 'username/repo_name')
            local_dir (Union[str, Path]) : Répertoire local où cloner le dépôt
            repo_type (str) : Type de dépôt ('dataset' ou 'model')
        
        Sorties :
            Optional[Repository] : Objet Repository si le clonage a réussi, None sinon
        """
        if not self.is_authenticated:
            logging.error("Authentification requise pour cloner un dépôt")
            return None
        
        try:
            # Valider l'identifiant du dépôt
            validate_repo_id(repo_id)
            
            # Cloner le dépôt
            repo = Repository(
                local_dir=str(local_dir),
                clone_from=repo_id,
                repo_type=repo_type,
                use_auth_token=self.token
            )
            
            logging.info(f"Dépôt cloné avec succès: {repo_id} -> {local_dir}")
            return repo
            
        except Exception as e:
            logging.error(f"Erreur lors du clonage du dépôt {repo_id}: {e}")
            return None
    
    def upload_file(self, 
                   repo_id: str, 
                   file_path: Union[str, Path], 
                   path_in_repo: Optional[str] = None,
                   repo_type: str = "dataset") -> bool:
        """
        Téléverse un fichier vers un dépôt Hugging Face.
        
        Entrées :
            repo_id (str) : Identifiant du dépôt (format: 'username/repo_name')
            file_path (Union[str, Path]) : Chemin du fichier à téléverser
            path_in_repo (Optional[str]) : Chemin dans le dépôt où placer le fichier
            repo_type (str) : Type de dépôt ('dataset' ou 'model')
        
        Sorties :
            bool : True si le téléversement a réussi, False sinon
        """
        if not self.is_authenticated:
            logging.error("Authentification requise pour téléverser un fichier")
            return False
        
        try:
            file_path = Path(file_path)
            
            # Si path_in_repo n'est pas spécifié, utiliser le nom du fichier
            if path_in_repo is None:
                path_in_repo = file_path.name
            
            # Téléverser le fichier
            self.api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type
            )
            
            logging.info(f"Fichier téléversé avec succès: {file_path} -> {repo_id}/{path_in_repo}")
            return True
            
        except Exception as e:
            logging.error(f"Erreur lors du téléversement de {file_path} vers {repo_id}: {e}")
            return False
    
    def upload_folder(self, 
                     repo_id: str, 
                     folder_path: Union[str, Path], 
                     path_in_repo: str = "",
                     repo_type: str = "dataset",
                     ignore_patterns: Optional[List[str]] = None) -> bool:
        """
        Téléverse un dossier entier vers un dépôt Hugging Face.
        
        Entrées :
            repo_id (str) : Identifiant du dépôt (format: 'username/repo_name')
            folder_path (Union[str, Path]) : Chemin du dossier à téléverser
            path_in_repo (str) : Chemin dans le dépôt où placer le dossier
            repo_type (str) : Type de dépôt ('dataset' ou 'model')
            ignore_patterns (Optional[List[str]]) : Motifs de fichiers à ignorer
        
        Sorties :
            bool : True si le téléversement a réussi, False sinon
        """
        if not self.is_authenticated:
            logging.error("Authentification requise pour téléverser un dossier")
            return False
        
        try:
            folder_path = Path(folder_path)
            
            # Vérifier que le dossier existe
            if not folder_path.exists() or not folder_path.is_dir():
                logging.error(f"Le dossier {folder_path} n'existe pas")
                return False
            
            # Téléverser le dossier
            if ignore_patterns is None:
                ignore_patterns = [".git", ".gitignore", "__pycache__", "*.pyc"]
            
            self.api.upload_folder(
                folder_path=str(folder_path),
                repo_id=repo_id,
                repo_type=repo_type,
                path_in_repo=path_in_repo,
                ignore_patterns=ignore_patterns
            )
            
            logging.info(f"Dossier téléversé avec succès: {folder_path} -> {repo_id}/{path_in_repo}")
            return True
            
        except Exception as e:
            logging.error(f"Erreur lors du téléversement du dossier {folder_path} vers {repo_id}: {e}")
            return False
