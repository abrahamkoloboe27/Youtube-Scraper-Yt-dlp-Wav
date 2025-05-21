"""
Module client pour Azure Blob Storage.

Ce module fournit une classe pour interagir avec Azure Blob Storage,
permettant de gérer les conteneurs et les blobs.
"""

import os
import logging
from typing import Optional, List
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/azure.log'),
        logging.StreamHandler()
    ]
)

class AzureClient:
    """
    Client pour interagir avec Azure Blob Storage.
    
    Cette classe fournit des méthodes pour se connecter à Azure Blob Storage
    et gérer les conteneurs et les blobs.
    """
    
    def __init__(self, 
                account_url: Optional[str] = None,
                credential: Optional[str] = None):
        """
        Initialise le client Azure Blob Storage.
        
        Entrées :
            account_url (Optional[str]) : URL du compte de stockage Azure
            credential (Optional[str]) : Jeton SAS ou clé de compte pour l'authentification
        """
        load_dotenv()
        
        self.account_url = account_url or os.getenv("AZURE_ACCOUNT_URL")
        self.credential = credential or os.getenv("AZURE_SAS_TOKEN")
        
        if not self.account_url or not self.credential:
            logging.warning("URL du compte ou identifiants Azure manquants")
        
        self.client = self._create_client()
    
    def _create_client(self) -> Optional[BlobServiceClient]:
        """
        Crée et retourne un client Azure Blob Storage.
        
        Sorties :
            Optional[BlobServiceClient] : Client Azure Blob Storage configuré, ou None en cas d'erreur
        """
        try:
            if not self.account_url or not self.credential:
                return None
                
            client = BlobServiceClient(
                account_url=self.account_url,
                credential=self.credential
            )
            logging.info(f"Client Azure Blob Storage créé avec succès pour le compte: {self.account_url}")
            return client
        except Exception as e:
            logging.error(f"Erreur lors de la création du client Azure Blob Storage: {e}")
            return None
    
    def is_connected(self) -> bool:
        """
        Vérifie si le client est connecté à Azure Blob Storage.
        
        Sorties :
            bool : True si le client est connecté, False sinon
        """
        if not self.client:
            return False
            
        try:
            # Tester la connexion en listant les conteneurs
            next(self.client.list_containers(max_results=1), None)
            return True
        except Exception:
            return False
    
    def container_exists(self, container_name: str) -> bool:
        """
        Vérifie si un conteneur existe.
        
        Entrées :
            container_name (str) : Nom du conteneur à vérifier
        Sorties :
            bool : True si le conteneur existe, False sinon
        """
        if not self.client:
            return False
            
        try:
            container_client = self.client.get_container_client(container_name)
            return container_client.exists()
        except Exception as e:
            logging.error(f"Erreur lors de la vérification du conteneur {container_name}: {e}")
            return False
    
    def create_container(self, container_name: str) -> bool:
        """
        Crée un conteneur s'il n'existe pas.
        
        Entrées :
            container_name (str) : Nom du conteneur à créer
        Sorties :
            bool : True si le conteneur a été créé ou existe déjà, False en cas d'erreur
        """
        if not self.client:
            return False
            
        try:
            if not self.container_exists(container_name):
                container_client = self.client.create_container(container_name)
                logging.info(f"Conteneur créé: {container_name}")
                return True
            return True
        except Exception as e:
            logging.error(f"Erreur lors de la création du conteneur {container_name}: {e}")
            return False
    
    def blob_exists(self, container_name: str, blob_name: str) -> bool:
        """
        Vérifie si un blob existe dans un conteneur.
        
        Entrées :
            container_name (str) : Nom du conteneur
            blob_name (str) : Nom du blob à vérifier
        Sorties :
            bool : True si le blob existe, False sinon
        """
        if not self.client:
            return False
            
        try:
            # Vérifier d'abord si le conteneur existe
            if not self.container_exists(container_name):
                return False
                
            # Vérifier si le blob existe
            blob_client = self.client.get_blob_client(container=container_name, blob=blob_name)
            return blob_client.exists()
        except Exception as e:
            logging.error(f"Erreur lors de la vérification du blob {blob_name} dans le conteneur {container_name}: {e}")
            return False
    
    def list_blobs(self, container_name: str, name_starts_with: str = None) -> List[str]:
        """
        Liste les blobs dans un conteneur.
        
        Entrées :
            container_name (str) : Nom du conteneur
            name_starts_with (str) : Préfixe pour filtrer les blobs
        Sorties :
            List[str] : Liste des noms de blobs
        """
        if not self.client:
            return []
            
        try:
            if not self.container_exists(container_name):
                return []
                
            container_client = self.client.get_container_client(container_name)
            blobs = container_client.list_blobs(name_starts_with=name_starts_with)
            return [blob.name for blob in blobs]
        except Exception as e:
            logging.error(f"Erreur lors de la liste des blobs dans le conteneur {container_name}: {e}")
            return []
    
    def get_container_client(self, container_name: str) -> Optional[ContainerClient]:
        """
        Retourne un client pour un conteneur spécifique.
        
        Entrées :
            container_name (str) : Nom du conteneur
        Sorties :
            Optional[ContainerClient] : Client de conteneur, ou None en cas d'erreur
        """
        if not self.client:
            return None
            
        try:
            return self.client.get_container_client(container_name)
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du client pour le conteneur {container_name}: {e}")
            return None
    
    def get_blob_client(self, container_name: str, blob_name: str) -> Optional[BlobClient]:
        """
        Retourne un client pour un blob spécifique.
        
        Entrées :
            container_name (str) : Nom du conteneur
            blob_name (str) : Nom du blob
        Sorties :
            Optional[BlobClient] : Client de blob, ou None en cas d'erreur
        """
        if not self.client:
            return None
            
        try:
            return self.client.get_blob_client(container=container_name, blob=blob_name)
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du client pour le blob {blob_name}: {e}")
            return None
