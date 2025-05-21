"""
Module client pour MinIO.

Ce module fournit une classe pour interagir avec le stockage objet MinIO,
permettant de gérer les buckets et les objets.
"""

import os
import logging
from datetime import datetime
from typing import Optional
from minio import Minio
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ││ %(levelname)s ││ %(name)s ││ %(message)s ",
    datefmt="%Y-%m-%d %H:%M:%S",

    handlers=[
        logging.FileHandler('logs/minio.log'),
        logging.StreamHandler()
    ]
)

class MinioClient:
    """
    Client pour interagir avec MinIO.
    
    Cette classe fournit des méthodes pour se connecter à MinIO et gérer les buckets et objets.
    """
    
    def __init__(self, 
                endpoint: Optional[str] = None,
                access_key: Optional[str] = None,
                secret_key: Optional[str] = None,
                secure: bool = False):
        """
        Initialise le client MinIO.
        Essaie d'abord l'endpoint du service, puis localhost en cas d'échec.
        
        Entrées :
            endpoint (Optional[str]) : Point d'accès MinIO (hôte:port)
            access_key (Optional[str]) : Clé d'accès MinIO
            secret_key (Optional[str]) : Clé secrète MinIO
            secure (bool) : Utiliser HTTPS si True
        """
        load_dotenv()
        
        # Liste des endpoints à essayer dans l'ordre
        self.endpoints = [
            "localhost:9000" ,  # Endpoint local (hors docker)
            endpoint or os.getenv("MINIO_ENDPOINT", "minio:9000"),  # Endpoint du service (docker)
            
        ]
        self.access_key = access_key or os.getenv("MINIO_ROOT_USER", "minioadmin")
        self.secret_key = secret_key or os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
        self.secure = secure
        
        # Essayer de créer un client avec chaque endpoint
        self.client = None
        self.endpoint = None
        self.in_memory_mode = False
        
        self._create_client()
    
    def _create_client(self) -> Optional[Minio]:
        """
        Crée et retourne un client MinIO.
        Essaie chaque endpoint jusqu'à ce qu'une connexion réussisse.
        Si aucune connexion ne réussit, utilise un mode "en mémoire".
        
        Sorties :
            Optional[Minio] : Client MinIO configuré ou None en cas d'échec
        """
        # Essayer chaque endpoint jusqu'à ce qu'une connexion réussisse
        last_error = None
        for endpoint in self.endpoints:
            try:
                self.endpoint = endpoint
                client = Minio(
                    self.endpoint,
                    access_key=self.access_key,
                    secret_key=self.secret_key,
                    secure=self.secure
                )
                
                # Vérifier que la connexion fonctionne en listant les buckets
                # Cette opération va échouer si le client ne peut pas se connecter
                list(client.list_buckets())
                
                logging.info(f"Client MinIO créé avec succès pour l'endpoint: {self.endpoint}")
                self.client = client
                self.in_memory_mode = False
                return client
            except Exception as e:
                last_error = e
                logging.warning(f"Échec de connexion à MinIO avec l'endpoint {endpoint}: {e}")
        
        # Si aucune connexion n'a réussi, utiliser une version en mémoire (pour les tests)
        logging.warning("Aucune connexion MinIO n'a réussi. Utilisation d'une version en mémoire pour les tests.")
        self.in_memory_mode = True
        self.in_memory_buckets = {}
        return None
    
    def bucket_exists(self, bucket_name: str) -> bool:
        """
        Vérifie si un bucket existe.
        
        Entrées :
            bucket_name (str) : Nom du bucket à vérifier
        Sorties :
            bool : True si le bucket existe, False sinon
        """
        # Mode en mémoire (pour les tests sans MinIO)
        if self.in_memory_mode:
            return bucket_name in self.in_memory_buckets
        
        # Mode MinIO
        try:
            return self.client.bucket_exists(bucket_name)
        except Exception as e:
            logging.error(f"Erreur lors de la vérification du bucket {bucket_name}: {e}")
            return False
    
    def create_bucket(self, bucket_name: str) -> bool:
        """
        Crée un bucket s'il n'existe pas.
        
        Entrées :
            bucket_name (str) : Nom du bucket à créer
        Sorties :
            bool : True si le bucket a été créé ou existe déjà, False en cas d'erreur
        """
        # Mode en mémoire (pour les tests sans MinIO)
        if self.in_memory_mode:
            if not self.bucket_exists(bucket_name):
                self.in_memory_buckets[bucket_name] = {}
                logging.info(f"Bucket créé en mémoire: {bucket_name}")
            return True
        
        # Mode MinIO
        try:
            if not self.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logging.info(f"Bucket créé: {bucket_name}")
            return True
        except Exception as e:
            logging.error(f"Erreur lors de la création du bucket {bucket_name}: {e}")
            return False
    
    def object_exists(self, bucket_name: str, object_name: str) -> bool:
        """
        Vérifie si un objet existe dans un bucket.
        
        Entrées :
            bucket_name (str) : Nom du bucket
            object_name (str) : Nom de l'objet à vérifier
        Sorties :
            bool : True si l'objet existe, False sinon
        """
        # Mode en mémoire (pour les tests sans MinIO)
        if self.in_memory_mode:
            # Vérifier d'abord si le bucket existe
            if not self.bucket_exists(bucket_name):
                return False
            
            # Vérifier si l'objet existe dans le bucket
            return object_name in self.in_memory_buckets[bucket_name]
        
        # Mode MinIO
        try:
            # Vérifier d'abord si le bucket existe
            if not self.bucket_exists(bucket_name):
                return False
            
            # Essayer de récupérer les stats de l'objet
            try:
                self.client.stat_object(bucket_name, object_name)
                return True
            except Exception:
                return False
                
        except Exception as e:
            logging.error(f"Erreur lors de la vérification de l'objet {object_name} dans le bucket {bucket_name}: {e}")
            return False
    
    def list_objects(self, bucket_name: str, prefix: str = "", recursive: bool = True):
        """
        Liste les objets dans un bucket.
        
        Entrées :
            bucket_name (str) : Nom du bucket
            prefix (str) : Préfixe pour filtrer les objets
            recursive (bool) : Recherche récursive si True
        Sorties :
            Generator : Générateur d'objets MinIO
        """
        # Mode en mémoire (pour les tests sans MinIO)
        if self.in_memory_mode:
            if not self.bucket_exists(bucket_name):
                return []
            
            # Filtrer les objets par préfixe
            objects = []
            for obj_name in self.in_memory_buckets[bucket_name].keys():
                if obj_name.startswith(prefix):
                    # Créer un objet similaire à ceux retournés par MinIO
                    obj = type('obj', (), {
                        'object_name': obj_name,
                        'size': len(self.in_memory_buckets[bucket_name][obj_name]),
                        'last_modified': datetime.now()
                    })
                    objects.append(obj)
            return objects
        
        # Mode MinIO
        try:
            if not self.bucket_exists(bucket_name):
                return []
                
            return self.client.list_objects(bucket_name, prefix=prefix, recursive=recursive)
        except Exception as e:
            logging.error(f"Erreur lors de la liste des objets dans le bucket {bucket_name}: {e}")
            return []
