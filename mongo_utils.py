"""
Utilitaires pour l'interaction avec la base de données MongoDB.

Ce script fournit des fonctions pour obtenir un client MongoDB, accéder à la base, et journaliser les téléchargements réussis ou échoués.

Fonctions principales :
- get_mongo_client : Retourne un client MongoDB.
- get_db : Retourne la base de données cible.
- log_download : Journalise un téléchargement réussi.
- log_failed_download : Journalise un téléchargement échoué.
- get_failed_downloads : Retourne la liste des téléchargements échoués.
"""
import os
from pymongo import MongoClient

def get_mongo_client():
    """
    Retourne un client MongoDB à partir de la variable d'environnement MONGO_URI.

    Entrées :
        Néant (utilise la variable d'environnement MONGO_URI)
    Sorties :
        MongoClient : client MongoDB prêt à l'emploi
    """
    uri = os.getenv("MONGO_URI", "mongodb://mongodb:27017/")
    return MongoClient(uri)

def get_db():
    """
    Retourne la base de données MongoDB cible.

    Entrées :
        Néant (utilise la variable d'environnement MONGO_DB)
    Sorties :
        Database : objet base de données MongoDB
    """
    client = get_mongo_client()
    db_name = os.getenv("MONGO_DB", "scraper_db")
    return client[db_name]

def log_download(metadata: dict) -> None:
    """
    Journalise un téléchargement réussi dans la collection 'downloads'.

    Entrées :
        metadata (dict) : métadonnées du téléchargement
    Sorties :
        None
    """
    db = get_db()
    db.downloads.insert_one(metadata)

def log_failed_download(metadata: dict) -> None:
    """
    Journalise un téléchargement échoué dans la collection 'failed_downloads'.

    Entrées :
        metadata (dict) : métadonnées de l'échec
    Sorties :
        None
    """
    db = get_db()
    db.failed_downloads.insert_one(metadata)

def get_failed_downloads() -> list:
    """
    Retourne la liste de tous les téléchargements échoués.

    Entrées :
        Néant
    Sorties :
        list : liste des documents représentant les téléchargements échoués
    """
    db = get_db()
    return list(db.failed_downloads.find({}))
