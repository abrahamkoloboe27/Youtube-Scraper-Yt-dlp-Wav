import os
from pymongo import MongoClient

def get_mongo_client():
    uri = os.getenv("MONGO_URI", "mongodb://mongodb:27017/")
    return MongoClient(uri)

def get_db():
    client = get_mongo_client()
    db_name = os.getenv("MONGO_DB", "scraper_db")
    return client[db_name]

def log_download(metadata: dict):
    db = get_db()
    db.downloads.insert_one(metadata)

def log_failed_download(metadata: dict):
    db = get_db()
    db.failed_downloads.insert_one(metadata)

def get_failed_downloads():
    db = get_db()
    return list(db.failed_downloads.find({}))
