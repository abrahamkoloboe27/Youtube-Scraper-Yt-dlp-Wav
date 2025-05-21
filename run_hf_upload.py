#!/usr/bin/env python3
"""
Script principal pour l'upload des données audio Fongbè vers Hugging Face.

Ce script permet de téléverser facilement les fichiers audio traités et leurs métadonnées
vers un dataset Hugging Face depuis la ligne de commande.

Utilisation :
    python run_hf_upload.py --metadata_file /chemin/vers/metadata.csv --audio_dir /chemin/vers/audios --repo_id username/fongbe_dataset
    
Options :
    --metadata_file : Fichier de métadonnées (CSV ou Parquet)
    --audio_dir : Répertoire contenant les fichiers audio
    --repo_id : Identifiant du dépôt Hugging Face (format: 'username/repo_name')
    --hf_token : Token d'authentification Hugging Face (optionnel, peut être défini via HF_TOKEN)
    --config : Fichier de configuration JSON (optionnel)
    --local_dir : Répertoire local pour le dataset (optionnel)
    --dataset_name : Nom du dataset (optionnel)
    --dataset_language : Code de langue du dataset (optionnel)
    --private : Si spécifié, le dépôt sera privé
    --incremental : Si spécifié, utilise l'upload incrémental
    --batch_size : Nombre de fichiers à téléverser par lot
    --include_transcription : Si spécifié, inclut un champ pour la transcription
    --dataset_description : Description du dataset (optionnel)
    --dataset_citation : Citation du dataset (optionnel)
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from upload_hf.upload_manager import UploadManager
from upload_hf.mongo_logger import HFMongoLogger

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hf_upload.log'),
        logging.StreamHandler()
    ]
)

def parse_args():
    """
    Parse les arguments de la ligne de commande.
    
    Sorties :
        argparse.Namespace : Arguments parsés
    """
    parser = argparse.ArgumentParser(description="Upload des données audio Fongbè vers Hugging Face")
    
    parser.add_argument(
        "--metadata_file", 
        type=str, 
        required=True,
        help="Fichier de métadonnées (CSV ou Parquet)"
    )
    
    parser.add_argument(
        "--audio_dir", 
        type=str, 
        required=True,
        help="Répertoire contenant les fichiers audio"
    )
    
    parser.add_argument(
        "--repo_id", 
        type=str, 
        required=True,
        help="Identifiant du dépôt Hugging Face (format: 'username/repo_name')"
    )
    
    parser.add_argument(
        "--hf_token", 
        type=str, 
        default=None,
        help="Token d'authentification Hugging Face (optionnel, peut être défini via HF_TOKEN)"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Fichier de configuration JSON (optionnel)"
    )
    
    parser.add_argument(
        "--local_dir", 
        type=str, 
        default="hf_dataset",
        help="Répertoire local pour le dataset (optionnel)"
    )
    
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="fongbe_audio",
        help="Nom du dataset (optionnel)"
    )
    
    parser.add_argument(
        "--dataset_language", 
        type=str, 
        default="fon",
        help="Code de langue du dataset (optionnel)"
    )
    
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Si spécifié, le dépôt sera privé"
    )
    
    parser.add_argument(
        "--incremental", 
        action="store_true",
        help="Si spécifié, utilise l'upload incrémental"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=50,
        help="Nombre de fichiers à téléverser par lot"
    )
    
    parser.add_argument(
        "--include_transcription", 
        action="store_true",
        help="Si spécifié, inclut un champ pour la transcription"
    )
    
    parser.add_argument(
        "--dataset_description", 
        type=str, 
        default="",
        help="Description du dataset (optionnel)"
    )
    
    parser.add_argument(
        "--dataset_citation", 
        type=str, 
        default="",
        help="Citation du dataset (optionnel)"
    )
    
    parser.add_argument(
        "--skip_format_metadata", 
        action="store_true",
        help="Si spécifié, saute l'étape de formatage des métadonnées"
    )
    
    parser.add_argument(
        "--skip_prepare_local", 
        action="store_true",
        help="Si spécifié, saute l'étape de préparation locale du dataset"
    )
    
    return parser.parse_args()

def main():
    """
    Fonction principale du script.
    """
    # Charger les variables d'environnement
    load_dotenv()
    
    # Parser les arguments
    args = parse_args()
    
    # Vérifier que les fichiers et répertoires existent
    metadata_file = Path(args.metadata_file)
    if not metadata_file.exists():
        logging.error(f"Le fichier de métadonnées n'existe pas: {metadata_file}")
        sys.exit(1)
    
    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists() or not audio_dir.is_dir():
        logging.error(f"Le répertoire audio n'existe pas: {audio_dir}")
        sys.exit(1)
    
    # Charger la configuration si spécifiée
    config = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logging.error(f"Le fichier de configuration n'existe pas: {config_path}")
            sys.exit(1)
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logging.error(f"Erreur lors du chargement de la configuration: {e}")
            sys.exit(1)
    
    # Initialiser le logger MongoDB
    mongo_logger = HFMongoLogger(collection_name="hf_uploads")
    
    # Vérifier la connexion MongoDB
    if not mongo_logger.is_connected():
        logging.warning("Impossible de se connecter à MongoDB. La journalisation sera limitée aux fichiers de logs.")
    else:
        logging.info("Connexion à MongoDB établie avec succès.")
    
    # Initialiser le gestionnaire d'upload
    upload_manager = UploadManager(
        config_path=args.config,
        hf_token=args.hf_token,
        repo_id=args.repo_id,
        dataset_name=args.dataset_name,
        dataset_language=args.dataset_language,
        local_dir=args.local_dir,
        private=args.private,
        mongo_logger=mongo_logger
    )
    
    try:
        # Exécuter le processus d'upload
        result = upload_manager.run_full_upload(
            metadata_file=args.metadata_file,
            audio_dir=args.audio_dir,
            format_metadata=not args.skip_format_metadata,
            prepare_local=not args.skip_prepare_local,
            incremental=args.incremental,
            batch_size=args.batch_size,
            include_transcription=args.include_transcription,
            dataset_description=args.dataset_description,
            dataset_citation=args.dataset_citation
        )
        
        if result["success"]:
            logging.info(f"Upload terminé avec succès vers {args.repo_id}")
            
            # Afficher un résumé
            if "upload_stats" in result.get("execution_stats", {}):
                stats = result["execution_stats"]["upload_stats"]
                logging.info(f"Fichiers téléversés : {stats.get('uploaded_files', 0)}/{stats.get('total_files', 0)}")
                logging.info(f"Taille totale : {stats.get('total_size_mb', 0):.2f} MB")
                logging.info(f"Durée totale : {result['execution_stats'].get('duration_sec', 0):.2f}s")
                
                # Afficher l'ID du document MongoDB si disponible
                if "mongo_doc_id" in result and result["mongo_doc_id"]:
                    logging.info(f"ID du document MongoDB : {result['mongo_doc_id']}")
                    logging.info("Les détails complets de l'upload sont disponibles dans la collection MongoDB 'hf_uploads'.")
            
            sys.exit(0)
        else:
            logging.error(f"Échec de l'upload : {result.get('error', 'Erreur inconnue')}")
            sys.exit(1)
        
    except Exception as e:
        logging.error(f"Erreur lors de l'exécution du processus d'upload: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
