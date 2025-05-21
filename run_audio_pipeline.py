#!/usr/bin/env python3
"""
Script principal pour l'exécution du pipeline de traitement audio Fongbè.

Ce script permet de lancer facilement le pipeline de traitement audio Fongbè 
depuis la ligne de commande, en spécifiant les paramètres et options souhaités.

Utilisation :
    python run_audio_pipeline.py --input_dir /chemin/vers/audios --config config.json
    
Options :
    --input_dir : Répertoire contenant les fichiers audio à traiter
    --config : Fichier de configuration JSON (optionnel)
    --output_dir : Répertoire de sortie pour les fichiers traités
    --skip_stages : Étapes à sauter (séparées par des virgules)
    --only_stages : Seulement ces étapes (séparées par des virgules)
    --max_files : Nombre maximum de fichiers à traiter
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from processing.pipeline import AudioPipeline

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)

def parse_args():
    """
    Parse les arguments de la ligne de commande.
    
    Sorties :
        argparse.Namespace : Arguments parsés
    """
    parser = argparse.ArgumentParser(description="Pipeline de traitement audio Fongbè")
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="Répertoire contenant les fichiers audio à traiter"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Fichier de configuration JSON (optionnel)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="processed",
        help="Répertoire de sortie pour les fichiers traités"
    )
    
    parser.add_argument(
        "--skip_stages", 
        type=str, 
        default=None,
        help="Étapes à sauter (séparées par des virgules)"
    )
    
    parser.add_argument(
        "--only_stages", 
        type=str, 
        default=None,
        help="Seulement ces étapes (séparées par des virgules)"
    )
    
    parser.add_argument(
        "--max_files", 
        type=int, 
        default=None,
        help="Nombre maximum de fichiers à traiter"
    )
    
    return parser.parse_args()

def main():
    """
    Fonction principale du script.
    """
    # Parser les arguments
    args = parse_args()
    
    # Vérifier que le répertoire d'entrée existe
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logging.error(f"Le répertoire d'entrée n'existe pas: {input_dir}")
        sys.exit(1)
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Préparer les étapes à sauter ou à inclure
    skip_stages = args.skip_stages.split(",") if args.skip_stages else None
    only_stages = args.only_stages.split(",") if args.only_stages else None
    
    # Initialiser le pipeline
    pipeline = AudioPipeline(
        config_path=args.config,
        base_output_dir=output_dir
    )
    
    try:
        # Exécuter le pipeline
        results = pipeline.run_full_pipeline(
            input_dir=input_dir,
            skip_stages=skip_stages,
            max_files=args.max_files
        )
        
        # Afficher un résumé
        n_files = len([k for k in results.keys() if k != "metadata" and k != "quality_report"])
        logging.info(f"Pipeline terminé avec succès : {n_files} fichiers traités")
        
        if "quality_report" in results:
            quality = results["quality_report"]
            logging.info(f"Segments validés : {quality.get('valid_segments', 0)}/{quality.get('total_segments', 0)}")
        
    except Exception as e:
        logging.error(f"Erreur lors de l'exécution du pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
