"""
Module de gestion des métadonnées pour les fichiers audio.

Ce module permet de créer, gérer et exporter les métadonnées des fichiers audio,
notamment pour la création des ensembles d'entraînement/validation/test.

Classes principales :
- MetadataManager : Classe gérant les métadonnées des fichiers audio
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Set
import pandas as pd
import numpy as np
import json
import librosa
import random
from datetime import datetime
import re
from .mongo_logger import MongoLogger

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ││ %(levelname)s ││ %(name)s ││ %(message)s ",
    datefmt="%Y-%m-%d %H:%M:%S",

    handlers=[
        logging.FileHandler('logs/audio_processing.log'),
        logging.StreamHandler()
    ]
)


import pyannote
pyannote_logger = logging.getLogger("pyannote.audio")
pyannote_logger.setLevel(logging.INFO)  


class MetadataManager:
    """
    Classe pour la gestion des métadonnées des fichiers audio.
    
    Cette classe permet de créer, gérer et exporter les métadonnées des fichiers audio,
    notamment pour la création des ensembles d'entraînement/validation/test.
    """
    
    def __init__(self, 
                output_dir: Union[str, Path] = "processed/metadata",
                test_ratio: float = 0.10,
                dev_ratio: float = 0.10,
                random_seed: int = 42):
        """
        Initialise le gestionnaire de métadonnées avec les paramètres spécifiés.
        
        Entrées :
            output_dir (Union[str, Path]) : Répertoire de sortie pour les métadonnées
            test_ratio (float) : Ratio de données pour l'ensemble de test
            dev_ratio (float) : Ratio de données pour l'ensemble de développement/validation
            random_seed (int) : Graine aléatoire pour la reproductibilité
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_ratio = test_ratio
        self.dev_ratio = dev_ratio
        self.random_seed = random_seed
        self.logger = MongoLogger()
        
        # Initialisation des DataFrames
        self.metadata_df = None
        self.train_df = None
        self.dev_df = None
        self.test_df = None
    
    def extract_metadata_from_mongo(self, collection_name: str = "audio_processing") -> pd.DataFrame:
        """
        Extrait les métadonnées des fichiers audio depuis MongoDB.
        
        Entrées :
            collection_name (str) : Nom de la collection MongoDB à utiliser
        
        Sorties :
            pd.DataFrame : DataFrame contenant les métadonnées
        """
        # Accéder à la collection MongoDB
        db = self.logger.db
        collection = db[collection_name]
        
        # Récupérer tous les documents
        documents = list(collection.find({}))
        
        # Créer une liste pour stocker les métadonnées extraites
        metadata_entries = []
        
        for doc in documents:
            # Traiter chaque segment du document
            for segment in doc.get("segments", []):
                entry = {
                    "original_file": doc.get("file", ""),
                    "segment_file": segment.get("file", ""),
                    "speaker_id": segment.get("speaker_id", "unknown"),
                    "start_time": segment.get("start_time", 0),
                    "end_time": segment.get("end_time", 0),
                    "duration": segment.get("duration", 0),
                    "processing_stages": doc.get("processing_stages", {}),
                }
                
                # Ajouter les métadonnées spécifiques au segment
                segment_metadata = segment.get("metadata", {})
                for key, value in segment_metadata.items():
                    entry[f"segment_{key}"] = value
                
                # Ajouter les détails de traitement
                processing_details = doc.get("processing_details", {})
                
                # Ajouter les métriques de qualité audio si disponibles
                if "cleaned" in processing_details:
                    cleaned_details = processing_details.get("cleaned", {})
                    after_metrics = cleaned_details.get("after", {})
                    
                    if after_metrics:
                        entry["snr"] = after_metrics.get("snr", 0)
                        entry["rms_db"] = after_metrics.get("rms_db", 0)
                        entry["loudness_lufs"] = after_metrics.get("loudness_lufs", 0)
                
                # Ajouter l'entrée à la liste
                metadata_entries.append(entry)
        
        # Créer un DataFrame à partir des entrées
        df = pd.DataFrame(metadata_entries)
        
        # Tri par locuteur et durée
        if not df.empty and "speaker_id" in df.columns and "duration" in df.columns:
            df = df.sort_values(by=["speaker_id", "duration"], ascending=[True, False])
        
        return df
    
    def split_by_speaker(self, 
                        metadata_df: pd.DataFrame, 
                        test_ratio: float = None, 
                        dev_ratio: float = None) -> Dict[str, pd.DataFrame]:
        """
        Divise les métadonnées en ensembles train/dev/test en évitant la fuite de locuteurs.
        
        Entrées :
            metadata_df (pd.DataFrame) : DataFrame contenant les métadonnées
            test_ratio (float) : Ratio de données pour l'ensemble de test (si None, utilise self.test_ratio)
            dev_ratio (float) : Ratio de données pour l'ensemble de dev (si None, utilise self.dev_ratio)
        
        Sorties :
            Dict[str, pd.DataFrame] : Dictionnaire contenant les DataFrames train/dev/test
        """
        if test_ratio is None:
            test_ratio = self.test_ratio
            
        if dev_ratio is None:
            dev_ratio = self.dev_ratio
        
        # S'assurer que les ratios sont valides
        if test_ratio + dev_ratio >= 1.0:
            raise ValueError("La somme des ratios de test et dev doit être inférieure à 1.0")
        
        # Récupérer la liste des locuteurs uniques
        speakers = metadata_df["speaker_id"].unique().tolist()
        
        # Mélanger la liste des locuteurs
        random.seed(self.random_seed)
        random.shuffle(speakers)
        
        # Calculer le nombre de locuteurs pour chaque ensemble
        n_speakers = len(speakers)
        n_test_speakers = max(1, int(n_speakers * test_ratio))
        n_dev_speakers = max(1, int(n_speakers * dev_ratio))
        n_train_speakers = n_speakers - n_test_speakers - n_dev_speakers
        
        # Répartir les locuteurs
        test_speakers = speakers[:n_test_speakers]
        dev_speakers = speakers[n_test_speakers:n_test_speakers + n_dev_speakers]
        train_speakers = speakers[n_test_speakers + n_dev_speakers:]
        
        # Créer les DataFrames correspondants
        test_df = metadata_df[metadata_df["speaker_id"].isin(test_speakers)].copy()
        dev_df = metadata_df[metadata_df["speaker_id"].isin(dev_speakers)].copy()
        train_df = metadata_df[metadata_df["speaker_id"].isin(train_speakers)].copy()
        
        # Ajouter une colonne indiquant l'ensemble
        test_df["split"] = "test"
        dev_df["split"] = "dev"
        train_df["split"] = "train"
        
        # Vérifier que chaque ensemble contient des données
        if test_df.empty:
            logging.warning("L'ensemble de test est vide, allocation d'un locuteur depuis l'ensemble d'entraînement")
            if not train_df.empty:
                # Prendre un locuteur de l'ensemble d'entraînement
                speaker_to_move = train_df["speaker_id"].unique()[0]
                moved_rows = train_df[train_df["speaker_id"] == speaker_to_move].copy()
                moved_rows["split"] = "test"
                test_df = moved_rows
                train_df = train_df[train_df["speaker_id"] != speaker_to_move]
        
        if dev_df.empty:
            logging.warning("L'ensemble de développement est vide, allocation d'un locuteur depuis l'ensemble d'entraînement")
            if not train_df.empty:
                # Prendre un locuteur de l'ensemble d'entraînement
                speaker_to_move = train_df["speaker_id"].unique()[0]
                moved_rows = train_df[train_df["speaker_id"] == speaker_to_move].copy()
                moved_rows["split"] = "dev"
                dev_df = moved_rows
                train_df = train_df[train_df["speaker_id"] != speaker_to_move]
        
        # Journaliser les statistiques
        logging.info(f"Split par locuteur : Train={len(train_df)} exemples ({train_df['duration'].sum():.2f}s), "
                    f"Dev={len(dev_df)} exemples ({dev_df['duration'].sum():.2f}s), "
                    f"Test={len(test_df)} exemples ({test_df['duration'].sum():.2f}s)")
        
        return {
            "train": train_df,
            "dev": dev_df,
            "test": test_df
        }
    
    def enrich_metadata(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrichit les métadonnées avec des informations supplémentaires.
        
        Entrées :
            metadata_df (pd.DataFrame) : DataFrame contenant les métadonnées de base
        
        Sorties :
            pd.DataFrame : DataFrame enrichi avec des informations supplémentaires
        """
        # Copier le DataFrame pour éviter de modifier l'original
        df = metadata_df.copy()
        
        # Ajouter la date de génération des métadonnées
        df["metadata_date"] = datetime.now().strftime("%Y-%m-%d")
        
        # Extraire le type de contenu à partir du nom de fichier (si disponible)
        def extract_content_type(filename):
            # Exemple simple : si le nom contient 'music', c'est de la musique, etc.
            if isinstance(filename, str):
                if "music" in filename.lower():
                    return "music"
                elif "speech" in filename.lower():
                    return "speech"
                elif "interview" in filename.lower():
                    return "interview"
            return "unknown"
        
        df["content_type"] = df["original_file"].apply(extract_content_type)
        
        # Ajouter des chemins absolus pour plus de facilité
        def get_abs_path(file):
            if isinstance(file, str) and not os.path.isabs(file):
                return os.path.abspath(file)
            return file
        
        if "segment_file" in df.columns:
            df["segment_path"] = df["segment_file"].apply(get_abs_path)
        
        return df
    
    def export_metadata(self, 
                       metadata_df: pd.DataFrame, 
                       filename: str = "metadata.csv",
                       format: str = "csv") -> str:
        """
        Exporte les métadonnées vers un fichier CSV ou Parquet.
        
        Entrées :
            metadata_df (pd.DataFrame) : DataFrame contenant les métadonnées
            filename (str) : Nom du fichier de sortie
            format (str) : Format de sortie ('csv' ou 'parquet')
        
        Sorties :
            str : Chemin du fichier exporté
        """
        # Définir le chemin de sortie
        if format.lower() not in ["csv", "parquet"]:
            raise ValueError("Le format doit être 'csv' ou 'parquet'")
            
        # Garantir la bonne extension
        base_name = Path(filename).stem
        output_path = self.output_dir / f"{base_name}.{format.lower()}"
        
        # Exporter selon le format spécifié
        if format.lower() == "csv":
            metadata_df.to_csv(output_path, index=False, encoding='utf-8')
        else:  # parquet
            metadata_df.to_parquet(output_path, index=False)
        
        logging.info(f"Métadonnées exportées vers {output_path}")
        return str(output_path)
    
    def process(self, 
               segment_dir: Union[str, Path] = "processed/cleaned",
               export_format: str = "csv") -> Dict[str, str]:
        """
        Traite les métadonnées : extraction, enrichissement, division et export.
        
        Entrées :
            segment_dir (Union[str, Path]) : Répertoire contenant les segments audio
            export_format (str) : Format d'export ('csv' ou 'parquet')
        
        Sorties :
            Dict[str, str] : Chemins des fichiers de métadonnées exportés
        """
        logging.info("Traitement des métadonnées...")
        
        # Extraire les métadonnées depuis MongoDB
        self.metadata_df = self.extract_metadata_from_mongo()
        
        if self.metadata_df.empty:
            logging.error("Aucune métadonnée extraite de MongoDB")
            return {}
        
        # Enrichir les métadonnées
        self.metadata_df = self.enrich_metadata(self.metadata_df)
        
        # Diviser les métadonnées en ensembles train/dev/test
        split_dfs = self.split_by_speaker(self.metadata_df)
        self.train_df = split_dfs["train"]
        self.dev_df = split_dfs["dev"]
        self.test_df = split_dfs["test"]
        
        # Combiner tous les ensembles avec la colonne de split
        combined_df = pd.concat([self.train_df, self.dev_df, self.test_df])
        
        # Exporter les métadonnées
        output_paths = {}
        output_paths["all"] = self.export_metadata(
            combined_df, f"metadata_all.{export_format}", export_format
        )
        output_paths["train"] = self.export_metadata(
            self.train_df, f"metadata_train.{export_format}", export_format
        )
        output_paths["dev"] = self.export_metadata(
            self.dev_df, f"metadata_dev.{export_format}", export_format
        )
        output_paths["test"] = self.export_metadata(
            self.test_df, f"metadata_test.{export_format}", export_format
        )
        
        # Mettre à jour MongoDB pour tous les fichiers traités
        processed_files = set(combined_df["original_file"].unique())
        for file_name in processed_files:
            mongo_doc = self.logger.get_processing_status(file_name)
            if mongo_doc:
                doc_id = str(mongo_doc["_id"])
                self.logger.update_stage(
                    doc_id, 
                    "metadata_tagged", 
                    True, 
                    {
                        "export_paths": output_paths,
                        "n_segments": len(combined_df[combined_df["original_file"] == file_name]),
                        "splits": {
                            "train": len(self.train_df[self.train_df["original_file"] == file_name]),
                            "dev": len(self.dev_df[self.dev_df["original_file"] == file_name]),
                            "test": len(self.test_df[self.test_df["original_file"] == file_name])
                        }
                    }
                )
        
        logging.info(f"Traitement des métadonnées terminé : {len(combined_df)} segments traités")
        return output_paths
    
    def get_splits_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Récupère un résumé des statistiques des ensembles train/dev/test.
        
        Sorties :
            Dict[str, Dict[str, Any]] : Résumé des statistiques
        """
        if self.train_df is None or self.dev_df is None or self.test_df is None:
            logging.error("Les ensembles de données ne sont pas encore définis")
            return {}
        
        summary = {}
        
        for name, df in [("train", self.train_df), ("dev", self.dev_df), ("test", self.test_df)]:
            summary[name] = {
                "n_segments": len(df),
                "total_duration": df["duration"].sum(),
                "n_speakers": df["speaker_id"].nunique(),
                "avg_segment_duration": df["duration"].mean(),
                "min_segment_duration": df["duration"].min(),
                "max_segment_duration": df["duration"].max(),
                "speakers": df["speaker_id"].unique().tolist()
            }
            
            # Ajouter des statistiques de qualité audio si disponibles
            if "snr" in df.columns:
                summary[name]["avg_snr"] = df["snr"].mean()
                summary[name]["min_snr"] = df["snr"].min()
                summary[name]["max_snr"] = df["snr"].max()
        
        return summary
