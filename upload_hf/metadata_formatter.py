"""
Module de formatage des métadonnées pour Hugging Face.

Ce module permet de formater les métadonnées des fichiers audio
pour les rendre compatibles avec le format attendu par Hugging Face.

Classes principales :
- MetadataFormatter : Classe gérant le formatage des métadonnées
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hf_upload.log'),
        logging.StreamHandler()
    ]
)

class MetadataFormatter:
    """
    Classe pour formater les métadonnées des fichiers audio pour Hugging Face.
    
    Cette classe permet de convertir les métadonnées générées par le pipeline
    de traitement audio en un format compatible avec Hugging Face.
    """
    
    def __init__(self, 
                language_code: str = "fon",
                audio_base_path: Optional[Union[str, Path]] = None,
                output_dir: Union[str, Path] = "hf_metadata"):
        """
        Initialise le formateur de métadonnées avec les paramètres spécifiés.
        
        Entrées :
            language_code (str) : Code de langue ISO 639-3
            audio_base_path (Optional[Union[str, Path]]) : Chemin de base pour les fichiers audio
            output_dir (Union[str, Path]) : Répertoire de sortie pour les métadonnées formatées
        """
        self.language_code = language_code
        self.audio_base_path = Path(audio_base_path) if audio_base_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def format_metadata(self, 
                       metadata_file: Union[str, Path],
                       output_file: Optional[Union[str, Path]] = None,
                       required_columns: Optional[List[str]] = None,
                       rename_columns: Optional[Dict[str, str]] = None,
                       add_columns: Optional[Dict[str, Any]] = None) -> str:
        """
        Formate les métadonnées pour Hugging Face.
        
        Entrées :
            metadata_file (Union[str, Path]) : Chemin du fichier de métadonnées
            output_file (Optional[Union[str, Path]]) : Chemin du fichier de sortie
            required_columns (Optional[List[str]]) : Colonnes requises
            rename_columns (Optional[Dict[str, str]]) : Colonnes à renommer
            add_columns (Optional[Dict[str, Any]]) : Colonnes à ajouter
        
        Sorties :
            str : Chemin du fichier formaté
        """
        metadata_file = Path(metadata_file)
        
        # Définir le fichier de sortie si non spécifié
        if output_file is None:
            output_file = self.output_dir / f"hf_{metadata_file.name}"
        else:
            output_file = Path(output_file)
        
        try:
            # Charger les métadonnées
            if metadata_file.suffix.lower() == '.csv':
                df = pd.read_csv(metadata_file)
            elif metadata_file.suffix.lower() == '.parquet':
                df = pd.read_parquet(metadata_file)
            else:
                raise ValueError(f"Format de fichier non pris en charge: {metadata_file.suffix}")
            
            # Vérifier les colonnes requises
            if required_columns:
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Colonnes manquantes dans le fichier de métadonnées: {missing_columns}")
            
            # Renommer les colonnes si nécessaire
            if rename_columns:
                df = df.rename(columns=rename_columns)
            
            # Ajouter les colonnes manquantes
            if add_columns:
                for col, value in add_columns.items():
                    if col not in df.columns:
                        df[col] = value
            
            # Ajouter la colonne de langue si elle n'existe pas
            if 'language' not in df.columns:
                df['language'] = self.language_code
            
            # Formater les chemins des fichiers audio si un chemin de base est spécifié
            if self.audio_base_path and 'audio' in df.columns:
                df['audio'] = df['audio'].apply(lambda x: str(self.audio_base_path / x) if isinstance(x, str) else x)
            
            # Sauvegarder les métadonnées formatées
            if output_file.suffix.lower() == '.csv':
                df.to_csv(output_file, index=False)
            elif output_file.suffix.lower() == '.parquet':
                df.to_parquet(output_file, index=False)
            elif output_file.suffix.lower() == '.json':
                df.to_json(output_file, orient='records', lines=True)
            else:
                # Par défaut, sauvegarder en CSV
                output_file = output_file.with_suffix('.csv')
                df.to_csv(output_file, index=False)
            
            logging.info(f"Métadonnées formatées: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logging.error(f"Erreur lors du formatage des métadonnées: {e}")
            raise
    
    def merge_metadata_files(self, 
                           metadata_files: List[Union[str, Path]],
                           output_file: Union[str, Path],
                           validate_schema: bool = True) -> str:
        """
        Fusionne plusieurs fichiers de métadonnées en un seul.
        
        Entrées :
            metadata_files (List[Union[str, Path]]) : Liste des fichiers de métadonnées
            output_file (Union[str, Path]) : Chemin du fichier de sortie
            validate_schema (bool) : Si True, vérifie que les schémas sont compatibles
        
        Sorties :
            str : Chemin du fichier fusionné
        """
        output_file = Path(output_file)
        
        try:
            # Charger les métadonnées
            dfs = []
            for file in metadata_files:
                file = Path(file)
                if file.suffix.lower() == '.csv':
                    df = pd.read_csv(file)
                elif file.suffix.lower() == '.parquet':
                    df = pd.read_parquet(file)
                elif file.suffix.lower() == '.json':
                    df = pd.read_json(file, lines=True)
                else:
                    logging.warning(f"Format de fichier non pris en charge: {file.suffix}")
                    continue
                
                dfs.append(df)
            
            if not dfs:
                raise ValueError("Aucun fichier de métadonnées valide fourni")
            
            # Vérifier les schémas si demandé
            if validate_schema:
                # Vérifier que toutes les DataFrames ont les mêmes colonnes
                columns_sets = [set(df.columns) for df in dfs]
                common_columns = set.intersection(*columns_sets)
                
                if not common_columns:
                    raise ValueError("Les fichiers de métadonnées n'ont aucune colonne en commun")
                
                # Filtrer les DataFrames pour ne garder que les colonnes communes
                dfs = [df[list(common_columns)] for df in dfs]
            
            # Fusionner les DataFrames
            merged_df = pd.concat(dfs, ignore_index=True)
            
            # Sauvegarder les métadonnées fusionnées
            if output_file.suffix.lower() == '.csv':
                merged_df.to_csv(output_file, index=False)
            elif output_file.suffix.lower() == '.parquet':
                merged_df.to_parquet(output_file, index=False)
            elif output_file.suffix.lower() == '.json':
                merged_df.to_json(output_file, orient='records', lines=True)
            else:
                # Par défaut, sauvegarder en CSV
                output_file = output_file.with_suffix('.csv')
                merged_df.to_csv(output_file, index=False)
            
            logging.info(f"Métadonnées fusionnées: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logging.error(f"Erreur lors de la fusion des métadonnées: {e}")
            raise
    
    def create_hf_metadata(self, 
                         metadata_file: Union[str, Path],
                         audio_column: str = "segment_file",
                         split_column: Optional[str] = "split",
                         speaker_column: Optional[str] = "speaker_id",
                         duration_column: Optional[str] = "duration",
                         output_file: Optional[Union[str, Path]] = None) -> str:
        """
        Crée des métadonnées spécifiques pour Hugging Face.
        
        Entrées :
            metadata_file (Union[str, Path]) : Chemin du fichier de métadonnées
            audio_column (str) : Nom de la colonne contenant les chemins des fichiers audio
            split_column (Optional[str]) : Nom de la colonne contenant les splits
            speaker_column (Optional[str]) : Nom de la colonne contenant les IDs de locuteurs
            duration_column (Optional[str]) : Nom de la colonne contenant les durées
            output_file (Optional[Union[str, Path]]) : Chemin du fichier de sortie
        
        Sorties :
            str : Chemin du fichier formaté
        """
        metadata_file = Path(metadata_file)
        
        # Définir le fichier de sortie si non spécifié
        if output_file is None:
            output_file = self.output_dir / f"hf_{metadata_file.stem}.csv"
        else:
            output_file = Path(output_file)
        
        try:
            # Charger les métadonnées
            if metadata_file.suffix.lower() == '.csv':
                df = pd.read_csv(metadata_file)
            elif metadata_file.suffix.lower() == '.parquet':
                df = pd.read_parquet(metadata_file)
            else:
                raise ValueError(f"Format de fichier non pris en charge: {metadata_file.suffix}")
            
            # Vérifier que la colonne audio existe
            if audio_column not in df.columns:
                raise ValueError(f"La colonne {audio_column} n'existe pas dans le fichier de métadonnées")
            
            # Créer un DataFrame pour Hugging Face
            hf_df = pd.DataFrame()
            
            # Ajouter la colonne audio
            hf_df['file'] = df[audio_column]
            
            # Ajouter la colonne audio avec le chemin complet si un chemin de base est spécifié
            if self.audio_base_path:
                hf_df['audio'] = hf_df['file'].apply(lambda x: str(self.audio_base_path / x) if isinstance(x, str) else x)
            else:
                hf_df['audio'] = hf_df['file']
            
            # Ajouter la colonne de langue
            hf_df['language'] = self.language_code
            
            # Ajouter la colonne de split si elle existe
            if split_column and split_column in df.columns:
                hf_df['split'] = df[split_column]
            
            # Ajouter la colonne de locuteur si elle existe
            if speaker_column and speaker_column in df.columns:
                hf_df['speaker_id'] = df[speaker_column]
            
            # Ajouter la colonne de durée si elle existe
            if duration_column and duration_column in df.columns:
                hf_df['duration'] = df[duration_column]
            
            # Ajouter d'autres colonnes utiles
            for col in ['snr', 'rms_db', 'loudness_lufs', 'sampling_rate']:
                if col in df.columns:
                    hf_df[col] = df[col]
            
            # Sauvegarder les métadonnées formatées
            hf_df.to_csv(output_file, index=False)
            
            logging.info(f"Métadonnées HF créées: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logging.error(f"Erreur lors de la création des métadonnées HF: {e}")
            raise
