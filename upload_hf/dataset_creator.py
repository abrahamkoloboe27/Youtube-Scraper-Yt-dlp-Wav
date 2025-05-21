"""
Module de création de dataset pour Hugging Face.

Ce module permet de créer et configurer un dataset sur Hugging Face,
en définissant sa structure, ses métadonnées et ses configurations.

Classes principales :
- DatasetCreator : Classe gérant la création et la configuration de datasets
"""

import os
import logging
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Audio, ClassLabel
from .hf_client import HFClient

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hf_upload.log'),
        logging.StreamHandler()
    ]
)

class DatasetCreator:
    """
    Classe pour créer et configurer des datasets sur Hugging Face.
    
    Cette classe permet de définir la structure d'un dataset, ses métadonnées,
    et de le préparer pour l'upload vers Hugging Face.
    """
    
    def __init__(self, 
                hf_client: Optional[HFClient] = None,
                dataset_name: str = "fongbe_audio",
                dataset_language: str = "fon",
                dataset_license: str = "cc-by-4.0",
                local_dir: Union[str, Path] = "hf_dataset"):
        """
        Initialise le créateur de dataset avec les paramètres spécifiés.
        
        Entrées :
            hf_client (Optional[HFClient]) : Client Hugging Face
            dataset_name (str) : Nom du dataset
            dataset_language (str) : Code de langue du dataset (ISO 639-3)
            dataset_license (str) : Licence du dataset
            local_dir (Union[str, Path]) : Répertoire local pour le dataset
        """
        self.hf_client = hf_client or HFClient()
        self.dataset_name = dataset_name
        self.dataset_language = dataset_language
        self.dataset_license = dataset_license
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        # Sous-répertoires pour les données
        self.audio_dir = self.local_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        
        self.metadata_dir = self.local_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Dataset et features
        self.dataset = None
        self.features = None
    
    def define_features(self, 
                       include_transcription: bool = False,
                       speaker_labels: Optional[List[str]] = None,
                       additional_features: Optional[Dict[str, Any]] = None) -> Features:
        """
        Définit les features du dataset.
        
        Entrées :
            include_transcription (bool) : Si True, inclut un champ pour la transcription
            speaker_labels (Optional[List[str]]) : Liste des labels de locuteurs
            additional_features (Optional[Dict[str, Any]]) : Features supplémentaires
        
        Sorties :
            Features : Objet Features pour le dataset
        """
        # Features de base
        features = {
            "file": Value("string"),
            "audio": Audio(),
            "sampling_rate": Value("int32"),
            "duration": Value("float"),
            "language": Value("string"),
            "speaker_id": Value("string"),
        }
        
        # Ajouter la transcription si demandée
        if include_transcription:
            features["transcription"] = Value("string")
        
        # Ajouter les labels de locuteurs si fournis
        if speaker_labels:
            features["speaker_label"] = ClassLabel(names=speaker_labels)
        
        # Ajouter les features supplémentaires
        if additional_features:
            features.update(additional_features)
        
        self.features = Features(features)
        return self.features
    
    def create_dataset_from_metadata(self, 
                                   metadata_file: Union[str, Path],
                                   audio_column: str = "segment_file",
                                   split_column: Optional[str] = "split") -> DatasetDict:
        """
        Crée un dataset à partir d'un fichier de métadonnées.
        
        Entrées :
            metadata_file (Union[str, Path]) : Chemin du fichier de métadonnées (CSV ou Parquet)
            audio_column (str) : Nom de la colonne contenant les chemins des fichiers audio
            split_column (Optional[str]) : Nom de la colonne contenant les splits (train/dev/test)
        
        Sorties :
            DatasetDict : Dataset créé
        """
        metadata_file = Path(metadata_file)
        
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
            
            # Ajouter la colonne language si elle n'existe pas
            if "language" not in df.columns:
                df["language"] = self.dataset_language
            
            # Créer le dataset
            if split_column and split_column in df.columns:
                # Créer un dataset par split
                datasets = {}
                for split, split_df in df.groupby(split_column):
                    datasets[split] = Dataset.from_pandas(split_df)
                
                self.dataset = DatasetDict(datasets)
            else:
                # Créer un seul dataset
                self.dataset = Dataset.from_pandas(df)
            
            logging.info(f"Dataset créé à partir de {metadata_file}")
            return self.dataset
            
        except Exception as e:
            logging.error(f"Erreur lors de la création du dataset à partir de {metadata_file}: {e}")
            raise
    
    def create_dataset_card(self, 
                          output_file: Union[str, Path] = "README.md",
                          dataset_description: str = "",
                          dataset_citation: str = "",
                          dataset_homepage: str = "",
                          dataset_repo: str = "") -> str:
        """
        Crée une carte de dataset (README.md) pour Hugging Face.
        
        Entrées :
            output_file (Union[str, Path]) : Chemin du fichier de sortie
            dataset_description (str) : Description du dataset
            dataset_citation (str) : Citation du dataset
            dataset_homepage (str) : Page d'accueil du dataset
            dataset_repo (str) : Dépôt du dataset
        
        Sorties :
            str : Chemin du fichier créé
        """
        output_path = self.local_dir / output_file
        
        # Description par défaut si non fournie
        if not dataset_description:
            dataset_description = f"""# Dataset {self.dataset_name}

Ce dataset contient des fichiers audio en langue Fongbè ({self.dataset_language}), prétraités et segmentés pour l'entraînement de modèles de reconnaissance vocale.

## Contenu

Le dataset contient des fichiers audio au format WAV (16 kHz, 16 bits, mono), avec les métadonnées associées.
"""
        
        # Ajouter des informations sur le dataset
        if self.dataset:
            dataset_info = "\n## Statistiques\n\n"
            
            if isinstance(self.dataset, DatasetDict):
                for split, ds in self.dataset.items():
                    dataset_info += f"- **{split}** : {len(ds)} exemples\n"
            else:
                dataset_info += f"- **Total** : {len(self.dataset)} exemples\n"
            
            dataset_description += dataset_info
        
        # Ajouter la citation si fournie
        if dataset_citation:
            dataset_description += f"\n## Citation\n\n```bibtex\n{dataset_citation}\n```\n"
        
        # Ajouter les liens si fournis
        if dataset_homepage or dataset_repo:
            dataset_description += "\n## Liens\n\n"
            if dataset_homepage:
                dataset_description += f"- [Page d'accueil]({dataset_homepage})\n"
            if dataset_repo:
                dataset_description += f"- [Dépôt]({dataset_repo})\n"
        
        # Ajouter la licence
        dataset_description += f"\n## Licence\n\n{self.dataset_license}\n"
        
        # Écrire le fichier
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(dataset_description)
        
        logging.info(f"Carte de dataset créée: {output_path}")
        return str(output_path)
    
    def create_dataset_config(self, 
                            output_file: Union[str, Path] = "dataset_infos.json") -> str:
        """
        Crée un fichier de configuration pour le dataset.
        
        Entrées :
            output_file (Union[str, Path]) : Chemin du fichier de sortie
        
        Sorties :
            str : Chemin du fichier créé
        """
        output_path = self.local_dir / output_file
        
        # Informations de base
        config = {
            "name": self.dataset_name,
            "version": "1.0.0",
            "description": f"Dataset audio en langue Fongbè ({self.dataset_language})",
            "license": self.dataset_license,
            "language": [self.dataset_language],
            "features": self.features.to_dict() if self.features else {},
        }
        
        # Ajouter des informations sur les splits si disponibles
        if isinstance(self.dataset, DatasetDict):
            config["splits"] = {}
            for split, ds in self.dataset.items():
                config["splits"][split] = {"num_examples": len(ds)}
        elif self.dataset:
            config["num_examples"] = len(self.dataset)
        
        # Écrire le fichier
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"Configuration de dataset créée: {output_path}")
        return str(output_path)
    
    def prepare_local_dataset(self, 
                            metadata_file: Union[str, Path],
                            audio_dir: Union[str, Path],
                            copy_audio: bool = True) -> Path:
        """
        Prépare un dataset local pour l'upload vers Hugging Face.
        
        Entrées :
            metadata_file (Union[str, Path]) : Chemin du fichier de métadonnées
            audio_dir (Union[str, Path]) : Répertoire contenant les fichiers audio
            copy_audio (bool) : Si True, copie les fichiers audio dans le répertoire local
        
        Sorties :
            Path : Chemin du répertoire local préparé
        """
        try:
            # Copier le fichier de métadonnées
            metadata_file = Path(metadata_file)
            dest_metadata = self.metadata_dir / metadata_file.name
            
            import shutil
            shutil.copy2(metadata_file, dest_metadata)
            
            # Créer le dataset à partir des métadonnées
            self.create_dataset_from_metadata(dest_metadata)
            
            # Définir les features si ce n'est pas déjà fait
            if not self.features:
                self.define_features()
            
            # Créer la carte et la configuration du dataset
            self.create_dataset_card()
            self.create_dataset_config()
            
            # Copier les fichiers audio si demandé
            if copy_audio:
                audio_dir = Path(audio_dir)
                
                # Récupérer la liste des fichiers audio à partir des métadonnées
                if metadata_file.suffix.lower() == '.csv':
                    df = pd.read_csv(dest_metadata)
                else:
                    df = pd.read_parquet(dest_metadata)
                
                audio_files = df["segment_file"].unique() if "segment_file" in df.columns else []
                
                # Copier chaque fichier audio
                for audio_file in audio_files:
                    src_path = audio_dir / audio_file
                    if src_path.exists():
                        dest_path = self.audio_dir / audio_file
                        shutil.copy2(src_path, dest_path)
                    else:
                        logging.warning(f"Fichier audio non trouvé: {src_path}")
            
            logging.info(f"Dataset local préparé: {self.local_dir}")
            return self.local_dir
            
        except Exception as e:
            logging.error(f"Erreur lors de la préparation du dataset local: {e}")
            raise
