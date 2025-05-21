"""
Module principal du pipeline de traitement audio.

Ce module orchestre l'ensemble du processus de traitement des fichiers audio Fongbè,
en exécutant les différentes étapes dans l'ordre approprié et en gérant les dépendances.

Classes principales :
- AudioPipeline : Classe orchestrant l'ensemble du pipeline de traitement
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Set
import argparse
import pandas as pd
import json

from .audio_loader import AudioLoader
from .loudness_normalizer import LoudnessNormalizer
from .silence_remover import SilenceRemover
from .diarization import Diarization
from .segmentation import Segmentation
from .audio_cleaner import AudioCleaner
from .metadata_manager import MetadataManager
from .data_augmentation import DataAugmentation
from .quality_checker import QualityChecker
from .mongo_logger import MongoLogger

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)

class AudioPipeline:
    """
    Classe pour orchestrer l'ensemble du pipeline de traitement audio.
    
    Cette classe permet d'exécuter les différentes étapes du pipeline de traitement
    dans l'ordre approprié, en gérant les dépendances et en assurant la traçabilité.
    """
    
    def __init__(self, 
                config_path: Optional[Union[str, Path]] = None,
                base_output_dir: Union[str, Path] = "processed"):
        """
        Initialise le pipeline avec la configuration spécifiée.
        
        Entrées :
            config_path (Optional[Union[str, Path]]) : Chemin du fichier de configuration
            base_output_dir (Union[str, Path]) : Répertoire de base pour les sorties
        """
        # Charger la configuration si spécifiée, sinon utiliser les valeurs par défaut
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialiser le répertoire de base pour les sorties
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialiser le logger MongoDB
        self.logger = MongoLogger()
        
        # Initialiser les composants du pipeline avec la configuration
        self._init_components()
        
        # État d'exécution
        self.current_stage = None
        self.processed_files = {}
        
        # Suivi du temps d'exécution
        self.stage_timings = {}
    
    def _load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Charge la configuration à partir d'un fichier JSON.
        
        Entrées :
            config_path (Union[str, Path]) : Chemin du fichier de configuration
        
        Sorties :
            Dict[str, Any] : Configuration chargée
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logging.info(f"Configuration chargée depuis {config_path}")
            return config
        except Exception as e:
            logging.error(f"Erreur lors du chargement de la configuration: {e}")
            return {}
    
    def _init_components(self):
        """
        Initialise les composants du pipeline avec la configuration.
        """
        # Récupérer les configurations spécifiques
        loader_config = self.config.get('audio_loader', {})
        normalizer_config = self.config.get('loudness_normalizer', {})
        silence_config = self.config.get('silence_remover', {})
        diarization_config = self.config.get('diarization', {})
        segmentation_config = self.config.get('segmentation', {})
        cleaner_config = self.config.get('audio_cleaner', {})
        metadata_config = self.config.get('metadata_manager', {})
        augmentation_config = self.config.get('data_augmentation', {})
        quality_config = self.config.get('quality_checker', {})
        
        # Initialiser les composants
        self.audio_loader = AudioLoader(**loader_config)
        self.loudness_normalizer = LoudnessNormalizer(**normalizer_config)
        self.silence_remover = SilenceRemover(**silence_config)
        self.diarization = Diarization(**diarization_config)
        self.segmentation = Segmentation(**segmentation_config)
        self.audio_cleaner = AudioCleaner(**cleaner_config)
        self.metadata_manager = MetadataManager(**metadata_config)
        self.data_augmentation = DataAugmentation(**augmentation_config)
        self.quality_checker = QualityChecker(**quality_config)
    
    def _time_stage(self, stage_name: str, func, *args, **kwargs):
        """
        Exécute une étape du pipeline en mesurant le temps d'exécution.
        
        Entrées :
            stage_name (str) : Nom de l'étape
            func : Fonction à exécuter
            *args, **kwargs : Arguments à passer à la fonction
        
        Sorties :
            Any : Résultat de la fonction
        """
        self.current_stage = stage_name
        logging.info(f"Début de l'étape: {stage_name}")
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        self.stage_timings[stage_name] = duration
        
        logging.info(f"Fin de l'étape: {stage_name} (durée: {duration:.2f}s)")
        return result
    
    def process_file(self, 
                    file_path: Union[str, Path],
                    skip_stages: Optional[List[str]] = None,
                    only_stages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Traite un fichier audio à travers l'ensemble du pipeline.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier audio à traiter
            skip_stages (Optional[List[str]]) : Étapes à sauter
            only_stages (Optional[List[str]]) : Seulement ces étapes (si spécifié)
        
        Sorties :
            Dict[str, Any] : Résultats du traitement
        """
        file_path = Path(file_path)
        results = {"file": str(file_path)}
        
        # Déterminer les étapes à exécuter
        all_stages = [
            "loading", "normalization", "silence_removal", 
            "diarization", "segmentation", "cleaning", 
            "metadata", "augmentation", "quality_check"
        ]
        
        if only_stages:
            stages_to_run = [s for s in all_stages if s in only_stages]
        elif skip_stages:
            stages_to_run = [s for s in all_stages if s not in skip_stages]
        else:
            stages_to_run = all_stages
        
        # 1. Chargement et uniformisation
        if "loading" in stages_to_run:
            uniformized_path = self._time_stage(
                "loading",
                self.audio_loader.process_file,
                file_path
            )
            results["uniformized_path"] = uniformized_path
            
            if not uniformized_path:
                logging.error(f"Échec du chargement du fichier: {file_path}")
                return results
            
            file_path = uniformized_path
        
        # 2. Normalisation de loudness
        if "normalization" in stages_to_run:
            normalized_path = self._time_stage(
                "normalization",
                self.loudness_normalizer.process_file,
                file_path
            )
            results["normalized_path"] = normalized_path
            
            if normalized_path:
                file_path = normalized_path
        
        # 3. Suppression des silences
        if "silence_removal" in stages_to_run:
            no_silence_path = self._time_stage(
                "silence_removal",
                self.silence_remover.process_file,
                file_path
            )
            results["no_silence_path"] = no_silence_path
            
            if no_silence_path:
                file_path = no_silence_path
        
        # 4. Diarisation (isolation des locuteurs)
        if "diarization" in stages_to_run:
            diarization_results = self._time_stage(
                "diarization",
                self.diarization.process_file,
                file_path
            )
            results["diarization_results"] = diarization_results
            
            # La diarisation crée plusieurs fichiers, un par locuteur
            speaker_files = []
            if diarization_results and "speakers" in diarization_results:
                for speaker, segments in diarization_results["speakers"].items():
                    for segment in segments:
                        if "output_file" in segment:
                            speaker_files.append(segment["output_file"])
            
            # Si des fichiers de locuteurs ont été créés, on les utilise pour la suite
            # Sinon, on continue avec le fichier original
            if speaker_files:
                diarized_files = speaker_files
            else:
                diarized_files = [file_path]
            
            results["diarized_files"] = diarized_files
        else:
            diarized_files = [file_path]
        
        # 5. Segmentation fine
        if "segmentation" in stages_to_run:
            all_segments = []
            
            for diarized_file in diarized_files:
                # Extraire l'ID du locuteur si disponible
                speaker_id = None
                if "_speaker_" in Path(diarized_file).name:
                    import re
                    match = re.search(r"_speaker_([^\.]+)", Path(diarized_file).name)
                    if match:
                        speaker_id = match.group(1)
                
                segments = self._time_stage(
                    "segmentation",
                    self.segmentation.process_file,
                    diarized_file,
                    speaker_id
                )
                all_segments.extend(segments)
            
            results["segments"] = all_segments
            
            # Utiliser ces segments pour la suite
            if all_segments:
                segment_files = all_segments
            else:
                segment_files = diarized_files
        else:
            segment_files = diarized_files
        
        # 6. Nettoyage sonore
        if "cleaning" in stages_to_run:
            cleaned_files = []
            
            for segment_file in segment_files:
                cleaned_path = self._time_stage(
                    "cleaning",
                    self.audio_cleaner.process_file,
                    segment_file
                )
                if cleaned_path:
                    cleaned_files.append(cleaned_path)
            
            results["cleaned_files"] = cleaned_files
            
            # Utiliser ces fichiers nettoyés pour la suite
            if cleaned_files:
                final_segments = cleaned_files
            else:
                final_segments = segment_files
        else:
            final_segments = segment_files
        
        # 7. Étiquetage et gestion des métadonnées
        # Note: Cette étape est généralement exécutée sur l'ensemble du corpus, pas sur un seul fichier
        if "metadata" in stages_to_run and self.processed_files.get("segments", []):
            # Ajouter les segments traités à la liste globale
            self.processed_files["segments"].extend(final_segments)
        else:
            if "segments" not in self.processed_files:
                self.processed_files["segments"] = []
            self.processed_files["segments"].extend(final_segments)
        
        # 8. Augmentation de données
        if "augmentation" in stages_to_run:
            augmented_files = {}
            
            for segment_file in final_segments:
                aug_result = self._time_stage(
                    "augmentation",
                    self.data_augmentation.augment_audio,
                    segment_file
                )
                if aug_result:
                    augmented_files[segment_file] = aug_result
            
            results["augmented_files"] = augmented_files
            
            # Ajouter ces fichiers augmentés à la liste des segments traités
            augmented_segments = []
            for segments in augmented_files.values():
                augmented_segments.extend([path for path, _ in segments])
            
            if "augmented_segments" not in self.processed_files:
                self.processed_files["augmented_segments"] = []
            self.processed_files["augmented_segments"].extend(augmented_segments)
        
        # 9. Vérification finale et exports
        # Note: Cette étape est généralement exécutée sur l'ensemble du corpus après l'augmentation
        if "quality_check" in stages_to_run and final_segments:
            validated_files = []
            
            for segment_file in final_segments:
                output_path = self._time_stage(
                    "quality_check",
                    self.quality_checker.process_file,
                    segment_file
                )
                if output_path:
                    validated_files.append(output_path)
            
            results["validated_files"] = validated_files
        
        results["stage_timings"] = self.stage_timings.copy()
        return results
    
    def process_directory(self, 
                         dir_path: Union[str, Path],
                         recursive: bool = True,
                         extensions: List[str] = ['.wav', '.mp3', '.flac', '.ogg'],
                         skip_stages: Optional[List[str]] = None,
                         only_stages: Optional[List[str]] = None,
                         max_files: Optional[int] = None) -> Dict[str, Any]:
        """
        Traite tous les fichiers audio d'un répertoire.
        
        Entrées :
            dir_path (Union[str, Path]) : Chemin du répertoire à traiter
            recursive (bool) : Si True, recherche récursivement dans les sous-répertoires
            extensions (List[str]) : Extensions de fichiers à rechercher
            skip_stages (Optional[List[str]]) : Étapes à sauter
            only_stages (Optional[List[str]]) : Seulement ces étapes (si spécifié)
            max_files (Optional[int]) : Nombre maximum de fichiers à traiter
        
        Sorties :
            Dict[str, Any] : Résultats du traitement par fichier
        """
        dir_path = Path(dir_path)
        logging.info(f"Traitement du répertoire: {dir_path}")
        
        # Trouver tous les fichiers audio dans le répertoire
        audio_files = []
        for ext in extensions:
            if recursive:
                audio_files.extend(list(dir_path.glob(f"**/*{ext}")))
            else:
                audio_files.extend(list(dir_path.glob(f"*{ext}")))
        
        # Limiter le nombre de fichiers si spécifié
        if max_files and len(audio_files) > max_files:
            audio_files = audio_files[:max_files]
        
        logging.info(f"Nombre de fichiers à traiter: {len(audio_files)}")
        
        # Traiter chaque fichier
        all_results = {}
        for file_path in audio_files:
            logging.info(f"Traitement de {file_path}")
            result = self.process_file(file_path, skip_stages, only_stages)
            all_results[str(file_path)] = result
        
        # 7. Étiquetage et gestion des métadonnées (sur l'ensemble du corpus)
        if (only_stages is None or "metadata" in only_stages) and \
           (skip_stages is None or "metadata" not in skip_stages):
            
            metadata_output = self._time_stage(
                "metadata",
                self.metadata_manager.process,
                self.base_output_dir / "cleaned",
                "csv"
            )
            all_results["metadata"] = metadata_output
        
        # 9. Vérification finale (statistiques globales)
        if (only_stages is None or "quality_check" in only_stages) and \
           (skip_stages is None or "quality_check" not in skip_stages):
            
            quality_report = self.quality_checker.quality_metrics
            all_results["quality_report"] = quality_report
        
        logging.info(f"Traitement du répertoire terminé: {len(all_results)} fichiers traités")
        
        # Générer un rapport de traitement global
        self._generate_report(all_results)
        
        return all_results
    
    def _generate_report(self, results: Dict[str, Any]):
        """
        Génère un rapport de traitement global.
        
        Entrées :
            results (Dict[str, Any]) : Résultats du traitement
        """
        report_path = self.base_output_dir / "pipeline_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Rapport de traitement du pipeline audio\n")
            f.write("="*80 + "\n\n")
            
            # Statistiques globales
            total_files = len([k for k in results.keys() if k != "metadata" and k != "quality_report"])
            f.write(f"Nombre de fichiers traités: {total_files}\n\n")
            
            # Statistiques des étapes
            if "quality_report" in results:
                quality = results["quality_report"]
                f.write("Statistiques de qualité:\n")
                f.write(f"  - Segments traités: {quality.get('total_segments', 0)}\n")
                f.write(f"  - Segments validés: {quality.get('valid_segments', 0)} "
                       f"({quality.get('valid_segments', 0)/max(1, quality.get('total_segments', 1))*100:.2f}%)\n")
                f.write(f"  - SNR moyen: {quality.get('avg_snr', 0):.2f} dB\n\n")
            
            # Temps d'exécution des étapes
            f.write("Temps d'exécution par étape:\n")
            for stage, timing in self.stage_timings.items():
                f.write(f"  - {stage}: {timing:.2f}s\n")
            
            f.write(f"\nRapport généré le: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logging.info(f"Rapport de traitement généré: {report_path}")
    
    def run_full_pipeline(self, 
                         input_dir: Union[str, Path],
                         skip_stages: Optional[List[str]] = None,
                         max_files: Optional[int] = None) -> Dict[str, Any]:
        """
        Exécute le pipeline complet sur un répertoire d'entrée.
        
        Entrées :
            input_dir (Union[str, Path]) : Répertoire contenant les fichiers audio à traiter
            skip_stages (Optional[List[str]]) : Étapes à sauter
            max_files (Optional[int]) : Nombre maximum de fichiers à traiter
        
        Sorties :
            Dict[str, Any] : Résultats du traitement
        """
        logging.info(f"Exécution du pipeline complet sur {input_dir}")
        
        start_time = time.time()
        results = self.process_directory(
            input_dir,
            recursive=True,
            skip_stages=skip_stages,
            max_files=max_files
        )
        end_time = time.time()
        
        total_duration = end_time - start_time
        logging.info(f"Pipeline terminé en {total_duration:.2f} secondes")
        
        return results
