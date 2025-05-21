"""
Module de diarisation pour l'isolation des locuteurs dans les fichiers audio.

Ce module permet d'identifier et de séparer les différents intervenants dans un
fichier audio en utilisant pyannote.audio.

Classes principales :
- Diarization : Classe gérant la diarisation des fichiers audio
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment, Timeline, Annotation
import librosa
from .mongo_logger import MongoLogger

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/audio_processing.log'),
        logging.StreamHandler()
    ]
)

class Diarization:
    """
    Classe pour la diarisation des fichiers audio.
    
    Cette classe utilise pyannote.audio pour identifier et séparer les différents
    intervenants dans un fichier audio.
    """
    
    def __init__(self, 
                auth_token: Optional[str] = None,
                min_speakers: int = 1,
                max_speakers: int = 5,
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialise le système de diarisation avec les paramètres spécifiés.
        
        Entrées :
            auth_token (Optional[str]) : Token d'authentification HuggingFace
            min_speakers (int) : Nombre minimum de locuteurs attendus
            max_speakers (int) : Nombre maximum de locuteurs attendus
            device (str) : Appareil à utiliser pour l'inférence ('cuda' ou 'cpu')
        """
        # Vérifier et récupérer le token d'authentification
        self.auth_token = auth_token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        
        if not self.auth_token:
            logging.error("Aucun token Hugging Face trouvé. Veuillez définir la variable d'environnement HF_TOKEN ou HUGGINGFACE_TOKEN.")
            logging.error("Visitez https://hf.co/settings/tokens pour créer votre token d'accès.")
            logging.error("Puis, acceptez les conditions d'utilisation sur https://hf.co/pyannote/speaker-diarization-3.1")
            raise ValueError("Token Hugging Face manquant. Voir les logs pour plus d'informations.")
            
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.device = device
        self.logger = MongoLogger()
        
        # Créer le dossier de sortie s'il n'existe pas
        self.output_dir = Path("processed/diarized")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialiser le pipeline de diarisation
        try:
            logging.info(f"Chargement du pipeline de diarisation avec le token HF (longueur: {len(self.auth_token)})")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.auth_token
            )
            
            if self.pipeline is None:
                raise ValueError("Le pipeline n'a pas pu être chargé correctement")
                
            self.pipeline = self.pipeline.to(torch.device(self.device))
            logging.info(f"Pipeline de diarisation initialisé avec succès sur {self.device}")
        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation du pipeline de diarisation: {e}")
            logging.error("Assurez-vous que:")
            logging.error("1. Votre token HF est valide (vérifiez sur https://hf.co/settings/tokens)")
            logging.error("2. Vous avez accepté les conditions d'utilisation sur https://hf.co/pyannote/speaker-diarization-3.1")
            logging.error("3. Votre token a les permissions nécessaires pour accéder à ce modèle")
            raise
    
    def process_file(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Effectue la diarisation d'un fichier audio.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier audio à traiter
        
        Sorties :
            Optional[Dict[str, Any]] : Résultats de la diarisation ou None en cas d'échec
        """
        file_path = Path(file_path)
        logging.info(f"Diarisation du fichier: {file_path}")
        
        # Récupérer le document MongoDB pour ce fichier
        mongo_doc = self.logger.get_processing_status(file_path.name)
        if not mongo_doc:
            logging.error(f"Document MongoDB non trouvé pour {file_path}")
            return None
        
        doc_id = str(mongo_doc["_id"])
        
        try:
            # Charger le fichier audio avec librosa pour obtenir sa durée
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Effectuer la diarisation
            diarization = self.pipeline(
                str(file_path),
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )
            
            # Extraire les informations de diarisation
            speakers = {}
            timeline_by_speaker = {}
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speakers:
                    speakers[speaker] = []
                    timeline_by_speaker[speaker] = Timeline()
                
                segment = {
                    "start": turn.start,
                    "end": turn.end,
                    "duration": turn.end - turn.start
                }
                speakers[speaker].append(segment)
                timeline_by_speaker[speaker].add(Segment(turn.start, turn.end))
            
            # Calculer la durée totale par locuteur
            speaker_stats = {}
            for speaker, segments in speakers.items():
                total_duration = sum(segment["duration"] for segment in segments)
                speaker_stats[speaker] = {
                    "n_segments": len(segments),
                    "total_duration": total_duration,
                    "percentage": (total_duration / duration) * 100 if duration > 0 else 0
                }
            
            # Préparer les résultats
            results = {
                "n_speakers": len(speakers),
                "speakers": speakers,
                "speaker_stats": speaker_stats,
                "file_duration": duration
            }
            
            # Sauvegarder la diarisation dans MongoDB
            self.logger.update_stage(doc_id, "diarized", True, results)
            
            # Extraire chaque locuteur dans un fichier séparé
            for speaker, timeline in timeline_by_speaker.items():
                speaker_segments = []
                for segment in timeline:
                    # Extraire le segment audio
                    start_sample = int(segment.start * sr)
                    end_sample = int(segment.end * sr)
                    
                    # Vérifier que les indices sont valides
                    if start_sample >= end_sample or end_sample > len(y):
                        continue
                    
                    speaker_segments.append(y[start_sample:end_sample])
                
                if speaker_segments:
                    # Concaténer tous les segments du locuteur
                    speaker_audio = np.concatenate(speaker_segments)
                    
                    # Sauvegarder le fichier du locuteur
                    speaker_file = self.output_dir / f"{file_path.stem}_speaker_{speaker}.wav"
                    sf.write(speaker_file, speaker_audio, sr, subtype='PCM_16')
                    
                    # Ajouter le chemin du fichier aux résultats
                    results["speakers"][speaker].append({
                        "output_file": str(speaker_file),
                        "duration": len(speaker_audio) / sr
                    })
                    
                    # Enregistrer le segment dans MongoDB
                    self.logger.add_segment(
                        doc_id=doc_id,
                        segment_file=str(speaker_file),
                        speaker_id=speaker,
                        start_time=0,  # Ce fichier est une concaténation, donc on commence à 0
                        end_time=len(speaker_audio) / sr,
                        metadata={
                            "type": "speaker_diarization",
                            "n_original_segments": len(speakers[speaker]),
                            "diarization_confidence": 0.8  # À ajuster selon le modèle
                        }
                    )
            
            logging.info(f"Diarisation terminée: {len(speakers)} locuteurs détectés")
            return results
            
        except Exception as e:
            logging.error(f"Erreur lors de la diarisation de {file_path}: {e}")
            self.logger.update_stage(doc_id, "diarized", False, {"error": str(e)})
            return None
    
    def process_directory(self, dir_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Traite tous les fichiers audio d'un répertoire.
        
        Entrées :
            dir_path (Union[str, Path]) : Chemin du répertoire à traiter
        
        Sorties :
            List[Dict[str, Any]] : Liste des résultats de diarisation pour chaque fichier
        """
        dir_path = Path(dir_path)
        logging.info(f"Diarisation du répertoire: {dir_path}")
        
        # Trouver tous les fichiers WAV dans le répertoire
        wav_files = list(dir_path.glob("**/*.wav"))
        
        results = []
        for file_path in wav_files:
            result = self.process_file(file_path)
            if result:
                results.append({
                    "file": str(file_path),
                    "diarization": result
                })
        
        logging.info(f"Diarisation terminée: {len(results)} fichiers traités")
        return results
