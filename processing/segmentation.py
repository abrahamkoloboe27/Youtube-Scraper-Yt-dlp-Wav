"""
Module de segmentation fine des fichiers audio.

Ce module permet de segmenter les fichiers audio en segments plus courts,
idéalement de 6 à 10 secondes ou par phrase complète.

Classes principales :
- Segmentation : Classe gérant la segmentation fine des fichiers audio
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
import torch
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


class Segmentation:
    """
    Classe pour la segmentation fine des fichiers audio.
    
    Cette classe permet de segmenter les fichiers audio en segments plus courts,
    idéalement de 6 à 10 secondes ou par phrase complète, en utilisant différentes
    méthodes de segmentation.
    """
    
    def __init__(self, 
                method: str = 'adaptive',
                target_length: float = 8.0,  # en secondes
                min_segment_length: float = 2.0,  # en secondes
                max_segment_length: float = 15.0,  # en secondes
                silence_threshold: int = -32,  # en dB
                min_silence_length: int = 500,  # en ms
                keep_silence: int = 200):  # en ms
        """
        Initialise le segmenteur audio avec les paramètres spécifiés.
        
        Entrées :
            method (str) : Méthode de segmentation ('fixed', 'silence', 'adaptive')
            target_length (float) : Durée cible des segments en secondes
            min_segment_length (float) : Durée minimale des segments en secondes
            max_segment_length (float) : Durée maximale des segments en secondes
            silence_threshold (int) : Seuil de détection du silence en dB
            min_silence_length (int) : Durée minimale de silence pour la segmentation en ms
            keep_silence (int) : Durée de silence à conserver autour des segments en ms
        """
        self.method = method.lower()
        self.target_length = target_length
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.silence_threshold = silence_threshold
        self.min_silence_length = min_silence_length
        self.keep_silence = keep_silence
        self.logger = MongoLogger()
        
        # Vérifier la méthode spécifiée
        if self.method not in ['fixed', 'silence', 'adaptive']:
            raise ValueError("La méthode doit être 'fixed', 'silence' ou 'adaptive'")
        
        # Créer le dossier de sortie s'il n'existe pas
        self.output_dir = Path("processed/segmented")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def segment_fixed(self, 
                     audio: np.ndarray, 
                     sample_rate: int) -> List[Tuple[np.ndarray, float, float]]:
        """
        Segmente un audio en morceaux de durée fixe.
        
        Entrées :
            audio (np.ndarray) : Données audio
            sample_rate (int) : Fréquence d'échantillonnage
        
        Sorties :
            List[Tuple[np.ndarray, float, float]] : 
                Liste de tuples (segment audio, temps de début, temps de fin)
        """
        segment_length_samples = int(self.target_length * sample_rate)
        audio_length_samples = len(audio)
        
        segments = []
        start_sample = 0
        
        while start_sample < audio_length_samples:
            # Calculer la fin du segment
            end_sample = min(start_sample + segment_length_samples, audio_length_samples)
            
            # Calculer les temps de début et de fin
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            
            # Extraire le segment
            segment = audio[start_sample:end_sample]
            
            # Vérifier si le segment est assez long
            if end_time - start_time >= self.min_segment_length:
                segments.append((segment, start_time, end_time))
            
            # Passer au segment suivant
            start_sample = end_sample
        
        return segments
    
    def segment_silence(self, 
                       audio_path: Union[str, Path]) -> List[Tuple[AudioSegment, float, float]]:
        """
        Segmente un audio en utilisant la détection de silence avec pydub.
        
        Entrées :
            audio_path (Union[str, Path]) : Chemin du fichier audio
        
        Sorties :
            List[Tuple[AudioSegment, float, float]] : 
                Liste de tuples (segment audio, temps de début, temps de fin)
        """
        # Charger l'audio avec pydub
        audio = AudioSegment.from_file(str(audio_path))
        
        # Détecter les segments non silencieux
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=self.min_silence_length,
            silence_thresh=self.silence_threshold
        )
        
        segments = []
        for start_ms, end_ms in nonsilent_ranges:
            # Ajouter le silence conservé
            segment_start = max(0, start_ms - self.keep_silence)
            segment_end = min(len(audio), end_ms + self.keep_silence)
            
            # Extraire le segment
            segment = audio[segment_start:segment_end]
            
            # Convertir en secondes
            start_time = segment_start / 1000.0
            end_time = segment_end / 1000.0
            
            # Vérifier si le segment est assez long
            if end_time - start_time >= self.min_segment_length:
                segments.append((segment, start_time, end_time))
        
        return segments
    
    def segment_adaptive(self, 
                        audio_path: Union[str, Path]) -> List[Tuple[AudioSegment, float, float]]:
        """
        Segmente un audio en utilisant une méthode adaptive qui combine détection de silence
        et respect des durées cibles.
        
        Entrées :
            audio_path (Union[str, Path]) : Chemin du fichier audio
        
        Sorties :
            List[Tuple[AudioSegment, float, float]] : 
                Liste de tuples (segment audio, temps de début, temps de fin)
        """
        # Charger l'audio avec pydub
        audio = AudioSegment.from_file(str(audio_path))
        
        # Détecter les segments non silencieux
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=self.min_silence_length,
            silence_thresh=self.silence_threshold
        )
        
        # Si pas de segments détectés, utiliser la méthode fixed
        if not nonsilent_ranges:
            y, sr = librosa.load(audio_path, sr=None)
            fixed_segments = self.segment_fixed(y, sr)
            
            # Convertir les segments NumPy en AudioSegment
            segments = []
            for segment, start_time, end_time in fixed_segments:
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                segment_audio = audio[start_ms:end_ms]
                segments.append((segment_audio, start_time, end_time))
            
            return segments
        
        # Fusionner les segments trop courts
        target_length_ms = self.target_length * 1000
        min_length_ms = self.min_segment_length * 1000
        max_length_ms = self.max_segment_length * 1000
        
        merged_ranges = []
        current_start, current_end = nonsilent_ranges[0]
        current_length = current_end - current_start
        
        for i in range(1, len(nonsilent_ranges)):
            next_start, next_end = nonsilent_ranges[i]
            gap = next_start - current_end
            next_length = next_end - next_start
            
            # Si le segment actuel est trop court et le gap n'est pas trop grand
            if current_length < target_length_ms and gap < self.min_silence_length * 2:
                # Fusionner avec le segment suivant
                current_end = next_end
                current_length = current_end - current_start
            else:
                # Ajouter le segment actuel et passer au suivant
                merged_ranges.append((current_start, current_end))
                current_start = next_start
                current_end = next_end
                current_length = current_end - current_start
        
        # Ajouter le dernier segment
        merged_ranges.append((current_start, current_end))
        
        # Diviser les segments trop longs
        final_ranges = []
        for start_ms, end_ms in merged_ranges:
            segment_length = end_ms - start_ms
            
            # Si le segment est trop long, le diviser
            if segment_length > max_length_ms:
                n_chunks = int(np.ceil(segment_length / target_length_ms))
                chunk_length = segment_length / n_chunks
                
                for i in range(n_chunks):
                    chunk_start = start_ms + int(i * chunk_length)
                    chunk_end = start_ms + int((i + 1) * chunk_length)
                    if i == n_chunks - 1:  # dernier chunk
                        chunk_end = end_ms
                    
                    final_ranges.append((chunk_start, chunk_end))
            else:
                final_ranges.append((start_ms, end_ms))
        
        # Extraire les segments finaux
        segments = []
        for start_ms, end_ms in final_ranges:
            # Ajouter le silence conservé
            segment_start = max(0, start_ms - self.keep_silence)
            segment_end = min(len(audio), end_ms + self.keep_silence)
            
            # Extraire le segment
            segment = audio[segment_start:segment_end]
            
            # Convertir en secondes
            start_time = segment_start / 1000.0
            end_time = segment_end / 1000.0
            
            # Vérifier si le segment est assez long
            if end_time - start_time >= self.min_segment_length:
                segments.append((segment, start_time, end_time))
        
        return segments
    
    def process_file(self, 
                    file_path: Union[str, Path],
                    speaker_id: Optional[str] = None) -> List[str]:
        """
        Traite un fichier audio : le segmente et sauvegarde les segments.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier audio à traiter
            speaker_id (Optional[str]) : Identifiant du locuteur, si disponible
        
        Sorties :
            List[str] : Liste des chemins des segments créés
        """
        file_path = Path(file_path)
        logging.info(f"Segmentation du fichier: {file_path}")
        
        # Récupérer le document MongoDB pour ce fichier
        original_file = file_path.name
        # Si c'est un fichier segmenté par diarisation, on récupère le fichier original
        if "_speaker_" in original_file:
            original_file = re.sub(r"_speaker_[^\.]+", "", original_file)
            
        mongo_doc = self.logger.get_processing_status(original_file)
        if not mongo_doc:
            logging.error(f"Document MongoDB non trouvé pour {original_file}")
            return []
        
        doc_id = str(mongo_doc["_id"])
        
        try:
            # Segmenter l'audio selon la méthode choisie
            segments = []
            
            if self.method == 'fixed':
                y, sr = librosa.load(file_path, sr=None)
                segments = [(seg, start, end) for seg, start, end in self.segment_fixed(y, sr)]
                
            elif self.method == 'silence':
                segments = self.segment_silence(file_path)
                
            elif self.method == 'adaptive':
                segments = self.segment_adaptive(file_path)
            
            # Sauvegarder les segments
            output_paths = []
            for i, (segment, start_time, end_time) in enumerate(segments):
                # Générer un nom de fichier pour le segment
                if speaker_id:
                    segment_file = self.output_dir / f"{file_path.stem}_speaker_{speaker_id}_seg{i+1:03d}.wav"
                else:
                    segment_file = self.output_dir / f"{file_path.stem}_seg{i+1:03d}.wav"
                
                # Sauvegarder le segment
                if isinstance(segment, np.ndarray):
                    # Si c'est un tableau NumPy (librosa/fixed)
                    y, sr = librosa.load(file_path, sr=None)
                    sf.write(segment_file, segment, sr, subtype='PCM_16')
                else:
                    # Si c'est un AudioSegment (pydub/silence/adaptive)
                    segment.export(segment_file, format="wav")
                
                output_paths.append(str(segment_file))
                
                # Enregistrer le segment dans MongoDB
                segment_duration = end_time - start_time
                self.logger.add_segment(
                    doc_id=doc_id,
                    segment_file=str(segment_file),
                    speaker_id=speaker_id or "unknown",
                    start_time=start_time,
                    end_time=end_time,
                    metadata={
                        "type": "fine_segmentation",
                        "method": self.method,
                        "original_file": str(file_path),
                        "segment_index": i + 1,
                        "duration": segment_duration
                    }
                )
            
            # Mettre à jour le document MongoDB
            segmentation_details = {
                "method": self.method,
                "n_segments": len(segments),
                "output_files": output_paths,
                "target_length": self.target_length,
                "min_segment_length": self.min_segment_length,
                "max_segment_length": self.max_segment_length
            }
            self.logger.update_stage(doc_id, "segmented", True, segmentation_details)
            
            logging.info(f"Segmentation terminée: {len(segments)} segments créés")
            return output_paths
            
        except Exception as e:
            logging.error(f"Erreur lors de la segmentation de {file_path}: {e}")
            self.logger.update_stage(doc_id, "segmented", False, {"error": str(e)})
            return []
    
    def process_directory(self, 
                         dir_path: Union[str, Path],
                         speaker_prefix: bool = False) -> Dict[str, List[str]]:
        """
        Traite tous les fichiers audio d'un répertoire.
        
        Entrées :
            dir_path (Union[str, Path]) : Chemin du répertoire à traiter
            speaker_prefix (bool) : Si True, considère que les fichiers sont des sorties de diarisation
        
        Sorties :
            Dict[str, List[str]] : Dictionnaire avec les fichiers traités par locuteur
        """
        dir_path = Path(dir_path)
        logging.info(f"Segmentation du répertoire: {dir_path}")
        
        # Trouver tous les fichiers WAV dans le répertoire
        wav_files = list(dir_path.glob("**/*.wav"))
        
        results = {}
        for file_path in wav_files:
            # Déterminer le locuteur si speaker_prefix est True
            speaker_id = None
            if speaker_prefix:
                match = re.search(r"_speaker_([^\.]+)", file_path.name)
                if match:
                    speaker_id = match.group(1)
            
            output_paths = self.process_file(file_path, speaker_id)
            if output_paths:
                speaker_key = speaker_id or "unknown"
                if speaker_key not in results:
                    results[speaker_key] = []
                results[speaker_key].extend(output_paths)
        
        logging.info(f"Segmentation terminée: {sum(len(v) for v in results.values())} segments créés")
        return results
