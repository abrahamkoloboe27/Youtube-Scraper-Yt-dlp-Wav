"""
Module de suppression des silences dans les fichiers audio.

Ce module permet de détecter et de supprimer automatiquement les portions sans voix
dans les fichiers audio, en utilisant WebRTC VAD ou pyannote.audio.

Classes principales :
- SilenceRemover : Classe gérant la détection et la suppression des silences
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import soundfile as sf
import librosa
import webrtcvad
from pydub import AudioSegment
from pydub.silence import split_on_silence
import contextlib
import wave
from .mongo_logger import MongoLogger

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s │ %(levelname)s │ %(message)s ",
    datefmt="%Y-%m-%d %H:%M:%S",

    handlers=[
        logging.FileHandler('logs/audio_processing.log'),
        logging.StreamHandler()
    ]
)


import pyannote
pyannote_logger = logging.getLogger("pyannote.audio")
pyannote_logger.setLevel(logging.INFO)  

class SilenceRemover:
    """
    Classe pour la détection et la suppression des silences dans les fichiers audio.
    
    Cette classe utilise WebRTC VAD ou pyannote.audio pour détecter et supprimer
    les portions sans voix dans les fichiers audio.
    """
    
    def __init__(self, 
                method: str = 'webrtcvad',
                vad_aggressiveness: int = 3,
                min_silence_duration: int = 500,  # en ms
                silence_threshold: int = -32,  # en dB
                keep_silence: int = 100,  # en ms
                min_segment_duration: float = 0.5):  # en secondes
        """
        Initialise le suppresseur de silence avec les paramètres spécifiés.
        
        Entrées :
            method (str) : Méthode de détection ('webrtcvad' ou 'pydub')
            vad_aggressiveness (int) : Niveau d'agressivité du VAD (0-3, 3 étant le plus agressif)
            min_silence_duration (int) : Durée minimale de silence à détecter (en ms)
            silence_threshold (int) : Seuil de détection du silence (en dB)
            keep_silence (int) : Durée de silence à conserver autour des segments de parole (en ms)
            min_segment_duration (float) : Durée minimale des segments de parole (en secondes)
        """
        self.method = method.lower()
        self.vad_aggressiveness = vad_aggressiveness
        self.min_silence_duration = min_silence_duration
        self.silence_threshold = silence_threshold
        self.keep_silence = keep_silence
        self.min_segment_duration = min_segment_duration
        self.logger = MongoLogger()
        
        # Vérifier la méthode spécifiée
        if self.method not in ['webrtcvad', 'pydub']:
            raise ValueError("La méthode doit être 'webrtcvad' ou 'pydub'")
        
        # Initialiser le VAD si nécessaire
        if self.method == 'webrtcvad':
            self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        
        # Créer le dossier de sortie s'il n'existe pas
        self.output_dir = Path("processed/silence_removed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _read_wave(self, path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Lit un fichier WAV et retourne les données et la fréquence d'échantillonnage.
        
        Entrées :
            path (Union[str, Path]) : Chemin du fichier WAV
        
        Sorties :
            Tuple[np.ndarray, int] : Données audio et fréquence d'échantillonnage
        """
        with contextlib.closing(wave.open(str(path), 'rb')) as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            pcm_data = wf.readframes(wf.getnframes())
            
            # Convertir les données PCM en tableau numpy
            if n_channels == 1:
                samples = np.frombuffer(pcm_data, dtype=np.int16)
            else:
                # Convertir en mono en faisant la moyenne des canaux
                samples = np.frombuffer(pcm_data, dtype=np.int16)
                samples = samples.reshape(-1, n_channels)
                samples = np.mean(samples, axis=1).astype(np.int16)
            
            return samples, sample_rate
    
    def _frame_generator(self, 
                         audio: np.ndarray, 
                         sample_rate: int, 
                         frame_duration_ms: int = 30) -> np.ndarray:
        """
        Génère des trames audio pour le traitement VAD.
        
        Entrées :
            audio (np.ndarray) : Données audio
            sample_rate (int) : Fréquence d'échantillonnage
            frame_duration_ms (int) : Durée de trame en millisecondes
        
        Sorties :
            np.ndarray : Trame audio
        """
        frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
        offset = 0
        while offset + frame_size < len(audio):
            yield audio[offset:offset + frame_size]
            offset += frame_size
    
    def remove_silence_webrtcvad(self, 
                              audio: np.ndarray, 
                              sample_rate: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Supprime les silences en utilisant WebRTC VAD.
        
        Entrées :
            audio (np.ndarray) : Données audio
            sample_rate (int) : Fréquence d'échantillonnage
        
        Sorties :
            Tuple[np.ndarray, Dict[str, Any]] : 
                - Audio sans silence
                - Métadonnées de suppression
        """
        # S'assurer que l'échantillonnage est compatible avec WebRTC VAD (8kHz, 16kHz, 32kHz ou 48kHz)
        valid_sample_rates = [8000, 16000, 32000, 48000]
        if sample_rate not in valid_sample_rates:
            # Rééchantillonner à 16kHz pour WebRTC VAD
            audio = librosa.resample(audio.astype(float), orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Convertir en format int16 pour WebRTC VAD
        if audio.dtype != np.int16:
            audio = np.array(audio * 32767, dtype=np.int16)
        
        # Détecter les segments de parole
        frame_duration_ms = 30  # WebRTC VAD fonctionne avec des trames de 10, 20 ou 30 ms
        speech_frames = []
        for frame in self._frame_generator(audio, sample_rate, frame_duration_ms):
            if len(frame) < sample_rate * (frame_duration_ms / 1000.0):
                continue  # Ignorer les trames incomplètes
            is_speech = self.vad.is_speech(frame.tobytes(), sample_rate)
            speech_frames.append((frame, is_speech))
        
        # Regrouper les segments contigus de parole
        silence_threshold_frames = int(self.min_silence_duration / frame_duration_ms)
        speech_segments = []
        current_segment = []
        
        for i, (frame, is_speech) in enumerate(speech_frames):
            if is_speech:
                current_segment.append(frame)
            elif current_segment:
                # Si on a dépassé la durée minimale de silence ou si c'est le dernier segment
                if len(current_segment) >= silence_threshold_frames or i == len(speech_frames) - 1:
                    speech_segments.append(np.concatenate(current_segment))
                    current_segment = []
        
        # Ajouter le dernier segment s'il y en a un
        if current_segment:
            speech_segments.append(np.concatenate(current_segment))
        
        # Concaténer tous les segments
        if speech_segments:
            audio_no_silence = np.concatenate(speech_segments)
        else:
            # Retourner l'audio original si aucun segment n'a été trouvé
            audio_no_silence = audio
        
        # Calculer les statistiques
        original_duration = len(audio) / sample_rate
        processed_duration = len(audio_no_silence) / sample_rate
        silent_duration = original_duration - processed_duration
        silent_percentage = (silent_duration / original_duration) * 100 if original_duration > 0 else 0
        
        metadata = {
            'method': 'webrtcvad',
            'vad_aggressiveness': self.vad_aggressiveness,
            'original_duration': original_duration,
            'processed_duration': processed_duration,
            'silent_duration': silent_duration,
            'silent_percentage': silent_percentage,
            'n_speech_segments': len(speech_segments)
        }
        
        return audio_no_silence, metadata
    
    def remove_silence_pydub(self, 
                          audio_path: Union[str, Path]) -> Tuple[AudioSegment, Dict[str, Any]]:
        """
        Supprime les silences en utilisant pydub.
        
        Entrées :
            audio_path (Union[str, Path]) : Chemin du fichier audio
        
        Sorties :
            Tuple[AudioSegment, Dict[str, Any]] : 
                - Audio sans silence
                - Métadonnées de suppression
        """
        # Charger l'audio avec pydub
        audio = AudioSegment.from_file(str(audio_path))
        
        # Calculer les statistiques avant traitement
        original_duration = len(audio) / 1000.0  # en secondes
        
        # Détecter et supprimer les silences
        segments = split_on_silence(
            audio,
            min_silence_len=self.min_silence_duration,
            silence_thresh=self.silence_threshold,
            keep_silence=self.keep_silence
        )
        
        # Filtrer les segments trop courts
        segments = [seg for seg in segments if len(seg) / 1000.0 >= self.min_segment_duration]
        
        # Concaténer les segments
        if segments:
            audio_no_silence = segments[0]
            for segment in segments[1:]:
                audio_no_silence += segment
        else:
            # Retourner l'audio original si aucun segment n'a été trouvé
            audio_no_silence = audio
        
        # Calculer les statistiques
        processed_duration = len(audio_no_silence) / 1000.0  # en secondes
        silent_duration = original_duration - processed_duration
        silent_percentage = (silent_duration / original_duration) * 100 if original_duration > 0 else 0
        
        metadata = {
            'method': 'pydub',
            'min_silence_duration': self.min_silence_duration,
            'silence_threshold': self.silence_threshold,
            'keep_silence': self.keep_silence,
            'original_duration': original_duration,
            'processed_duration': processed_duration,
            'silent_duration': silent_duration,
            'silent_percentage': silent_percentage,
            'n_speech_segments': len(segments)
        }
        
        return audio_no_silence, metadata
    
    def process_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Traite un fichier audio : détecte et supprime les silences.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier audio à traiter
        
        Sorties :
            Optional[str] : Chemin du fichier traité, ou None en cas d'échec
        """
        file_path = Path(file_path)
        logging.info(f"Suppression des silences dans le fichier: {file_path}")
        
        # Récupérer le document MongoDB pour ce fichier
        mongo_doc = self.logger.get_processing_status(file_path.name)
        if not mongo_doc:
            logging.error(f"Document MongoDB non trouvé pour {file_path}")
            return None
        
        doc_id = str(mongo_doc["_id"])
        
        try:
            # Définir le chemin de sortie
            output_path = self.output_dir / file_path.name
            
            if self.method == 'webrtcvad':
                # Charger l'audio et supprimer les silences avec WebRTC VAD
                audio, sample_rate = self._read_wave(file_path)
                audio_no_silence, metadata = self.remove_silence_webrtcvad(audio, sample_rate)
                
                # Sauvegarder l'audio sans silence
                sf.write(output_path, audio_no_silence, sample_rate, subtype='PCM_16')
                
            elif self.method == 'pydub':
                # Supprimer les silences avec pydub
                audio_no_silence, metadata = self.remove_silence_pydub(file_path)
                
                # Sauvegarder l'audio sans silence
                audio_no_silence.export(output_path, format="wav")
            
            # Mettre à jour le document MongoDB
            metadata['output_path'] = str(output_path)
            self.logger.update_stage(doc_id, "silence_removed", True, metadata)
            
            logging.info(f"Suppression des silences terminée: {output_path}")
            logging.info(f"Durée originale: {metadata['original_duration']:.2f}s, "
                        f"durée traitée: {metadata['processed_duration']:.2f}s, "
                        f"silence supprimé: {metadata['silent_percentage']:.2f}%")
            
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Erreur lors de la suppression des silences de {file_path}: {e}")
            self.logger.update_stage(doc_id, "silence_removed", False, {"error": str(e)})
            return None
    
    def process_directory(self, dir_path: Union[str, Path]) -> List[str]:
        """
        Traite tous les fichiers audio d'un répertoire.
        
        Entrées :
            dir_path (Union[str, Path]) : Chemin du répertoire à traiter
        
        Sorties :
            List[str] : Liste des chemins des fichiers traités
        """
        dir_path = Path(dir_path)
        logging.info(f"Suppression des silences dans le répertoire: {dir_path}")
        
        # Trouver tous les fichiers WAV dans le répertoire
        wav_files = list(dir_path.glob("**/*.wav"))
        
        processed_files = []
        for file_path in wav_files:
            output_path = self.process_file(file_path)
            if output_path:
                processed_files.append(output_path)
        
        logging.info(f"Suppression des silences terminée: {len(processed_files)} fichiers traités")
        return processed_files
