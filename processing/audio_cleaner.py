"""
Module de nettoyage sonore pour les fichiers audio.

Ce module permet d'améliorer la qualité des fichiers audio en réduisant le bruit,
en appliquant des filtres et en mesurant la qualité du signal.

Classes principales :
- AudioCleaner : Classe gérant le nettoyage sonore des fichiers audio
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
from scipy import signal
import scipy.io.wavfile as wavfile
import torch
import pyloudnorm as pyln
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

class AudioCleaner:
    """
    Classe pour le nettoyage sonore des fichiers audio.
    
    Cette classe permet d'améliorer la qualité des fichiers audio en réduisant le bruit,
    en appliquant des filtres et en mesurant la qualité du signal.
    """
    
    def __init__(self, 
                 noise_reduction: bool = True,
                 highpass_cutoff: Optional[int] = 80,  # Hz
                 lowpass_cutoff: Optional[int] = 8000,  # Hz
                 noise_stationary: bool = True,
                 noise_reduction_strength: float = 0.8,
                 quality_threshold_snr: float = 20.0,  # dB
                 apply_compression: bool = False):
        """
        Initialise le nettoyeur audio avec les paramètres spécifiés.
        
        Entrées :
            noise_reduction (bool) : Si True, applique la réduction de bruit
            highpass_cutoff (Optional[int]) : Fréquence de coupure du filtre passe-haut (Hz)
            lowpass_cutoff (Optional[int]) : Fréquence de coupure du filtre passe-bas (Hz)
            noise_stationary (bool) : Si True, considère le bruit comme stationnaire
            noise_reduction_strength (float) : Force de la réduction de bruit (0.0-1.0)
            quality_threshold_snr (float) : Seuil de SNR pour la qualité acceptable (dB)
            apply_compression (bool) : Si True, applique une compression dynamique
        """
        self.noise_reduction = noise_reduction
        self.highpass_cutoff = highpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff
        self.noise_stationary = noise_stationary
        self.noise_reduction_strength = noise_reduction_strength
        self.quality_threshold_snr = quality_threshold_snr
        self.apply_compression = apply_compression
        self.logger = MongoLogger()
        
        # Créer le dossier de sortie s'il n'existe pas
        self.output_dir = Path("processed/cleaned")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def estimate_snr(self, audio: np.ndarray) -> float:
        """
        Estime le rapport signal/bruit (SNR) d'un signal audio.
        
        Entrées :
            audio (np.ndarray) : Données audio
        
        Sorties :
            float : Estimation du SNR en dB
        """
        # Estimer le niveau de signal
        signal_power = np.mean(audio ** 2)
        
        # Estimer le niveau de bruit (on utilise les 100ms les plus calmes)
        frame_length = 1600  # ~100ms à 16kHz
        if len(audio) < frame_length:
            noise_power = signal_power * 0.1  # Approximation
        else:
            windowed = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length // 2)
            frame_powers = np.mean(windowed ** 2, axis=0)
            noise_power = np.min(frame_powers)
        
        # Calculer le SNR
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 100.0  # Valeur arbitrairement élevée
        
        return snr
    
    def apply_highpass_filter(self, 
                             audio: np.ndarray, 
                             sample_rate: int, 
                             cutoff_freq: int) -> np.ndarray:
        """
        Applique un filtre passe-haut au signal audio.
        
        Entrées :
            audio (np.ndarray) : Données audio
            sample_rate (int) : Fréquence d'échantillonnage
            cutoff_freq (int) : Fréquence de coupure en Hz
        
        Sorties :
            np.ndarray : Audio filtré
        """
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        
        # Créer un filtre Butterworth d'ordre 4
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        
        # Appliquer le filtre
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def apply_lowpass_filter(self, 
                            audio: np.ndarray, 
                            sample_rate: int, 
                            cutoff_freq: int) -> np.ndarray:
        """
        Applique un filtre passe-bas au signal audio.
        
        Entrées :
            audio (np.ndarray) : Données audio
            sample_rate (int) : Fréquence d'échantillonnage
            cutoff_freq (int) : Fréquence de coupure en Hz
        
        Sorties :
            np.ndarray : Audio filtré
        """
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        
        # Créer un filtre Butterworth d'ordre 4
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        
        # Appliquer le filtre
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def apply_noise_reduction(self, 
                             audio: np.ndarray, 
                             sample_rate: int) -> np.ndarray:
        """
        Applique une réduction de bruit au signal audio.
        
        Entrées :
            audio (np.ndarray) : Données audio
            sample_rate (int) : Fréquence d'échantillonnage
        
        Sorties :
            np.ndarray : Audio avec bruit réduit
        """
        # Appliquer la réduction de bruit de noisereduce
        reduced_noise = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=self.noise_stationary,
            prop_decrease=self.noise_reduction_strength
        )
        
        return reduced_noise
    
    def apply_compression(self, 
                         audio: np.ndarray, 
                         threshold: float = -20.0,  # dB
                         ratio: float = 4.0,
                         attack: float = 0.005,  # secondes
                         release: float = 0.05) -> np.ndarray:  # secondes
        """
        Applique une compression dynamique au signal audio.
        
        Entrées :
            audio (np.ndarray) : Données audio
            threshold (float) : Seuil de compression en dB
            ratio (float) : Ratio de compression
            attack (float) : Temps d'attaque en secondes
            release (float) : Temps de relâchement en secondes
        
        Sorties :
            np.ndarray : Audio compressé
        """
        # Convertir en dB
        db = 20 * np.log10(np.abs(audio) + 1e-8)
        
        # Calculer l'atténuation
        mask = db > threshold
        attenuation = np.zeros_like(audio)
        attenuation[mask] = (db[mask] - threshold) * (1 - 1 / ratio)
        
        # Convertir l'atténuation en facteur linéaire
        gain = 10 ** (-attenuation / 20)
        
        # Appliquer le gain
        compressed = audio * gain
        
        # Normaliser
        if np.max(np.abs(compressed)) > 0:
            compressed = compressed / np.max(np.abs(compressed)) * np.max(np.abs(audio))
        
        return compressed
    
    def calculate_audio_quality_metrics(self, 
                                      audio: np.ndarray, 
                                      sample_rate: int) -> Dict[str, float]:
        """
        Calcule diverses métriques de qualité audio.
        
        Entrées :
            audio (np.ndarray) : Données audio
            sample_rate (int) : Fréquence d'échantillonnage
        
        Sorties :
            Dict[str, float] : Dictionnaire des métriques de qualité
        """
        metrics = {}
        
        # SNR estimé
        metrics['snr'] = self.estimate_snr(audio)
        
        # RMS (volume)
        metrics['rms'] = np.sqrt(np.mean(audio ** 2))
        metrics['rms_db'] = 20 * np.log10(metrics['rms']) if metrics['rms'] > 0 else -100
        
        # Peak
        metrics['peak'] = np.max(np.abs(audio))
        metrics['peak_db'] = 20 * np.log10(metrics['peak']) if metrics['peak'] > 0 else -100
        
        # Métriques spectrales
        if len(audio) > 512:
            # Centroïde spectral (indique la "brillance" du son)
            metrics['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
            
            # Rolloff spectral (fréquence en dessous de laquelle se trouve 85% de l'énergie)
            metrics['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate))
            
            # Flux spectral (mesure la variation spectrale trame par trame)
            spectral = np.abs(librosa.stft(audio))
            metrics['spectral_flux'] = np.mean(np.diff(spectral, axis=1) ** 2)
            
            # Loudness EBU R128
            meter = pyln.Meter(sample_rate)
            metrics['loudness_lufs'] = meter.integrated_loudness(audio)
        
        return metrics
    
    def process_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Traite un fichier audio : réduit le bruit, applique des filtres et mesure la qualité.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier audio à traiter
        
        Sorties :
            Optional[str] : Chemin du fichier traité, ou None en cas d'échec
        """
        file_path = Path(file_path)
        logging.info(f"Nettoyage sonore du fichier: {file_path}")
        
        # Récupérer le document MongoDB pour ce fichier
        original_file = file_path.name
        # Pour les segments, on cherche le fichier original
        if "_seg" in original_file or "_speaker_" in original_file:
            parts = original_file.split("_")
            # Essayer de trouver la partie correspondant au fichier original
            for i in range(len(parts)):
                test_name = "_".join(parts[:i+1]) + ".wav"
                mongo_doc = self.logger.get_processing_status(test_name)
                if mongo_doc:
                    break
        else:
            mongo_doc = self.logger.get_processing_status(original_file)
        
        if not mongo_doc:
            logging.error(f"Document MongoDB non trouvé pour {file_path}")
            return None
        
        doc_id = str(mongo_doc["_id"])
        
        try:
            # Charger le fichier audio
            audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
            
            # Mesurer la qualité avant traitement
            before_metrics = self.calculate_audio_quality_metrics(audio, sample_rate)
            
            # Appliquer les traitements
            processed_audio = audio.copy()
            
            # 1. Appliquer le filtre passe-haut si spécifié
            if self.highpass_cutoff:
                processed_audio = self.apply_highpass_filter(processed_audio, sample_rate, self.highpass_cutoff)
            
            # 2. Appliquer le filtre passe-bas si spécifié
            if self.lowpass_cutoff:
                processed_audio = self.apply_lowpass_filter(processed_audio, sample_rate, self.lowpass_cutoff)
            
            # 3. Appliquer la réduction de bruit si activée
            if self.noise_reduction:
                processed_audio = self.apply_noise_reduction(processed_audio, sample_rate)
            
            # 4. Appliquer la compression si activée
            if self.apply_compression:
                processed_audio = self.apply_compression(processed_audio)
            
            # Mesurer la qualité après traitement
            after_metrics = self.calculate_audio_quality_metrics(processed_audio, sample_rate)
            
            # Déterminer si la qualité est acceptable
            is_quality_ok = after_metrics['snr'] >= self.quality_threshold_snr
            
            # Sauvegarder l'audio traité
            output_path = self.output_dir / file_path.name
            sf.write(output_path, processed_audio, sample_rate, subtype='PCM_16')
            
            # Préparer les métadonnées de traitement
            processing_details = {
                "before": before_metrics,
                "after": after_metrics,
                "output_path": str(output_path),
                "quality_acceptable": is_quality_ok,
                "applied_treatments": {
                    "noise_reduction": self.noise_reduction,
                    "highpass_filter": self.highpass_cutoff is not None,
                    "lowpass_filter": self.lowpass_cutoff is not None,
                    "compression": self.apply_compression
                },
                "filter_settings": {
                    "highpass_cutoff": self.highpass_cutoff,
                    "lowpass_cutoff": self.lowpass_cutoff,
                    "noise_reduction_strength": self.noise_reduction_strength
                }
            }
            
            # Mettre à jour le document MongoDB
            self.logger.update_stage(doc_id, "cleaned", True, processing_details)
            
            # Journaliser le résultat
            improvement = after_metrics['snr'] - before_metrics['snr']
            logging.info(f"Nettoyage terminé: {output_path}")
            logging.info(f"SNR avant: {before_metrics['snr']:.2f} dB, après: {after_metrics['snr']:.2f} dB, "
                        f"amélioration: {improvement:.2f} dB")
            
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Erreur lors du nettoyage sonore de {file_path}: {e}")
            self.logger.update_stage(doc_id, "cleaned", False, {"error": str(e)})
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
        logging.info(f"Nettoyage sonore du répertoire: {dir_path}")
        
        # Trouver tous les fichiers WAV dans le répertoire
        wav_files = list(dir_path.glob("**/*.wav"))
        
        processed_files = []
        for file_path in wav_files:
            output_path = self.process_file(file_path)
            if output_path:
                processed_files.append(output_path)
        
        logging.info(f"Nettoyage sonore terminé: {len(processed_files)} fichiers traités")
        return processed_files
