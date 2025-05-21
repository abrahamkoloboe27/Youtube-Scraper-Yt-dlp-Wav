"""
Module de normalisation de loudness pour les fichiers audio.

Ce module permet de normaliser le volume des fichiers audio selon les standards 
EBU R128 ou RMS, pour assurer une cohérence sonore dans le corpus.

Classes principales :
- LoudnessNormalizer : Classe gérant la normalisation de volume des fichiers audio
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
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

class LoudnessNormalizer:
    """
    Classe pour la normalisation de loudness des fichiers audio.
    
    Cette classe permet de normaliser les fichiers audio selon les standards
    EBU R128 ou RMS, pour assurer une cohérence sonore dans le corpus.
    """
    
    def __init__(self, 
                target_loudness: float = -23.0, 
                method: str = 'ebu',
                block_size: float = 0.400):
        """
        Initialise le normaliseur de loudness avec les paramètres cibles.
        
        Entrées :
            target_loudness (float) : Niveau de loudness cible en LUFS (défaut: -23.0 pour EBU R128)
            method (str) : Méthode de normalisation ('ebu' pour EBU R128, 'rms' pour RMS)
            block_size (float) : Taille de bloc en secondes pour l'analyse EBU R128 (défaut: 0.400)
        """
        self.target_loudness = target_loudness
        self.method = method.lower()
        self.block_size = block_size
        self.logger = MongoLogger()
        
        # Vérifier la méthode spécifiée
        if self.method not in ['ebu', 'rms']:
            raise ValueError("La méthode doit être 'ebu' ou 'rms'")
        
        # Créer le dossier de sortie s'il n'existe pas
        self.output_dir = Path("processed/normalized")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def measure_loudness(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """
        Mesure la loudness d'un fichier audio selon la méthode spécifiée.
        
        Entrées :
            audio (np.ndarray) : Données audio [n_samples]
            sample_rate (int) : Fréquence d'échantillonnage
        
        Sorties :
            Dict[str, float] : Dictionnaire contenant les mesures de loudness
        """
        results = {}
        
        # S'assurer que l'audio est mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        # Mesure EBU R128
        meter = pyln.Meter(sample_rate, block_size=self.block_size)
        loudness = meter.integrated_loudness(audio)
        results['integrated_lufs'] = loudness
        
        # Mesure RMS
        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(rms) if rms > 0 else -100
        results['rms_db'] = rms_db
        
        # Mesures supplémentaires
        peak = np.max(np.abs(audio))
        peak_db = 20 * np.log10(peak) if peak > 0 else -100
        results['peak'] = peak
        results['peak_db'] = peak_db
        
        # Dynamique (différence entre crête et loudness)
        results['dynamic_range'] = peak_db - loudness if loudness > -100 else 0
        
        return results
    
    def normalize_audio(self, 
                       audio: np.ndarray, 
                       sample_rate: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Normalise un audio selon la méthode spécifiée.
        
        Entrées :
            audio (np.ndarray) : Données audio
            sample_rate (int) : Fréquence d'échantillonnage
        
        Sorties :
            Tuple[np.ndarray, Dict[str, float]] : 
                - Audio normalisé
                - Métadonnées de normalisation
        """
        # S'assurer que l'audio est mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        # Mesurer la loudness avant normalisation
        before_metrics = self.measure_loudness(audio, sample_rate)
        
        # Normaliser selon la méthode choisie
        if self.method == 'ebu':
            # Mesurer la loudness avec pyloudnorm
            meter = pyln.Meter(sample_rate)
            current_loudness = meter.integrated_loudness(audio)
            
            # Éviter la division par zéro ou les valeurs extrêmes
            if current_loudness < -100:
                gain = 1.0
            else:
                # Calculer le gain nécessaire
                gain = pyln.normalize.loudness(audio, current_loudness, self.target_loudness)
            
            # Appliquer le gain
            normalized_audio = audio * gain
            
        elif self.method == 'rms':
            # Normalisation RMS
            current_rms = np.sqrt(np.mean(audio**2))
            target_rms = 10 ** (self.target_loudness / 20)
            
            # Éviter la division par zéro
            if current_rms < 1e-10:
                gain = 1.0
            else:
                gain = target_rms / current_rms
            
            # Appliquer le gain
            normalized_audio = audio * gain
        
        # Mesurer la loudness après normalisation
        after_metrics = self.measure_loudness(normalized_audio, sample_rate)
        
        # Limiter les crêtes si nécessaire pour éviter l'écrêtage
        if np.max(np.abs(normalized_audio)) > 0.99:
            normalized_audio = np.clip(normalized_audio, -0.99, 0.99)
            after_metrics['peak_limited'] = True
            
        # Préparer les métadonnées de normalisation
        metadata = {
            'method': self.method,
            'target_loudness': self.target_loudness,
            'before': before_metrics,
            'after': after_metrics,
            'gain_applied': gain
        }
        
        return normalized_audio, metadata
    
    def process_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Traite un fichier audio : charge, normalise et sauvegarde.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier audio à traiter
        
        Sorties :
            Optional[str] : Chemin du fichier traité, ou None en cas d'échec
        """
        file_path = Path(file_path)
        logging.info(f"Normalisation du fichier: {file_path}")
        
        # Récupérer le document MongoDB pour ce fichier
        mongo_doc = self.logger.get_processing_status(file_path.name)
        if not mongo_doc:
            logging.error(f"Document MongoDB non trouvé pour {file_path}")
            return None
        
        doc_id = str(mongo_doc["_id"])
        
        try:
            # Charger le fichier audio
            audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
            
            # Normaliser l'audio
            normalized_audio, metadata = self.normalize_audio(audio, sample_rate)
            
            # Sauvegarder l'audio normalisé
            output_path = self.output_dir / file_path.name
            sf.write(output_path, normalized_audio, sample_rate, subtype='PCM_16')
            
            # Mettre à jour le document MongoDB
            metadata['output_path'] = str(output_path)
            self.logger.update_stage(doc_id, "loudness_normalized", True, metadata)
            
            logging.info(f"Normalisation terminée: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Erreur lors de la normalisation de {file_path}: {e}")
            self.logger.update_stage(doc_id, "loudness_normalized", False, {"error": str(e)})
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
        logging.info(f"Normalisation du répertoire: {dir_path}")
        
        # Trouver tous les fichiers WAV dans le répertoire
        wav_files = list(dir_path.glob("**/*.wav"))
        
        processed_files = []
        for file_path in wav_files:
            output_path = self.process_file(file_path)
            if output_path:
                processed_files.append(output_path)
        
        logging.info(f"Normalisation terminée: {len(processed_files)} fichiers traités")
        return processed_files
