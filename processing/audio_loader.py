"""
Module de chargement et d'uniformisation des fichiers audio.

Ce module fournit des fonctionnalités pour charger des fichiers audio de différents formats
et les convertir en format WAV uniforme (16 kHz, 16 bits).

Classes principales :
- AudioLoader : Classe gérant le chargement et l'uniformisation des fichiers audio
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import librosa
import soundfile as sf
import torchaudio
import torch
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

class AudioLoader:
    """
    Classe pour le chargement et l'uniformisation des fichiers audio.
    
    Cette classe permet de charger des fichiers audio de différents formats et
    de les convertir en format WAV uniforme (16 kHz, 16 bits) pour le traitement
    ultérieur.
    """
    
    def __init__(self, target_sr: int = 16000, target_bits: int = 16):
        """
        Initialise le chargeur audio avec les paramètres cibles.
        
        Entrées :
            target_sr (int) : Fréquence d'échantillonnage cible en Hz (défaut: 16000)
            target_bits (int) : Nombre de bits cible (défaut: 16)
        """
        self.target_sr = target_sr
        self.target_bits = target_bits
        self.logger = MongoLogger()
        
        # Créer le dossier de sortie s'il n'existe pas
        self.output_dir = Path("processed/uniformized")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[Optional[torch.Tensor], Optional[int], Dict[str, Any]]:
        """
        Charge un fichier audio et retourne les données audio et la fréquence d'échantillonnage.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier audio
        
        Sorties :
            Tuple[Optional[torch.Tensor], Optional[int], Dict[str, Any]] : 
                - Données audio (None en cas d'échec)
                - Fréquence d'échantillonnage (None en cas d'échec)
                - Métadonnées
        """
        file_path = Path(file_path)
        metadata = {"original_format": file_path.suffix[1:]}
        
        try:
            # Essayer de charger avec torchaudio d'abord
            try:
                waveform, sample_rate = torchaudio.load(file_path)
                metadata["sample_rate"] = sample_rate
                metadata["channels"] = waveform.shape[0]
                metadata["duration"] = waveform.shape[1] / sample_rate
                metadata["loading_method"] = "torchaudio"
                return waveform, sample_rate, metadata
            except Exception as e:
                logging.warning(f"Échec de chargement avec torchaudio: {e}, essai avec librosa...")
            
            # Si ça échoue, essayer avec librosa
            y, sr = librosa.load(file_path, sr=None, mono=False)
            metadata["sample_rate"] = sr
            metadata["duration"] = librosa.get_duration(y=y, sr=sr)
            
            # Convertir en tensor torch
            if y.ndim == 1:  # Mono
                waveform = torch.tensor(y).unsqueeze(0)  # [1, n_samples]
                metadata["channels"] = 1
            else:  # Multi-canal
                waveform = torch.tensor(y)  # [n_channels, n_samples]
                metadata["channels"] = y.shape[0]
            
            metadata["loading_method"] = "librosa"
            return waveform, sr, metadata
            
        except Exception as e:
            logging.error(f"Erreur lors du chargement du fichier {file_path}: {e}")
            return None, None, {"error": str(e)}
    
    def convert_audio(self, 
                     waveform: torch.Tensor, 
                     sample_rate: int, 
                     target_sr: Optional[int] = None) -> torch.Tensor:
        """
        Convertit un audio à la fréquence d'échantillonnage cible.
        
        Entrées :
            waveform (torch.Tensor) : Données audio [n_channels, n_samples]
            sample_rate (int) : Fréquence d'échantillonnage actuelle
            target_sr (Optional[int]) : Fréquence d'échantillonnage cible (utilise self.target_sr par défaut)
        
        Sorties :
            torch.Tensor : Audio converti à la fréquence d'échantillonnage cible
        """
        if target_sr is None:
            target_sr = self.target_sr
            
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            return resampler(waveform)
        
        return waveform
    
    def convert_to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convertit un audio stéréo en mono en faisant la moyenne des canaux.
        
        Entrées :
            waveform (torch.Tensor) : Données audio [n_channels, n_samples]
        
        Sorties :
            torch.Tensor : Audio mono [1, n_samples]
        """
        if waveform.shape[0] > 1:
            return torch.mean(waveform, dim=0, keepdim=True)
        return waveform
    
    def save_audio(self, 
                  waveform: torch.Tensor, 
                  file_name: str, 
                  sample_rate: int) -> str:
        """
        Sauvegarde les données audio au format WAV.
        
        Entrées :
            waveform (torch.Tensor) : Données audio
            file_name (str) : Nom du fichier de sortie
            sample_rate (int) : Fréquence d'échantillonnage
        
        Sorties :
            str : Chemin du fichier sauvegardé
        """
        output_path = self.output_dir / f"{Path(file_name).stem}.wav"
        
        # Assurer que le format est correct pour soundfile
        audio_data = waveform.squeeze().numpy()
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)
        
        # Si l'audio est mono et a une dimension incorrecte
        if waveform.shape[0] == 1 and audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)
        
        # Transposer si nécessaire pour soundfile (qui attend [n_samples, n_channels])
        if audio_data.shape[0] < audio_data.shape[1]:
            audio_data = audio_data.T
            
        # Sauvegarder avec soundfile
        sf.write(output_path, audio_data, sample_rate, subtype='PCM_16')
        
        return str(output_path)
    
    def process_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Traite un fichier audio : charge, uniformise et sauvegarde.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier audio à traiter
        
        Sorties :
            Optional[str] : Chemin du fichier traité, ou None en cas d'échec
        """
        file_path = Path(file_path)
        logging.info(f"Traitement du fichier: {file_path}")
        
        # Créer un document dans MongoDB pour ce fichier
        original_metadata = {
            "filename": file_path.name,
            "file_size": file_path.stat().st_size,
            "path": str(file_path)
        }
        doc_id = self.logger.create_audio_document(str(file_path), original_metadata)
        
        # Charger le fichier audio
        waveform, sample_rate, metadata = self.load_audio(file_path)
        if waveform is None:
            self.logger.update_stage(doc_id, "loaded", False, {"error": "Échec de chargement"})
            return None
        
        # Fusionner les métadonnées originales et celles obtenues lors du chargement
        metadata.update(original_metadata)
        
        # Convertir à la fréquence d'échantillonnage cible
        waveform = self.convert_audio(waveform, sample_rate)
        
        # Convertir en mono si nécessaire
        waveform = self.convert_to_mono(waveform)
        
        # Sauvegarder en format WAV 16 kHz, 16 bits
        output_path = self.save_audio(waveform, file_path.name, self.target_sr)
        
        # Mettre à jour le document MongoDB
        processing_details = {
            "output_path": output_path,
            "target_sample_rate": self.target_sr,
            "target_bits": self.target_bits,
            "channels": 1,  # Mono
            "duration": waveform.shape[1] / self.target_sr
        }
        self.logger.update_stage(doc_id, "loaded", True, processing_details)
        
        logging.info(f"Fichier traité avec succès: {output_path}")
        return output_path
    
    def process_directory(self, dir_path: Union[str, Path]) -> List[str]:
        """
        Traite tous les fichiers audio d'un répertoire.
        
        Entrées :
            dir_path (Union[str, Path]) : Chemin du répertoire à traiter
        
        Sorties :
            List[str] : Liste des chemins des fichiers traités
        """
        dir_path = Path(dir_path)
        logging.info(f"Traitement du répertoire: {dir_path}")
        
        # Trouver tous les fichiers audio dans le répertoire
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(dir_path.glob(f"**/*{ext}"))
        
        processed_files = []
        for file_path in audio_files:
            output_path = self.process_file(file_path)
            if output_path:
                processed_files.append(output_path)
        
        logging.info(f"Traitement terminé: {len(processed_files)} fichiers traités")
        return processed_files
