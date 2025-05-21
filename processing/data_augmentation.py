"""
Module d'augmentation de données audio.

Ce module permet d'augmenter les données audio en appliquant diverses transformations
comme les perturbations temporelles, les perturbations fréquentielles et l'ajout de bruits.

Classes principales :
- DataAugmentation : Classe gérant l'augmentation des données audio
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Set
import numpy as np
import soundfile as sf
import librosa
import random
import torch
import torchaudio
import torchaudio.transforms as T
from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift, 
    Shift, AddBackgroundNoise
)
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

class DataAugmentation:
    """
    Classe pour l'augmentation des données audio.
    
    Cette classe permet d'augmenter les données audio en appliquant diverses transformations
    comme les perturbations temporelles, les perturbations fréquentielles et l'ajout de bruits.
    """
    
    def __init__(self, 
                 output_dir: Union[str, Path] = "processed/augmented",
                 noise_dir: Optional[Union[str, Path]] = None,
                 n_augmentations_per_sample: int = 2,
                 prob_tempo: float = 0.5,
                 prob_pitch: float = 0.5,
                 prob_noise: float = 0.3,
                 tempo_range: Tuple[float, float] = (0.9, 1.1),
                 pitch_range: Tuple[int, int] = (-2, 2),
                 noise_level_range: Tuple[float, float] = (0.001, 0.01),
                 background_noise_snr_range: Tuple[float, float] = (10, 20),
                 random_seed: int = 42):
        """
        Initialise l'augmenteur de données avec les paramètres spécifiés.
        
        Entrées :
            output_dir (Union[str, Path]) : Répertoire de sortie pour les fichiers augmentés
            noise_dir (Optional[Union[str, Path]]) : Répertoire contenant les bruits de fond
            n_augmentations_per_sample (int) : Nombre d'augmentations à créer par échantillon
            prob_tempo (float) : Probabilité d'appliquer une modification de tempo
            prob_pitch (float) : Probabilité d'appliquer une modification de hauteur
            prob_noise (float) : Probabilité d'appliquer un ajout de bruit
            tempo_range (Tuple[float, float]) : Plage de facteurs de tempo (0.9 = -10%, 1.1 = +10%)
            pitch_range (Tuple[int, int]) : Plage de modifications de hauteur en demi-tons
            noise_level_range (Tuple[float, float]) : Plage de niveaux de bruit gaussien
            background_noise_snr_range (Tuple[float, float]) : Plage de SNR pour l'ajout de bruit de fond
            random_seed (int) : Graine aléatoire pour la reproductibilité
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.noise_dir = Path(noise_dir) if noise_dir else None
        self.n_augmentations_per_sample = n_augmentations_per_sample
        self.prob_tempo = prob_tempo
        self.prob_pitch = prob_pitch
        self.prob_noise = prob_noise
        self.tempo_range = tempo_range
        self.pitch_range = pitch_range
        self.noise_level_range = noise_level_range
        self.background_noise_snr_range = background_noise_snr_range
        self.random_seed = random_seed
        self.logger = MongoLogger()
        
        # Initialiser le générateur aléatoire
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Créer les augmenteurs
        self._create_augmenters()
    
    def _create_augmenters(self):
        """
        Crée les différents augmenteurs selon les paramètres spécifiés.
        """
        # Augmenteur pour les perturbations temporelles
        self.time_augmenter = Compose([
            TimeStretch(
                min_rate=self.tempo_range[0],
                max_rate=self.tempo_range[1],
                p=self.prob_tempo
            ),
            Shift(
                min_shift=-0.05,
                max_shift=0.05,
                p=0.3
            )
        ])
        
        # Augmenteur pour les perturbations fréquentielles
        self.freq_augmenter = Compose([
            PitchShift(
                min_semitones=self.pitch_range[0],
                max_semitones=self.pitch_range[1],
                p=self.prob_pitch
            )
        ])
        
        # Augmenteur pour l'ajout de bruit gaussien
        self.noise_augmenter = Compose([
            AddGaussianNoise(
                min_amplitude=self.noise_level_range[0],
                max_amplitude=self.noise_level_range[1],
                p=self.prob_noise
            )
        ])
        
        # Augmenteur pour l'ajout de bruit de fond (si un répertoire de bruits est spécifié)
        if self.noise_dir and self.noise_dir.exists():
            self.background_noise_augmenter = Compose([
                AddBackgroundNoise(
                    sounds_path=str(self.noise_dir),
                    min_snr_in_db=self.background_noise_snr_range[0],
                    max_snr_in_db=self.background_noise_snr_range[1],
                    p=self.prob_noise
                )
            ])
        else:
            self.background_noise_augmenter = None
    
    def apply_speed_perturbation(self, 
                                 audio: np.ndarray, 
                                 sample_rate: int, 
                                 speed_factor: float) -> np.ndarray:
        """
        Applique une perturbation de vitesse au signal audio.
        
        Entrées :
            audio (np.ndarray) : Données audio
            sample_rate (int) : Fréquence d'échantillonnage
            speed_factor (float) : Facteur de vitesse (0.9 = -10%, 1.1 = +10%)
        
        Sorties :
            np.ndarray : Audio avec perturbation de vitesse
        """
        # Convertir en tensor torch
        if len(audio.shape) == 1:  # Mono
            waveform = torch.tensor(audio).unsqueeze(0)  # [1, n_samples]
        else:  # Multi-canal
            waveform = torch.tensor(audio.T)  # [n_channels, n_samples]
        
        # Appliquer le resampling pour changer la vitesse
        effects = [
            ["speed", str(speed_factor)],
            ["rate", str(sample_rate)]
        ]
        augmented, new_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate, effects
        )
        
        # Convertir en numpy
        if augmented.shape[0] == 1:  # Mono
            return augmented.squeeze().numpy()
        else:  # Multi-canal
            return augmented.numpy().T
    
    def apply_pitch_shift(self, 
                         audio: np.ndarray, 
                         sample_rate: int, 
                         n_steps: int) -> np.ndarray:
        """
        Applique une modification de hauteur au signal audio.
        
        Entrées :
            audio (np.ndarray) : Données audio
            sample_rate (int) : Fréquence d'échantillonnage
            n_steps (int) : Nombre de demi-tons à décaler
        
        Sorties :
            np.ndarray : Audio avec modification de hauteur
        """
        return librosa.effects.pitch_shift(
            audio.astype(np.float32), sr=sample_rate, n_steps=n_steps
        )
    
    def apply_spec_augment(self, 
                          audio: np.ndarray, 
                          sample_rate: int,
                          n_freq_masks: int = 2,
                          freq_mask_param: int = 10,
                          n_time_masks: int = 2,
                          time_mask_param: int = 10) -> np.ndarray:
        """
        Applique SpecAugment au signal audio (masquage fréquentiel et temporel).
        
        Entrées :
            audio (np.ndarray) : Données audio
            sample_rate (int) : Fréquence d'échantillonnage
            n_freq_masks (int) : Nombre de masques fréquentiels
            freq_mask_param (int) : Paramètre de largeur de masque fréquentiel
            n_time_masks (int) : Nombre de masques temporels
            time_mask_param (int) : Paramètre de largeur de masque temporel
        
        Sorties :
            np.ndarray : Audio avec SpecAugment appliqué
        """
        # Convertir en tensor torch
        if len(audio.shape) == 1:  # Mono
            waveform = torch.tensor(audio).unsqueeze(0)  # [1, n_samples]
        else:  # Multi-canal
            waveform = torch.tensor(audio.T)  # [n_channels, n_samples]
        
        # Calculer le spectrogramme
        n_fft = 512
        win_length = 400
        hop_length = 160
        
        spec = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )(waveform)
        
        # Appliquer SpecAugment
        spec_aug = T.FrequencyMasking(freq_mask_param)(spec)
        for _ in range(n_freq_masks - 1):
            spec_aug = T.FrequencyMasking(freq_mask_param)(spec_aug)
        
        spec_aug = T.TimeMasking(time_mask_param)(spec_aug)
        for _ in range(n_time_masks - 1):
            spec_aug = T.TimeMasking(time_mask_param)(spec_aug)
        
        # Convertir le spectrogramme en audio
        griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )(spec_aug)
        
        # Convertir en numpy
        if griffin_lim.shape[0] == 1:  # Mono
            return griffin_lim.squeeze().numpy()
        else:  # Multi-canal
            return griffin_lim.numpy().T
    
    def augment_audio(self, 
                     file_path: Union[str, Path], 
                     augmentation_types: Optional[List[str]] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Augmente un fichier audio en appliquant diverses transformations.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier audio à augmenter
            augmentation_types (Optional[List[str]]) : Types d'augmentation à appliquer
                Options : 'tempo', 'pitch', 'noise', 'background', 'specaugment'
                Si None, utilise une sélection aléatoire basée sur les probabilités
        
        Sorties :
            List[Tuple[str, Dict[str, Any]]] : 
                Liste des chemins des fichiers augmentés et leurs paramètres d'augmentation
        """
        file_path = Path(file_path)
        logging.info(f"Augmentation du fichier: {file_path}")
        
        # Récupérer le document MongoDB pour ce fichier
        original_file = file_path.name
        
        # Pour les segments, on cherche le fichier original
        if "_seg" in original_file or "_speaker_" in original_file:
            # Rechercher dans MongoDB pour trouver un segment correspondant
            segments = self.logger.collection.find(
                {"segments.file": original_file},
                {"file": 1}
            )
            for segment_doc in segments:
                if segment_doc:
                    original_file = segment_doc.get("file", original_file)
                    break
        
        mongo_doc = self.logger.get_processing_status(original_file)
        if not mongo_doc:
            logging.error(f"Document MongoDB non trouvé pour {original_file}")
            return []
        
        doc_id = str(mongo_doc["_id"])
        
        try:
            # Charger le fichier audio
            audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
            
            augmented_files = []
            
            # Déterminer les types d'augmentation à appliquer
            available_types = ['tempo', 'pitch', 'noise']
            if self.background_noise_augmenter:
                available_types.append('background')
            
            if augmentation_types is None:
                # Sélectionner aléatoirement les types d'augmentation selon les probabilités
                selected_types = []
                if random.random() < self.prob_tempo:
                    selected_types.append('tempo')
                if random.random() < self.prob_pitch:
                    selected_types.append('pitch')
                if random.random() < self.prob_noise:
                    if self.background_noise_augmenter and random.random() < 0.5:
                        selected_types.append('background')
                    else:
                        selected_types.append('noise')
                
                # Ajouter SpecAugment avec une probabilité
                if random.random() < 0.3:
                    selected_types.append('specaugment')
                
                # S'assurer qu'au moins un type est sélectionné
                if not selected_types:
                    selected_types = [random.choice(available_types)]
            else:
                selected_types = augmentation_types
            
            # Appliquer les augmentations
            for i in range(self.n_augmentations_per_sample):
                # Copier l'audio original
                aug_audio = audio.copy()
                
                # Initialiser le dictionnaire de paramètres
                aug_params = {
                    "original_file": str(file_path),
                    "applied_augmentations": []
                }
                
                # Appliquer les augmentations sélectionnées
                for aug_type in selected_types:
                    if aug_type == 'tempo':
                        # Perturbation temporelle
                        tempo_factor = np.random.uniform(self.tempo_range[0], self.tempo_range[1])
                        aug_audio = self.apply_speed_perturbation(aug_audio, sample_rate, tempo_factor)
                        aug_params["tempo_factor"] = tempo_factor
                        aug_params["applied_augmentations"].append("tempo")
                    
                    elif aug_type == 'pitch':
                        # Perturbation de hauteur
                        pitch_steps = np.random.randint(self.pitch_range[0], self.pitch_range[1] + 1)
                        aug_audio = self.apply_pitch_shift(aug_audio, sample_rate, pitch_steps)
                        aug_params["pitch_steps"] = pitch_steps
                        aug_params["applied_augmentations"].append("pitch")
                    
                    elif aug_type == 'noise':
                        # Ajout de bruit gaussien
                        noise_level = np.random.uniform(self.noise_level_range[0], self.noise_level_range[1])
                        noise = np.random.normal(0, noise_level, len(aug_audio))
                        aug_audio = aug_audio + noise
                        aug_params["noise_level"] = noise_level
                        aug_params["applied_augmentations"].append("noise")
                    
                    elif aug_type == 'background' and self.background_noise_augmenter:
                        # Ajout de bruit de fond
                        # Pour audiomentations, l'audio doit être un tableau 1D normalisé
                        aug_audio_norm = aug_audio / np.max(np.abs(aug_audio))
                        aug_audio = self.background_noise_augmenter(
                            samples=aug_audio_norm, sample_rate=sample_rate
                        )
                        snr = np.random.uniform(
                            self.background_noise_snr_range[0],
                            self.background_noise_snr_range[1]
                        )
                        aug_params["background_snr"] = snr
                        aug_params["applied_augmentations"].append("background")
                    
                    elif aug_type == 'specaugment':
                        # Appliquer SpecAugment
                        aug_audio = self.apply_spec_augment(aug_audio, sample_rate)
                        aug_params["applied_augmentations"].append("specaugment")
                
                # Normaliser l'audio pour éviter l'écrêtage
                aug_audio = aug_audio / (np.max(np.abs(aug_audio)) + 1e-8)
                
                # Générer un nom de fichier pour l'augmentation
                aug_file = self.output_dir / f"{file_path.stem}_aug{i+1}.wav"
                
                # Sauvegarder l'audio augmenté
                sf.write(aug_file, aug_audio, sample_rate, subtype='PCM_16')
                
                # Enregistrer l'augmentation dans MongoDB
                self.logger.add_augmentation(
                    doc_id=doc_id,
                    original_segment=str(file_path),
                    augmented_file=str(aug_file),
                    augmentation_type="_".join(aug_params["applied_augmentations"]),
                    parameters=aug_params
                )
                
                # Ajouter le chemin et les paramètres à la liste des résultats
                augmented_files.append((str(aug_file), aug_params))
            
            # Mettre à jour le document MongoDB
            augmentation_details = {
                "n_augmentations": len(augmented_files),
                "available_types": available_types,
                "applied_types": selected_types
            }
            self.logger.update_stage(doc_id, "augmented", True, augmentation_details)
            
            logging.info(f"Augmentation terminée: {len(augmented_files)} fichiers créés")
            return augmented_files
            
        except Exception as e:
            logging.error(f"Erreur lors de l'augmentation de {file_path}: {e}")
            self.logger.update_stage(doc_id, "augmented", False, {"error": str(e)})
            return []
    
    def process_directory(self, 
                         dir_path: Union[str, Path], 
                         file_pattern: str = "*.wav",
                         limit: Optional[int] = None) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
        """
        Traite tous les fichiers audio d'un répertoire.
        
        Entrées :
            dir_path (Union[str, Path]) : Chemin du répertoire à traiter
            file_pattern (str) : Motif de fichier à rechercher
            limit (Optional[int]) : Limite du nombre de fichiers à traiter
        
        Sorties :
            Dict[str, List[Tuple[str, Dict[str, Any]]]] : 
                Dictionnaire avec le fichier original comme clé et la liste des fichiers augmentés comme valeur
        """
        dir_path = Path(dir_path)
        logging.info(f"Augmentation du répertoire: {dir_path}")
        
        # Trouver tous les fichiers correspondant au motif
        audio_files = list(dir_path.glob(f"**/{file_pattern}"))
        
        # Limiter le nombre de fichiers si spécifié
        if limit is not None:
            random.shuffle(audio_files)
            audio_files = audio_files[:limit]
        
        results = {}
        for file_path in audio_files:
            augmented_files = self.augment_audio(file_path)
            if augmented_files:
                results[str(file_path)] = augmented_files
        
        logging.info(f"Augmentation terminée: {len(results)} fichiers traités, "
                    f"{sum(len(v) for v in results.values())} augmentations créées")
        return results
