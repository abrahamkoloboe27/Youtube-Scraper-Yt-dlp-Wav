"""
Module de vérification finale et exports des fichiers audio.

Ce module permet de vérifier la qualité des fichiers audio traités,
d'exclure les segments problématiques et d'exporter les fichiers validés.

Classes principales :
- QualityChecker : Classe gérant la vérification finale et les exports
"""

import os
import logging
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Set
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import jiwer
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

class QualityChecker:
    """
    Classe pour la vérification finale et les exports des fichiers audio.
    
    Cette classe permet de vérifier la qualité des fichiers audio traités,
    d'exclure les segments problématiques et d'exporter les fichiers validés.
    """
    
    def __init__(self, 
                 output_dir: Union[str, Path] = "processed/final",
                 min_snr: float = 15.0,
                 min_duration: float = 1.0,
                 max_duration: float = 20.0,
                 random_sample_size: int = 10,
                 generate_plots: bool = True):
        """
        Initialise le vérificateur de qualité avec les paramètres spécifiés.
        
        Entrées :
            output_dir (Union[str, Path]) : Répertoire de sortie pour les fichiers validés
            min_snr (float) : SNR minimum requis pour valider un segment
            min_duration (float) : Durée minimale requise pour valider un segment
            max_duration (float) : Durée maximale autorisée pour un segment
            random_sample_size (int) : Nombre de segments à échantillonner pour la vérification manuelle
            generate_plots (bool) : Si True, génère des graphiques pour l'échantillon aléatoire
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "quality_plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.min_snr = min_snr
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.random_sample_size = random_sample_size
        self.generate_plots = generate_plots
        self.logger = MongoLogger()
        
        # Métriques de qualité
        self.quality_metrics = {
            'total_segments': 0,
            'valid_segments': 0,
            'invalid_segments': 0,
            'reasons_for_rejection': {},
            'avg_snr': 0.0,
            'min_snr_observed': float('inf'),
            'max_snr_observed': float('-inf'),
            'avg_duration': 0.0
        }
    
    def check_segment(self, 
                     file_path: Union[str, Path],
                     metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Vérifie si un segment audio répond aux critères de qualité.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier audio
            metadata (Optional[Dict[str, Any]]) : Métadonnées du segment si disponibles
        
        Sorties :
            Tuple[bool, Dict[str, Any]] : 
                - True si le segment est valide, False sinon
                - Métriques de qualité du segment
        """
        file_path = Path(file_path)
        metrics = {}
        
        try:
            # Charger le fichier audio
            audio, sample_rate = librosa.load(file_path, sr=None)
            
            # Mesurer la durée
            duration = len(audio) / sample_rate
            metrics['duration'] = duration
            
            # Estimer le SNR
            signal_power = np.mean(audio ** 2)
            
            # Estimer le niveau de bruit (en utilisant les 100ms les plus calmes)
            frame_length = min(int(0.1 * sample_rate), len(audio) // 2)
            if frame_length > 0:
                windowed = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length // 2)
                if windowed.size > 0:
                    frame_powers = np.mean(windowed ** 2, axis=0)
                    noise_power = np.min(frame_powers)
                else:
                    noise_power = signal_power * 0.1
            else:
                noise_power = signal_power * 0.1
            
            # Calculer le SNR
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = 100.0  # Valeur arbitrairement élevée
            metrics['snr'] = snr
            
            # Vérifier si le segment est valide
            is_valid = True
            rejection_reasons = []
            
            # Vérification de la durée
            if duration < self.min_duration:
                is_valid = False
                rejection_reasons.append('too_short')
            elif duration > self.max_duration:
                is_valid = False
                rejection_reasons.append('too_long')
            
            # Vérification du SNR
            if snr < self.min_snr:
                is_valid = False
                rejection_reasons.append('low_snr')
            
            # Autres vérifications (à partir des métadonnées si disponibles)
            if metadata:
                # Exemple : vérifier si le segment est marqué comme problématique
                if metadata.get('is_problematic', False):
                    is_valid = False
                    rejection_reasons.append('marked_problematic')
            
            metrics['is_valid'] = is_valid
            metrics['rejection_reasons'] = rejection_reasons
            
            return is_valid, metrics
            
        except Exception as e:
            logging.error(f"Erreur lors de la vérification de {file_path}: {e}")
            return False, {'error': str(e), 'is_valid': False, 'rejection_reasons': ['error_during_check']}
    
    def generate_quality_plot(self, 
                             file_path: Union[str, Path], 
                             metrics: Dict[str, Any]) -> Optional[str]:
        """
        Génère un graphique de qualité pour un segment audio.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier audio
            metrics (Dict[str, Any]) : Métriques de qualité du segment
        
        Sorties :
            Optional[str] : Chemin du graphique généré, ou None en cas d'échec
        """
        if not self.generate_plots:
            return None
        
        file_path = Path(file_path)
        
        try:
            # Charger le fichier audio
            y, sr = librosa.load(file_path, sr=None)
            
            # Créer une figure avec plusieurs sous-graphiques
            plt.figure(figsize=(12, 10))
            
            # 1. Forme d'onde
            plt.subplot(3, 1, 1)
            librosa.display.waveshow(y, sr=sr)
            plt.title(f"Forme d'onde - {file_path.name}")
            plt.ylabel("Amplitude")
            
            # 2. Spectrogramme
            plt.subplot(3, 1, 2)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title("Spectrogramme")
            
            # 3. MFCC (coefficients cepstraux)
            plt.subplot(3, 1, 3)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            librosa.display.specshow(mfcc, x_axis='time')
            plt.colorbar()
            plt.title("MFCC")
            
            # Ajouter des informations de métriques
            plt.figtext(0.5, 0.01, 
                       f"Duration: {metrics['duration']:.2f}s | SNR: {metrics.get('snr', 'N/A'):.2f} dB | "
                       f"Valid: {metrics['is_valid']}", 
                       ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
            
            # Sauvegarder le graphique
            plot_path = self.plots_dir / f"{file_path.stem}_quality.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            logging.error(f"Erreur lors de la génération du graphique pour {file_path}: {e}")
            return None
    
    def calculate_wer(self, 
                     hypothesis: str, 
                     reference: str) -> Dict[str, float]:
        """
        Calcule le Word Error Rate (WER) et le Character Error Rate (CER).
        
        Entrées :
            hypothesis (str) : Texte hypothèse (transcription automatique)
            reference (str) : Texte de référence (transcription manuelle)
        
        Sorties :
            Dict[str, float] : Dictionnaire des métriques WER et CER
        """
        # Word Error Rate
        wer = jiwer.wer(reference, hypothesis)
        
        # Character Error Rate
        cer = jiwer.cer(reference, hypothesis)
        
        return {
            'wer': wer,
            'cer': cer
        }
    
    def process_file(self, 
                    file_path: Union[str, Path], 
                    metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Vérifie et exporte un fichier audio s'il est valide.
        
        Entrées :
            file_path (Union[str, Path]) : Chemin du fichier audio
            metadata (Optional[Dict[str, Any]]) : Métadonnées du segment si disponibles
        
        Sorties :
            Optional[str] : Chemin du fichier exporté, ou None si invalide
        """
        file_path = Path(file_path)
        logging.info(f"Vérification du fichier: {file_path}")
        
        # Vérifier le segment
        is_valid, metrics = self.check_segment(file_path, metadata)
        
        # Mettre à jour les métriques globales
        self.quality_metrics['total_segments'] += 1
        if is_valid:
            self.quality_metrics['valid_segments'] += 1
        else:
            self.quality_metrics['invalid_segments'] += 1
            for reason in metrics.get('rejection_reasons', []):
                if reason in self.quality_metrics['reasons_for_rejection']:
                    self.quality_metrics['reasons_for_rejection'][reason] += 1
                else:
                    self.quality_metrics['reasons_for_rejection'][reason] = 1
        
        # Mettre à jour les statistiques SNR
        if 'snr' in metrics:
            # Mise à jour cumulative de la moyenne du SNR
            n = self.quality_metrics['valid_segments'] + self.quality_metrics['invalid_segments']
            self.quality_metrics['avg_snr'] = ((n - 1) * self.quality_metrics['avg_snr'] + metrics['snr']) / n
            
            # Mise à jour min/max
            self.quality_metrics['min_snr_observed'] = min(self.quality_metrics['min_snr_observed'], metrics['snr'])
            self.quality_metrics['max_snr_observed'] = max(self.quality_metrics['max_snr_observed'], metrics['snr'])
        
        # Mise à jour de la durée moyenne
        if 'duration' in metrics:
            n = self.quality_metrics['valid_segments'] + self.quality_metrics['invalid_segments']
            self.quality_metrics['avg_duration'] = ((n - 1) * self.quality_metrics['avg_duration'] + metrics['duration']) / n
        
        # Récupérer le document MongoDB pour ce fichier
        mongo_doc = None
        segments = self.logger.collection.find(
            {"segments.file": file_path.name},
        )
        
        for segment_doc in segments:
            if segment_doc:
                mongo_doc = segment_doc
                break
        
        if not mongo_doc:
            logging.warning(f"Document MongoDB non trouvé pour {file_path}")
        else:
            doc_id = str(mongo_doc["_id"])
            
            # Générer un graphique si requis
            plot_path = None
            if self.generate_plots and (is_valid or random.random() < 0.2):  # Générer pour tous les valides et 20% des invalides
                plot_path = self.generate_quality_plot(file_path, metrics)
            
            # Exporter le fichier si valide
            output_path = None
            if is_valid:
                output_path = self.output_dir / file_path.name
                if file_path != output_path:  # Éviter de copier si c'est le même chemin
                    try:
                        # Copier le fichier vers le répertoire de sortie
                        audio, sr = librosa.load(file_path, sr=None)
                        sf.write(output_path, audio, sr, subtype='PCM_16')
                    except Exception as e:
                        logging.error(f"Erreur lors de la copie de {file_path} vers {output_path}: {e}")
                        output_path = None
            
            # Mettre à jour MongoDB
            quality_details = {
                "is_valid": is_valid,
                "metrics": metrics,
                "output_path": str(output_path) if output_path else None,
                "plot_path": plot_path
            }
            self.logger.update_stage(doc_id, "quality_checked", True, quality_details)
            
            if is_valid and output_path:
                self.logger.update_stage(doc_id, "exported", True, {"final_path": str(output_path)})
                logging.info(f"Fichier validé et exporté: {output_path}")
                return str(output_path)
            else:
                reason = ", ".join(metrics.get('rejection_reasons', ['unknown']))
                logging.info(f"Fichier rejeté ({reason}): {file_path}")
                return None
        
        return None
    
    def process_directory(self, 
                         dir_path: Union[str, Path],
                         metadata_file: Optional[Union[str, Path]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Vérifie tous les fichiers audio d'un répertoire et exporte ceux qui sont valides.
        
        Entrées :
            dir_path (Union[str, Path]) : Chemin du répertoire à traiter
            metadata_file (Optional[Union[str, Path]]) : Chemin du fichier de métadonnées (CSV ou Parquet)
        
        Sorties :
            Tuple[List[str], Dict[str, Any]] : 
                - Liste des chemins des fichiers exportés
                - Métriques de qualité globales
        """
        dir_path = Path(dir_path)
        logging.info(f"Vérification du répertoire: {dir_path}")
        
        # Charger les métadonnées si disponibles
        metadata_dict = {}
        if metadata_file:
            try:
                metadata_path = Path(metadata_file)
                if metadata_path.suffix.lower() == '.csv':
                    df = pd.read_csv(metadata_path)
                elif metadata_path.suffix.lower() == '.parquet':
                    df = pd.read_parquet(metadata_path)
                else:
                    logging.warning(f"Format de métadonnées non pris en charge: {metadata_path.suffix}")
                    df = None
                
                if df is not None:
                    # Créer un dictionnaire de métadonnées par fichier
                    if 'segment_file' in df.columns:
                        metadata_dict = {row['segment_file']: row.to_dict() for _, row in df.iterrows()}
            except Exception as e:
                logging.error(f"Erreur lors du chargement des métadonnées: {e}")
        
        # Trouver tous les fichiers WAV dans le répertoire
        wav_files = list(dir_path.glob("**/*.wav"))
        
        # Traiter chaque fichier
        exported_files = []
        for file_path in wav_files:
            # Récupérer les métadonnées si disponibles
            metadata = metadata_dict.get(file_path.name)
            
            # Vérifier et exporter le fichier
            output_path = self.process_file(file_path, metadata)
            if output_path:
                exported_files.append(output_path)
        
        # Sélectionner un échantillon aléatoire pour vérification manuelle
        if self.random_sample_size > 0 and exported_files:
            sample_size = min(self.random_sample_size, len(exported_files))
            random_sample = random.sample(exported_files, sample_size)
            
            # Générer un rapport d'échantillon
            sample_report_path = self.output_dir / "random_sample_report.txt"
            with open(sample_report_path, 'w') as f:
                f.write(f"Échantillon aléatoire pour vérification manuelle ({sample_size} fichiers)\n")
                f.write("="*80 + "\n\n")
                for i, path in enumerate(random_sample, 1):
                    f.write(f"{i}. {path}\n")
            
            logging.info(f"Échantillon aléatoire pour vérification manuelle: {sample_report_path}")
        
        # Générer un rapport de qualité global
        report_path = self.output_dir / "quality_report.txt"
        with open(report_path, 'w') as f:
            f.write("Rapport de qualité du traitement audio\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total des segments traités: {self.quality_metrics['total_segments']}\n")
            f.write(f"Segments valides: {self.quality_metrics['valid_segments']} "
                   f"({self.quality_metrics['valid_segments']/max(1, self.quality_metrics['total_segments'])*100:.2f}%)\n")
            f.write(f"Segments invalides: {self.quality_metrics['invalid_segments']} "
                   f"({self.quality_metrics['invalid_segments']/max(1, self.quality_metrics['total_segments'])*100:.2f}%)\n\n")
            
            f.write("Raisons de rejet:\n")
            for reason, count in self.quality_metrics['reasons_for_rejection'].items():
                f.write(f"  - {reason}: {count} segments "
                       f"({count/max(1, self.quality_metrics['invalid_segments'])*100:.2f}% des rejets)\n")
            
            f.write(f"\nSNR moyen: {self.quality_metrics['avg_snr']:.2f} dB\n")
            f.write(f"SNR minimum observé: {self.quality_metrics['min_snr_observed']:.2f} dB\n")
            f.write(f"SNR maximum observé: {self.quality_metrics['max_snr_observed']:.2f} dB\n")
            f.write(f"Durée moyenne: {self.quality_metrics['avg_duration']:.2f} s\n")
        
        logging.info(f"Vérification terminée: {len(exported_files)}/{len(wav_files)} fichiers validés et exportés")
        logging.info(f"Rapport de qualité: {report_path}")
        
        return exported_files, self.quality_metrics
