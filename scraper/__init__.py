"""
Module de scraping et téléchargement de vidéos YouTube.

Ce module permet de récupérer des vidéos à partir de playlists YouTube
et de télécharger leur contenu audio.

Classes principales :
- YouTubeClient : Gère l'interaction avec l'API YouTube
- VideoDownloader : Télécharge l'audio des vidéos YouTube
- PlaylistScraper : Récupère et traite des playlists YouTube complètes
"""

from .youtube_client import YouTubeClient
from .video_downloader import VideoDownloader
from .playlist_scraper import PlaylistScraper

__all__ = [
    'YouTubeClient',
    'VideoDownloader',
    'PlaylistScraper'
]
