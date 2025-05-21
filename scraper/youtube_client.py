"""
Module client pour l'API YouTube.

Ce module fournit une classe pour interagir avec l'API YouTube Data,
permettant de récupérer des informations sur les playlists et les vidéos.
"""

import os
import logging
import requests
from typing import List, Optional
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper.log'),
        logging.StreamHandler()
    ]
)

class YouTubeClient:
    """
    Client pour interagir avec l'API YouTube Data.
    
    Cette classe fournit des méthodes pour extraire des informations sur les playlists
    et récupérer les URLs des vidéos qu'elles contiennent.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le client YouTube avec une clé API.
        
        Entrées :
            api_key (Optional[str]) : Clé API YouTube (si None, utilise la variable d'environnement YOUTUBE_API_KEY)
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            logging.warning("Aucune clé API YouTube fournie ou trouvée dans les variables d'environnement")
    
    def extract_playlist_id(self, playlist_url: str) -> Optional[str]:
        """
        Extrait l'identifiant d'une playlist YouTube à partir de son URL.
        
        Entrées :
            playlist_url (str) : URL de la playlist YouTube
        Sorties :
            Optional[str] : ID de la playlist (ou None si non trouvé)
        """
        parsed = urlparse(playlist_url)
        query = parse_qs(parsed.query)
        return query.get('list', [None])[0]
    
    def get_videos_from_playlist(self, playlist_id: str) -> List[str]:
        """
        Récupère toutes les URLs des vidéos d'une playlist YouTube via l'API YouTube Data.
        
        Entrées :
            playlist_id (str) : identifiant de la playlist YouTube
        Sorties :
            List[str] : liste des URLs des vidéos de la playlist
        """
        if not self.api_key:
            logging.error("Clé API YouTube requise pour récupérer les vidéos d'une playlist")
            return []
            
        base_url = "https://www.googleapis.com/youtube/v3/playlistItems"
        video_urls = []
        
        params = {
            'part': 'snippet',
            'maxResults': 50,
            'playlistId': playlist_id,
            'key': self.api_key
        }
        
        next_page_token = None
        
        while True:
            if next_page_token:
                params['pageToken'] = next_page_token
                
            try:
                response = requests.get(base_url, params=params)
                data = response.json()
                
                if 'error' in data:
                    logging.error(f"Erreur API YouTube: {data['error']['message']}")
                    break
                    
                for item in data['items']:
                    video_id = item['snippet']['resourceId']['videoId']
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    video_urls.append(video_url)
                
                next_page_token = data.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except Exception as e:
                logging.error(f"Erreur lors de la récupération des vidéos de la playlist: {e}")
                break
                
        logging.info(f"Récupération de {len(video_urls)} vidéos de la playlist {playlist_id}")
        return video_urls
    
    def get_videos_from_playlist_url(self, playlist_url: str) -> List[str]:
        """
        Récupère toutes les URLs des vidéos d'une playlist YouTube à partir de son URL.
        
        Entrées :
            playlist_url (str) : URL de la playlist YouTube
        Sorties :
            List[str] : liste des URLs des vidéos de la playlist
        """
        playlist_id = self.extract_playlist_id(playlist_url)
        if not playlist_id:
            logging.error(f"Impossible d'extraire l'ID de la playlist: {playlist_url}")
            return []
            
        return self.get_videos_from_playlist(playlist_id)
