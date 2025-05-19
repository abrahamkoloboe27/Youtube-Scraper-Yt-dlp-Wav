# Youtube Scraper Yt-dlp Wav 🎵🚀

## 🗂️ Sommaire
- [Présentation](#présentation)
- [Fonctionnement général](#fonctionnement-général)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation & Lancement](#installation--lancement)
- [Utilisation](#utilisation)
- [Variables d'environnement](#variables-denvironnement-à-configurer-env)
- [Astuces et dépannage](#astuces-et-dépannage)
- [Structure du projet](#structure-du-projet)

## Présentation 🎯
Ce projet permet d'extraire automatiquement l'audio (au format WAV) de vidéos YouTube à partir de playlists, de stocker ces fichiers sur MinIO et Azure Blob Storage, et de journaliser les téléchargements dans MongoDB. Il gère aussi la relance automatique des téléchargements échoués.

## Fonctionnement général ⚙️
- **scraper.py** : lit une liste de playlists YouTube (`playlist.txt`), extrait les vidéos, télécharge l'audio de chaque vidéo, téléverse les fichiers audio sur MinIO et Azure, et journalise les succès/échecs dans MongoDB. Le tout en mode parallèle pour accélérer le traitement.
- **retry_failed.py** : relit les téléchargements échoués (stockés dans MongoDB) et tente de les relancer automatiquement.
- **minio_utils.py** : fonctions utilitaires pour interagir avec MinIO (stockage objet).
- **mongo_utils.py** : fonctions utilitaires pour journaliser et récupérer les téléchargements dans MongoDB.
- **azure_stats.py** : permet d'obtenir des statistiques détaillées sur le contenu du conteneur Azure.

## Architecture 🏗️
Le projet s'appuie sur :
- 🐳 **Docker & docker-compose** : orchestration des services (scraper, retry, MongoDB, MinIO).
- 🍃 **MongoDB** : base de données pour journaliser les téléchargements réussis ou échoués.
- 🗄️ **MinIO** : stockage objet compatible S3 pour les fichiers audio.
- ☁️ **Azure Blob Storage** : stockage cloud supplémentaire pour les fichiers audio.

## Prérequis 📝
- [Docker](https://www.docker.com/get-started) et [docker-compose](https://docs.docker.com/compose/) 🐳
- Un compte Azure avec un conteneur Blob Storage (et un SAS Token ou une chaîne de connexion) ☁️
- Une clé API YouTube Data v3 (pour récupérer les vidéos des playlists) 🔑

## Installation & Lancement 🚦
1. **Cloner le dépôt**
```bash
git clone <url_du_repo>
cd Youtube-Scraper-Yt-dlp-Wav
```

2. **Configurer les variables d'environnement**
- Copier `.env.example` en `.env` et compléter avec vos informations (voir section dédiée plus bas).

3. **Ajouter vos playlists**
- Créer un fichier `playlist.txt` (ou compléter celui existant) avec une URL de playlist YouTube par ligne.

4. **Lancer l'ensemble des services**
```bash
docker-compose up --build
```

5. **Consulter les logs**
- Les logs sont disponibles dans le dossier `logs/` pour chaque script (`scraper.log`, `retry.log`, etc.).

## Utilisation 🎬
- ▶️ **Téléchargement automatique** : le service `scraper` traite toutes les playlists listées dans `playlist.txt`.
- ♻️ **Gestion des échecs** : le service `retry_manager` relance automatiquement les téléchargements échoués.
- ➕ **Ajout de playlists** : ajoutez simplement une URL dans `playlist.txt` et relancez le service si besoin.
- 📊 **Statistiques Azure** : lancez `azure_stats.py` pour obtenir un rapport détaillé sur le conteneur Azure.

## Variables d'environnement à configurer (`.env`) 🛠️
```
# MongoDB
MONGO_URI=mongodb://mongodb:27017/
MONGO_DB=scraper_db

# MinIO
MINIO_ENDPOINT=minio:9000
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_BUCKET=audios

# Azure
AZURE_STORAGE_CONNECTION_STRING=... # OU
AZURE_ACCOUNT_URL=... # pour azure_stats.py
AZURE_SAS_TOKEN=...   # pour azure_stats.py
AZURE_CONTAINER_NAME=audios

# YouTube
YOUTUBE_API_KEY=...
```

## Astuces et dépannage 💡
- 📝 **Logs détaillés** : consultez les fichiers dans `logs/` pour diagnostiquer les erreurs.
- ♻️ **Relance des échecs** : les téléchargements échoués sont automatiquement réessayés par `retry_failed.py`.
- 🍪 **Gestion des cookies** : si des erreurs liées à `cookies.txt` apparaissent, vérifiez que le fichier est bien présent et à jour.
- 📊 **Statistiques Azure** : pour obtenir la liste et la taille des fichiers stockés sur Azure, lancez manuellement `azure_stats.py`.

## Structure du projet 🗃️
```
.
├── azure_stats.py         # Statistiques sur le conteneur Azure
├── scraper.py             # Script principal de scraping
├── retry_failed.py        # Relance les téléchargements échoués
├── minio_utils.py         # Fonctions utilitaires MinIO
├── mongo_utils.py         # Fonctions utilitaires MongoDB
├── playlist.txt           # Playlists à traiter
├── logs/                  # Logs des différents scripts
├── audios/                # Fichiers audio téléchargés
├── Dockerfile
├── docker-compose.yml
├── .env.example           # Exemple de configuration
└── README.md
```

---

✨ **Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue ou à contribuer !** 🚀
