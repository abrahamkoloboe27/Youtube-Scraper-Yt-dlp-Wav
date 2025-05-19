FROM python:3.11-slim

# Fixe la timezone à Paris (adapter si besoin)
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Fixe la locale à fr_FR.UTF-8
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    ffmpeg \
    locales && \
    sed -i '/fr_FR.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen && \
    rm -rf /var/lib/apt/lists/*
ENV LANG=fr_FR.UTF-8
ENV LANGUAGE=fr_FR:fr
ENV LC_ALL=fr_FR.UTF-8

# Installation de Chrome
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Création du répertoire de travail
WORKDIR /app

# Copie des fichiers de dépendances
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Installation des navigateurs Playwright
RUN playwright install chromium
RUN playwright install-deps

# Copie du code source
COPY . .

# Création du dossier de téléchargement
RUN mkdir -p downloads

# Commande par défaut
CMD ["python", "main.py"] 