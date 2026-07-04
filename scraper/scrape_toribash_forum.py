#!/usr/bin/env python3
from pathlib import Path
from urllib.parse import urljoin
import time
import hashlib
import requests
from bs4 import BeautifulSoup

BASE = Path.home() / "Documents/ToribashAI"
OUT_REPLAYS = BASE / "replays_raw/parkour"
OUT_MODS = BASE / "mods_raw"
META = BASE / "metadata/downloaded_urls.txt"

OUT_REPLAYS.mkdir(parents=True, exist_ok=True)
OUT_MODS.mkdir(parents=True, exist_ok=True)
META.parent.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "ToribashAI research scraper by Vio - slow archival downloader"
}

THREAD_URLS = [
    # On mettra ici les URLs de threads trouvés à la main ou par recherche
    "https://forum.toribash.com/showthread.php?t=353409",
    "https://forum.toribash.com/showthread.php?t=571231",
]

def load_seen():
    if not META.exists():
        return set()
    return set(META.read_text().splitlines())

def save_seen(url):
    with META.open("a") as f:
        f.write(url + "\n")

def safe_filename(name):
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. #[]()"
    cleaned = "".join(c if c in keep else "_" for c in name)
    return cleaned[:180]

def download_file(url, folder, filename):
    seen = load_seen()
    if url in seen:
        print("Déjà téléchargé:", url)
        return

    print("Download:", url)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()

    file_hash = hashlib.sha256(r.content).hexdigest()[:12]
    filename = safe_filename(filename)
    path = folder / f"{file_hash}_{filename}"

    path.write_bytes(r.content)
    save_seen(url)
    print("Sauvé:", path)

    time.sleep(2)

def scrape_thread(thread_url):
    print("Thread:", thread_url)
    r = requests.get(thread_url, headers=HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        href = a["href"]

        if ".rpl" in text.lower() or ".rpl" in href.lower():
            url = urljoin(thread_url, href)
            filename = text if text.lower().endswith(".rpl") else "replay.rpl"
            download_file(url, OUT_REPLAYS, filename)

        elif ".tbm" in text.lower() or ".tbm" in href.lower():
            url = urljoin(thread_url, href)
            filename = text if text.lower().endswith(".tbm") else "mod.tbm"
            download_file(url, OUT_MODS, filename)

def main():
    for url in THREAD_URLS:
        try:
            scrape_thread(url)
            time.sleep(5)
        except Exception as e:
            print("Erreur:", url, e)

if __name__ == "__main__":
    main()
