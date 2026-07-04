#!/usr/bin/env python3
from pathlib import Path
from urllib.parse import urljoin
import time
import json
import hashlib
import requests
import http.cookiejar
from bs4 import BeautifulSoup

BASE = Path.home() / "Documents/ToribashAI"
SCRAPER = BASE / "scraper"
COOKIES = SCRAPER / "cookies.txt"

REPLAYS = BASE / "replays_raw" / "parkour"
MODS = BASE / "mods_raw"
META = BASE / "metadata"
LOG = META / "toribash_downloads.jsonl"

THREAD_URL = "https://forum.toribash.com/showthread.php?t=647301"
PAGES = 6

for folder in [SCRAPER, REPLAYS, MODS, META]:
    folder.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 ToribashAI research scraper"
})

cookiejar = http.cookiejar.MozillaCookieJar(str(COOKIES))
cookiejar.load(ignore_discard=True, ignore_expires=True)
session.cookies = cookiejar


def safe_name(name):
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ .()[]"
    cleaned = "".join(c if c in allowed else "_" for c in name)
    return cleaned[:180] or "unknown_file"


def already_downloaded(url):
    if not LOG.exists():
        return False
    return url in LOG.read_text(errors="ignore")


def save_metadata(data):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def extract_filename(label, href):
    label = label.strip()
    href_name = href.split("/")[-1].split("?")[0]

    if label.lower().endswith((".rpl", ".tbm")):
        return label

    if href_name.lower().endswith((".rpl", ".tbm")):
        return href_name

    return label or href_name or "unknown_file"


def download_attachment(file_url, filename, source_page):
    if already_downloaded(file_url):
        print("Déjà téléchargé:", filename)
        return

    lower = filename.lower()

    if lower.endswith(".rpl"):
        folder = REPLAYS
    elif lower.endswith(".tbm"):
        folder = MODS
    else:
        return

    print("Téléchargement:", filename)

    r = session.get(file_url, timeout=60)
    r.raise_for_status()

    content = r.content

    # Sécurité : si on récupère une page HTML au lieu du fichier, on évite de la sauver comme replay
    if content[:100].lower().startswith(b"<!doctype html") or b"<html" in content[:500].lower():
        print("Ignoré: le téléchargement semble être une page HTML:", filename)
        return

    sha = hashlib.sha256(content).hexdigest()
    short_sha = sha[:12]

    final_name = f"{short_sha}_{safe_name(filename)}"
    path = folder / final_name

    path.write_bytes(content)

    save_metadata({
        "final_filename": final_name,
        "original_filename": filename,
        "source_page": source_page,
        "file_url": file_url,
        "sha256": sha,
        "size_bytes": len(content),
        "category_guess": "parkour",
    })

    print("Sauvé:", path)
    time.sleep(2)


def scrape_page(page_url):
    print("\nPage:", page_url)

    r = session.get(page_url, timeout=60)
    r.raise_for_status()

    if "Log in" in r.text:
        print("ATTENTION: la page contient 'Log in'. Les cookies sont peut-être expirés.")

    soup = BeautifulSoup(r.text, "html.parser")

    found = 0

    for a in soup.find_all("a", href=True):
        label = a.get_text(" ", strip=True)
        href = a["href"]
        combined = f"{label} {href}".lower()

        if ".rpl" not in combined and ".tbm" not in combined:
            continue

        file_url = urljoin(page_url, href)
        filename = extract_filename(label, href)

        if not filename.lower().endswith((".rpl", ".tbm")):
            continue

        found += 1
        download_attachment(file_url, filename, page_url)

    print("Fichiers trouvés sur cette page:", found)


def main():
    for page in range(1, PAGES + 1):
        if page == 1:
            url = THREAD_URL
        else:
            url = f"{THREAD_URL}&page={page}"

        scrape_page(url)
        time.sleep(5)

    print("\nTerminé.")
    print("Replays:", REPLAYS)
    print("Mods:", MODS)
    print("Metadata:", LOG)


if __name__ == "__main__":
    main()
