#!/usr/bin/env python3
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs
import time
import json
import hashlib
import requests
import http.cookiejar
from bs4 import BeautifulSoup

BASE = Path.home() / "Documents/ToribashAI"
SCRAPER = BASE / "scraper"
COOKIES = SCRAPER / "cookies.txt"

REPLAYS = BASE / "replays_raw" / "parkour_candidate"
MODS = BASE / "mods_raw"
META = BASE / "metadata"

LOG = META / "toribash_search_downloads.jsonl"
THREADS_LOG = META / "toribash_threads_seen.jsonl"
DONE_THREADS = META / "toribash_threads_done.txt"

SEARCH_BASE_URL = "https://forum.toribash.com/search.php?searchid=142214"
SEARCH_PAGES = 16

DELAY_FILE = 1
DELAY_PAGE = 1
DELAY_THREAD = 3
DELAY_SEARCH_PAGE = 2

for folder in [SCRAPER, REPLAYS, MODS, META]:
    folder.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 ToribashAI research scraper - slow dataset collector"
})

cookiejar = http.cookiejar.MozillaCookieJar(str(COOKIES))
cookiejar.load(ignore_discard=True, ignore_expires=True)
session.cookies = cookiejar


def make_search_page_url(page):
    if page <= 1:
        return SEARCH_BASE_URL
    return f"{SEARCH_BASE_URL}&page={page}"


def safe_name(name):
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ .()[]"
    return "".join(c if c in allowed else "_" for c in name)[:180] or "unknown_file"


def canonical_thread_url(url):
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    tid = qs.get("t", [None])[0]
    if not tid:
        return None
    return f"https://forum.toribash.com/showthread.php?t={tid}"


def get_thread_id(url):
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    return qs.get("t", [None])[0]


def thread_page_url(thread_url, page):
    if page <= 1:
        return thread_url
    return f"{thread_url}&page={page}"


def load_text_set(path):
    if not path.exists():
        return set()
    return set(line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip())


def mark_done(thread_url):
    with DONE_THREADS.open("a", encoding="utf-8") as f:
        f.write(thread_url + "\n")


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

    if "." in label:
        return label

    return href_name or label or "unknown_file"


def looks_like_html(content):
    head = content[:1000].lower()
    return b"<html" in head or b"<!doctype html" in head


def download_attachment(file_url, filename, source_page, thread_url, thread_title):
    lower = filename.lower()

    if not lower.endswith((".rpl", ".tbm")):
        return False

    if already_downloaded(file_url):
        print("    Déjà téléchargé:", filename)
        return False

    folder = REPLAYS if lower.endswith(".rpl") else MODS

    print("    Téléchargement:", filename)

    r = session.get(file_url, timeout=60)
    r.raise_for_status()

    content = r.content

    if looks_like_html(content):
        print("    Ignoré HTML:", filename)
        return False

    sha = hashlib.sha256(content).hexdigest()
    final_name = f"{sha[:12]}_{safe_name(filename)}"
    path = folder / final_name

    if path.exists():
        print("    Déjà présent:", path.name)
        return False

    path.write_bytes(content)

    save_metadata({
        "final_filename": final_name,
        "original_filename": filename,
        "source_page": source_page,
        "thread_url": thread_url,
        "thread_title": thread_title,
        "file_url": file_url,
        "sha256": sha,
        "size_bytes": len(content),
        "category_guess": "parkour_candidate",
    })

    print("    Sauvé:", path)
    time.sleep(DELAY_FILE)
    return True


def scrape_thread_page(url, thread_url, thread_title):
    print("  Page thread:", url)

    r = session.get(url, timeout=60)
    r.raise_for_status()

    if "Log in" in r.text:
        print("    ATTENTION: cookies possiblement expirés.")

    soup = BeautifulSoup(r.text, "html.parser")
    saved = 0
    seen_links = set()

    for a in soup.find_all("a", href=True):
        label = a.get_text(" ", strip=True)
        href = a["href"]
        combined = f"{label} {href}".lower()

        if ".rpl" not in combined and ".tbm" not in combined and "attachment.php" not in combined:
            continue

        filename = extract_filename(label, href)

        if not filename.lower().endswith((".rpl", ".tbm")):
            continue

        file_url = urljoin(url, href)

        if file_url in seen_links:
            continue

        seen_links.add(file_url)

        if download_attachment(file_url, filename, url, thread_url, thread_title):
            saved += 1

    print("    Fichiers sauvés sur cette page:", saved)
    return saved


def extract_all_threads():
    threads = {}

    for search_page in range(1, SEARCH_PAGES + 1):
        search_url = make_search_page_url(search_page)
        print("Lecture recherche:", search_url)

        r = session.get(search_url, timeout=60)
        r.raise_for_status()

        if "Log in" in r.text:
            print("ATTENTION: cookies possiblement expirés.")

        soup = BeautifulSoup(r.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]

            if "showthread.php" not in href:
                continue

            full = urljoin(search_url, href)
            thread_url = canonical_thread_url(full)

            if not thread_url:
                continue

            title = a.get_text(" ", strip=True)

            if not title or title.isdigit() or "»" in title or title == "#":
                continue

            parsed = urlparse(full)
            qs = parse_qs(parsed.query)
            page_nums = [1]

            if "page" in qs:
                try:
                    page_nums.append(int(qs["page"][0]))
                except ValueError:
                    pass

            old = threads.get(thread_url)

            if old:
                old["last_page"] = max(old["last_page"], max(page_nums))
            else:
                threads[thread_url] = {
                    "url": thread_url,
                    "title": title,
                    "last_page": max(page_nums),
                }

        print("  Threads cumulés:", len(threads))
        time.sleep(DELAY_SEARCH_PAGE)

    with THREADS_LOG.open("w", encoding="utf-8") as f:
        for data in threads.values():
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    return list(threads.values())


def main():
    done_threads = load_text_set(DONE_THREADS)
    threads = extract_all_threads()

    print("\nThreads uniques trouvés:", len(threads))
    print("Threads déjà terminés:", len(done_threads))
    print("Liste sauvée dans:", THREADS_LOG)

    total_saved = 0

    for index, thread in enumerate(threads, start=1):
        thread_url = thread["url"]
        title = thread["title"]
        last_page = thread["last_page"]

        if thread_url in done_threads:
            print(f"\nThread {index}/{len(threads)} déjà terminé, skip:", title)
            continue

        print(f"\nThread {index}/{len(threads)}:", title)
        print(thread_url)
        print("Dernière page détectée:", last_page)

        try:
            for page in range(1, last_page + 1):
                url = thread_page_url(thread_url, page)
                total_saved += scrape_thread_page(url, thread_url, title)
                time.sleep(DELAY_PAGE)

            mark_done(thread_url)
            done_threads.add(thread_url)

        except KeyboardInterrupt:
            print("\nInterruption demandée. Progression gardée.")
            break

        except Exception as e:
            print("  Erreur thread:", thread_url, e)

        time.sleep(DELAY_THREAD)

    print("\nTerminé ou interrompu proprement.")
    print("Total fichiers sauvés cette session:", total_saved)
    print("Replays candidats:", REPLAYS)
    print("Mods:", MODS)
    print("Metadata:", LOG)
    print("Threads vus:", THREADS_LOG)
    print("Threads terminés:", DONE_THREADS)


if __name__ == "__main__":
    main()
