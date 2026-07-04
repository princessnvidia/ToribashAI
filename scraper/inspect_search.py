#!/usr/bin/env python3
from pathlib import Path
from urllib.parse import urljoin
import requests
import http.cookiejar
from bs4 import BeautifulSoup

COOKIES = Path.home() / "Documents/ToribashAI/scraper/cookies.txt"
SEARCH_URL = "https://forum.toribash.com/search.php?searchid=142214"

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 ToribashAI scraper test"})

cj = http.cookiejar.MozillaCookieJar(str(COOKIES))
cj.load(ignore_discard=True, ignore_expires=True)
session.cookies = cj

r = session.get(SEARCH_URL, timeout=60)
r.raise_for_status()

soup = BeautifulSoup(r.text, "html.parser")

print("Status:", r.status_code)
print("Log in:", "Log in" in r.text)
print("paperclip.png count:", r.text.lower().count("paperclip"))

print("\nLiens suspects:")
for a in soup.find_all("a", href=True):
    label = a.get_text(" ", strip=True)
    href = a["href"]
    combined = (label + " " + href).lower()

    if "attachment" in combined or ".rpl" in combined or ".tbm" in combined or "showthread.php" in href:
        print(label[:80], "=>", urljoin(SEARCH_URL, href))
