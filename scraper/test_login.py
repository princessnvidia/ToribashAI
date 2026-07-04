import requests
import http.cookiejar

session = requests.Session()

cj = http.cookiejar.MozillaCookieJar(
    "/home/vio/Documents/ToribashAI/scraper/cookies.txt"
)

cj.load(ignore_discard=True, ignore_expires=True)
session.cookies = cj

r = session.get(
    "https://forum.toribash.com/showthread.php?t=647301"
)

print("Status:", r.status_code)
print("Contains 'Log in':", "Log in" in r.text)
