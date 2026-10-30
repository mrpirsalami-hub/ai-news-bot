#!/usr/bin/env python3
"""Daily AI news -> Persian summary (Gemini) -> Telegram channel.

Uses only the Python standard library. Configuration comes from environment
variables (set as GitHub Actions secrets):
  GEMINI_API_KEY      key from Google AI Studio
  TELEGRAM_BOT_TOKEN  token from @BotFather
  TELEGRAM_CHAT_ID    channel username like @mychannel (bot must be an admin)
  GEMINI_MODEL        optional, defaults to gemini-2.5-flash
"""
import html
import json
import os
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

FEEDS = [
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "https://venturebeat.com/category/ai/feed/",
]
WINDOW_HOURS = 30        # only articles newer than this
MAX_CANDIDATES = 12      # sent to the model
POSTED_FILE = Path(__file__).parent / "posted.json"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
USER_AGENT = "Mozilla/5.0 (compatible; ai-news-bot/1.0)"


def local(tag):
    return tag.split("}")[-1]


def clean(text, limit=None):
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    return text[:limit] if limit else text


def parse_date(text):
    if not text:
        return None
    text = text.strip()
    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_feed(xml_bytes):
    """Parse RSS 2.0 or Atom into a list of dicts."""
    root = ET.fromstring(xml_bytes)
    articles = []
    for el in root.iter():
        if local(el.tag) not in ("item", "entry"):
            continue
        title = link = snippet = date_text = ""
        for child in el:
            name = local(child.tag)
            if name == "title":
                title = clean(child.text)
            elif name == "link":
                href = child.get("href") or (child.text or "").strip()
                if href and (not link or child.get("rel") in (None, "alternate")):
                    link = href
            elif name in ("description", "summary", "content") and not snippet:
                snippet = clean(child.text, 300)
            elif name in ("pubDate", "published", "updated", "date") and not date_text:
                date_text = child.text or ""
        if title and link:
            articles.append(
                {"title": title, "link": link, "snippet": snippet,
                 "date": parse_date(date_text)}
            )
    return articles


def fetch(url, data=None, headers=None, timeout=30):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    req.add_header("User-Agent", USER_AGENT)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def load_posted():
    try:
        return json.loads(POSTED_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return []


def collect_candidates(posted):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)
    seen = set(posted)
    fresh = []
    for url in FEEDS:
        try:
            articles = parse_feed(fetch(url))
        except Exception as exc:  # one broken feed must not stop the run
            print(f"WARN: feed failed {url}: {exc}", file=sys.stderr)
            continue
        for a in articles:
            if a["link"] in seen or a["date"] is None or a["date"] < cutoff:
                continue
            seen.add(a["link"])
            fresh.append(a)
    fresh.sort(key=lambda a: a["date"], reverse=True)
    return fresh[:MAX_CANDIDATES]


def build_prompt(candidates):
    listing = "\n\n".join(
        f"{i}. {a['title']}\n{a['snippet']}\n{a['link']}"
        for i, a in enumerate(candidates, 1)
    )
    return (
        "You are the editor of a Persian-language Telegram channel about AI news.\n"
        "From the articles below, pick the 5 most important and interesting ones "
        "for Persian-speaking creators and AI enthusiasts.\n\n"
        "Write the post in natural, friendly Persian (Farsi). Rules:\n"
        "- Start with one line: 🤖 <b>اخبار هوش مصنوعی امروز</b>\n"
        "- For each chosen article: an emoji, a short bold title in Persian, "
        "then 1-2 sentences of summary, then the link.\n"
        '- Use only Telegram HTML tags (<b>, <i>, <a href="...">). No markdown.\n'
        "- Do not invent facts that are not in the article text.\n"
        "- Keep the whole post under 3500 characters.\n"
        "- Output only the post text.\n\n"
        f"Articles:\n\n{listing}"
    )


def ask_gemini(api_key, prompt):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    raw = fetch(
        url,
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        timeout=90,
    )
    data = json.loads(raw)
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response: {str(data)[:500]}")
    # remove accidental ``` fences
    text = re.sub(r"^```(?:html)?\s*|\s*```$", "", text).strip()
    return text[:4000]


def to_plain(text):
    text = re.sub(r'<a\s+href="([^"]+)"[^>]*>(.*?)</a>', r"\2 (\1)", text, flags=re.S)
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    def call(payload):
        fetch(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

    base = {"chat_id": chat_id, "disable_web_page_preview": True}
    try:
        call({**base, "text": text, "parse_mode": "HTML"})
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            raise
        # Model produced HTML Telegram rejects -> send as plain text.
        call({**base, "text": to_plain(text)[:4000]})


def main():
    try:
        gemini_key = os.environ["GEMINI_API_KEY"]
        token = os.environ["TELEGRAM_BOT_TOKEN"]
        chat_id = os.environ["TELEGRAM_CHAT_ID"]
    except KeyError as exc:
        sys.exit(f"Missing environment variable: {exc}")

    posted = load_posted()
    candidates = collect_candidates(posted)
    if not candidates:
        print("No new articles; nothing to post.")
        return

    post = ask_gemini(gemini_key, build_prompt(candidates))
    send_telegram(token, chat_id, post)

    posted.extend(a["link"] for a in candidates)
    POSTED_FILE.write_text(json.dumps(posted[-300:], indent=2))
    print(f"Posted {len(candidates)} candidate articles' summary.")


if __name__ == "__main__":
    main()
