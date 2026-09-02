import os
import requests

def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome."""
    cleaned = _clean_string(s)
    return cleaned == cleaned[::-1]

def _clean_string(s: str) -> str:
    """Helper: lowercase and strip spaces."""
    return s.lower().replace(" ", "")

def send_analytics_event(event_name: str):
    """Send a usage analytics event."""
    print(f"Tracking: {event_name}")

def fetch_url(url):
    return requests.fetch_url(url)
