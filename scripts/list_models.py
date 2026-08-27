# scripts/list_models.py
import os
from dotenv import load_dotenv
load_dotenv()

from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
for m in client.models.list():
    if "generateContent" in getattr(m, "supported_actions", []) or True:
        print(f"{m.name:<45} {getattr(m, 'display_name', '')}")