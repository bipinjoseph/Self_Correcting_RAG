import os
from google.genai import client
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

cl = client.Client(api_key=api_key)

models = cl.models.list()
for m in models:
    print(m.name, "| supported actions:", m.supported_actions)
