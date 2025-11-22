import os
from google.genai import client
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# Print key status (should NOT print the full key for security)
if api_key:
    print("API key found.")
else:
    print("API key NOT found.")
    exit()

# Initialize Gemini client
cl = client.Client(api_key=api_key)

# Test Gemini
try:
    response = cl.models.generate_content(
        model="gemini-2.5-flash",
        contents="Test message for Gemini API. Reply with: Success."
    )
    print("Gemini Response:", response.text)

except Exception as e:
    print("Error:", e)
