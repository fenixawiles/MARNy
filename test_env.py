from dotenv import load_dotenv
import os

load_dotenv()  # automatically finds .env in the current working directory
api_key = os.getenv("OPENAI_API_KEY")

print(api_key)  # should print your key if everything is loaded correctly