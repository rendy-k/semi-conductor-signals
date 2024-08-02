from os import environ
from dotenv import load_dotenv

load_dotenv()

SIGNAL_BACKEND = environ['SIGNAL_BACKEND']
