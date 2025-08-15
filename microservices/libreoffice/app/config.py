import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    LIBREOFFICE_HOST = os.getenv("LIBREOFFICE_HOST", "libreoffice")
    LIBREOFFICE_PORT = int(os.getenv("LIBREOFFICE_PORT", 2002))
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEFAULT_DPI = int(os.getenv("DEFAULT_DPI", 300))
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", 50)) * 1024 * 1024
    TIMEOUT = int(os.getenv("TIMEOUT_SECONDS", 300))
    DOCUMENTS_DIR = "/app/documents"
    TEMP_DIR = "/app/temp"