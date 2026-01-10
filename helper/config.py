import os
from dotenv import load_dotenv


def get_db_config():
    """Load and validate database configuration from environment variables.

    Returns:
        dict: Database configuration with keys: user, password, host, port, db

    Raises:
        RuntimeError: If required environment variables are missing
    """
    load_dotenv()

    # Get values with defaults
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'default_user')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'default_pass')
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', '5432'))
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'hacker_news')

    return {
        'user': POSTGRES_USER,
        'password': POSTGRES_PASSWORD,
        'host': POSTGRES_HOST,
        'port': POSTGRES_PORT,
        'db': POSTGRES_DB
    }
