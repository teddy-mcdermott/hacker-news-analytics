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
    user = os.getenv('POSTGRES_USER', 'default_user')
    password = os.getenv('POSTGRES_PASSWORD', 'default_pass')
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = int(os.getenv('POSTGRES_PORT', '5432'))
    db = os.getenv('POSTGRES_DB', 'hacker_news')

    return {
        'user': user,
        'password': password,
        'host': host,
        'port': port,
        'db': db
    }
