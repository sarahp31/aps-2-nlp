import os
from dotenv import load_dotenv
load_dotenv()

from app import create_app
from app.logging_config import setup_logging

setup_logging()

app = create_app()

if __name__ == '__main__':
    port = os.getenv('PORT', 5001)
    app.run(host='0.0.0.0', debug=True, port=port)
