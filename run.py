from app import create_app
import sys
import os
from pyngrok import ngrok

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = create_app()

USE_NGROK = False
if __name__ == '__main__':
    if USE_NGROK:
        PORT_NO = 5000
        public_url = ngrok.connect(PORT_NO).public_url

        print(f"Access here: {public_url}")
        app.run(port=PORT_NO)
    else:
        app.run(debug=True)

