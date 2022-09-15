from flask import Flask
# from config import Config
from dotenv import load_dotenv


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
# app.config.from_object(Config)
load_dotenv()

from application import routes