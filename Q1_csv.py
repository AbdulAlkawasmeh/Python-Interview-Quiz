import os
import pandas as pd
import requests
from io import StringIO
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

CSV_URL = "https://data.cityofnewyork.us/api/views/uvpi-gqnh/rows.csv?accessType=DOWNLOAD"
response = requests.get(CSV_URL)
if response.status_code == 200:
    data = StringIO(response.text)
    df = pd.read_csv(data)
else:
    raise Exception("Failed to fetch CSV file")




print(df.head())
