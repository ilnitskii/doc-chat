from typing import List
import json
import requests
import base64
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from app.types import VectorizedChunk

import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

def get_embedding_api(text: str) -> List[float] | None:
    """Получение эмбеддинга."""
    request_body = {'text': text}
    request_headers = {'Content-Type': 'application/json'}
    request_payload = json.dumps(request_body)
    try:
        response = requests.post(
            'http://127.0.0.1:5000/api/embedding/',
            data=request_payload,
            headers=request_headers
        )
    except Exception as error:
        print(f'Ошибка при выполнении запроса к embedding-сервису: {error}')
        return None

    if response.status_code != 200:
        print(f'Код ответа embedding-сервиса: {response.status_code}')
        return None

    try:
        response_embedding = deserialize_array(
            json.loads(response.content.decode('utf-8')).get('embedding'))
    except Exception as error:
        print(f'Ошибка при парсинге JSON ответа от embedding-сервиса: {error}')
    return response_embedding.tolist()


def deserialize_array(payload: dict) -> np.ndarray:
    """Десериализация эмбеддинга."""
    data = base64.b64decode(payload['data'])
    arr = np.frombuffer(data, dtype=np.dtype(payload['dtype']))
    return arr.reshape(payload['shape'])


def get_embedding(text: str) -> List[float] | None:
    """Формирует эмбеддинг текста."""
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


# MODEL_PATH = './models/all-MiniLM-L6-v2'
MODEL_PATH = './models/bge-m3'
"""Пусть к локальной модели для построения эмбеддингов."""

RERANKER_MODEL_PATH = './models/bge-reranker-v2-m3'
"""Пусть к локальной модели для реранкинга."""

USE_LOCAL_MODELS = False
if USE_LOCAL_MODELS:
    model = SentenceTransformer(MODEL_PATH, local_files_only=USE_LOCAL_MODELS)
else:
    model = SentenceTransformer('BAAI/bge-m3')
print(f'Модель-энкодер загружена. {USE_LOCAL_MODELS=}')

if USE_LOCAL_MODELS:
    reranker = CrossEncoder(RERANKER_MODEL_PATH, device='cpu')
else:
    reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu')


print('Реранкер загружен.')


def calc_relevance_logit(query, chunks):
    """Вычисляет логиты релевантности с помощью модели кросс-энкодера."""
    pairs = [[query, chunk.text] for chunk in chunks]
    scores = reranker.predict(pairs)
    for idx, chunk in enumerate(chunks):
        chunk.score = scores[idx]
    return chunks


# pairs1 = [['Coating market in Bahrain', 'Crude oil exports were halved in 2019, and inflation surged to over 20% in both 2018 and 2019. GDP fell by 7.6% in 2019 and by another 6% in 2020. In 2021, GDP increased by 4.6%, and in 2022 by 3%. The Iranian market for coatings is highly fragmented, with over 350 authorized paint-producing factories and a total annual capacity of 900,000 metric tons for various types of decorative and industrial paints. In addition, there are quite a number of active unauthorized producers as well. Still, the market is dominated by 20-30 local players. So far, only Iranian companies are allowed to manufacture in Iran, which adheres to the Iranian government economic policy that encourages national production; however, imports of specialized coatings are allowed. Kansai Paint built a plant in Iran in 2009 with an annual capacity of 18,000 metric tons to serve Iranian automakers. SigmaKalon (PPG) entered the anticorrosion coatings market in 2010 and Bajak Paints entered into a joint venture with KKC Corp from South Korea. Also in 2010, National Paints, one of the largest paint manufacturers in the Middle East, opened two plants in Iran with a total capacity of over 60,000 metric tons per year to supply Iranian and Iraqi markets. In 2016, Kansai Paint announced plans to re-enter Iran after Japan decided to lift sanctions. In a joint venture plant, Kansai wants to supply coatings to Iranian automakers and protective coatings for storage tanks and pipelines. Estimated consumption of coatings in 2022 was 480,000 metric tons. Due to the reimposition of sanctions, coatings consumption declined significantly in recent years. The market split is as follows: 55% decorative coatings, 16% protective coatings, 8% automotive coatings, 7% powder coatings and 5% marine coatings. The decorative coatings market is characterized by huge construction activities in Iran. About 70% of decorative coatings are sold through retail businesses.'], ['Coating market in Bahrain', 'Total 2022 paint and coatings consumption is estimated at approximately 2.9 million metric tons in the Middle East (including Turkey). Like all countries of the world, the economies of Middle Eastern countries declined in 2020. The decline varied by country but most countries experienced GDP declines of 6%-8%. Accordingly, the Middle Eastern coatings markets were down 6%. After recovery from the COVID-19 pandemic, overall growth is projected at 4%-5% per year on average through 2028. Growth may be lower if oil prices fall or if regional political instability increases. Lower oil prices will hurt growth in Saudi Arabia, Kuwait, Iran and especially the United Arab Emirates. Bahrain  In Bahrain, in the Persian Gulf, a booming economy and construction industry has pushed the value of the country’s coatings market to $20 million annually.  Iran  The elimination of sanctions in 2015 opened new business opportunities, but the reimposition of sanctions by the United States in 2018 dampened economic growth again. Crude oil exports were halved in 2019, and inflation surged to over 20% in both 2018 and 2019. GDP fell by 7.6% in 2019 and by another 6% in 2020. In 2021, GDP increased by 4.6%, and in 2022 by 3%.  The Iranian market for coatings is highly fragmented, with over 350 authorized paint-producing factories and a total annual capacity of 900,000 metric tons for various types of decorative and industrial paints.']]
# pairs2 = [['Coating market in Bahrain', 'Total 2022 paint and coatings consumption is estimated at approximately 2.9 million metric tons in the Middle East (including Turkey). Like all countries of the world, the economies of Middle Eastern countries declined in 2020. The decline varied by country but most countries experienced GDP declines of 6%-8%. Accordingly, the Middle Eastern coatings markets were down 6%. After recovery from the COVID-19 pandemic, overall growth is projected at 4%-5% per year on average through 2028. Growth may be lower if oil prices fall or if regional political instability increases. Lower oil prices will hurt growth in Saudi Arabia, Kuwait, Iran and especially the United Arab Emirates. Bahrain  In Bahrain, in the Persian Gulf, a booming economy and construction industry has pushed the value of the country’s coatings market to $20 million annually.  Iran  The elimination of sanctions in 2015 opened new business opportunities, but the reimposition of sanctions by the United States in 2018 dampened economic growth again. Crude oil exports were halved in 2019, and inflation surged to over 20% in both 2018 and 2019. GDP fell by 7.6% in 2019 and by another 6% in 2020. In 2021, GDP increased by 4.6%, and in 2022 by 3%.  The Iranian market for coatings is highly fragmented, with over 350 authorized paint-producing factories and a total annual capacity of 900,000 metric tons for various types of decorative and industrial paints.']]
# scores1 = reranker.predict(pairs1)
# scores2 = reranker.predict(pairs2)
# print(scores1)
# print(scores2)

