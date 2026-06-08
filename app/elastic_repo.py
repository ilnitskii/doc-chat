from elasticsearch import Elasticsearch, exceptions
import time
from app.types import MetaData, Chunk, ES_MAPPING


class ElasticBaseRepository:
    """Базовые операции с Elasticsearch."""

    def __init__(self, host: str = 'localhost', port: str = '9200'):
        self.es = Elasticsearch([f'http://{host}:{port}'])
        if not self.es.ping():
            raise ConnectionError('Не удалось подключиться к Elasticsearch')

    def get_indices(self) -> list | None:
        """Получение списка индексов на сервере."""
        try:
            indices = self.es.cat.indices(format='json')
        except Exception as err:
            print(err)
            return
        return [idx['index'] for idx in indices]

    def create_index(self, index_name: str, mapping: dict):
        """Создание индекса."""
        try:
            response = self.es.indices.create(
                index=index_name,
                body={
                        'settings': {
                            'number_of_shards': 1,
                            'number_of_replicas': 0
                        },
                        'mappings': {
                            'properties': mapping
                        }
                    },
                ignore=[400]
            )
        except Exception as e:
            print(e)
            return
        return response.get('status')

    def delete_index(self, index_name: str):
        """Удаление индекса."""
        try:
            self.es.indices.delete(index=index_name, ignore=[400, 404])
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            return

    def add_one(self, index_name: str, document: dict, doc_id: int):
        """Добавление документа в индекс."""
        for retry in range(5):
            try:
                response = self.es.index(
                    index=index_name, id=doc_id, document=document)
                self.es.indices.refresh(index=index_name)
                return result['result'] if (result := response.body) else 'ok'
            except Exception:
                print(f'Документ {doc_id} не индексирован. Повторная попытка.')
                time.sleep(3)

    def search(
            self,
            index_name: str,
            search_query: dict,
            fields: list,
            top_k: int = 50
    ):
        """Простой поиск (Simple Query String)."""
        simple_query = {
            'query': {
                'simple_query_string': {
                    'query': search_query,
                    'fields': fields,
                }
            },
            'size': top_k,
            'track_total_hits': True,
            'from': 0
        }
        response_data = self.es.search(index=index_name, body=simple_query)
        search_result_count = int(
            response_data.get('hits').get('total').get('value'))
        print(f'TOTAL RESULTS: {search_result_count}')
        hits = response_data['hits']['hits']
        return [hit.get('_source') for hit in hits]

    def delete_one(self, index_name: str, doc_id: int) -> int | None:
        """Удаление документа из индекса."""
        try:
            self.es.delete(index=index_name, id=doc_id)
            return doc_id
        except exceptions.NotFoundError:
            return None


class ElasticDocChat:
    """Elasticsearch для DocChat."""

    def __init__(
            self,
            host: str = 'localhost',
            port: str = '9200',
            index_name: str = None
    ):
        """Инициализация Elasticsearch."""
        map = ES_MAPPING
        self.elastic_repository = ElasticBaseRepository(host, port)
        print('Создание индекса ES...')
        self.elastic_repository.create_index(index_name, map)
        print(self.elastic_repository.get_indices())
        self.index_name = index_name

    def add_record(self, chunk: Chunk):
        """Добавление чанка в индекс."""
        self.elastic_repository.add_one(
            self.index_name, chunk.to_dict(), chunk.doc_id)

    def clear(self):
        """Очищает текущий индекс Elasticsearch."""
        self.elastic_repository.delete_index(self.index_name)
        self.elastic_repository.create_index(self.index_name, ES_MAPPING)
        print(f'Elasticsearch индекс очищен: {self.index_name}')

    def search_records(self, search_query: str, top_k: int = 50) -> list[Chunk]:
        """Полнотекстовый поиск.

        Возвращает список записей.
        """

        records = self.elastic_repository.search(
            self.index_name,
            search_query,
            ['text'],
            top_k
        )
        return [Chunk(
                    record['id'],
                    record['text'],
                    MetaData(**record['metadata'])) for record in records]


if __name__ == "__main__":
    es = ElasticBaseRepository()
    print(es.get_indices())
    if input('q - to exit, any char - to continue') == 'q':
        raise SystemExit
    # es.delete_index('navy-ru-1')
    # es.delete_index('my-collection-navy-ru')
    # es.delete_index('chat_doc_index')
    # es.delete_index('my-collection-1')

    # es = ElasticDocChat(index_name='my-collection-navy-ru')

    while True:
        search = input("search: ")
        search_result = es.search('paintings1', search, ['text'])
        search_result.sort(key=lambda x: x['metadata']['chunk_number'])

        for r in search_result:
            print(f'{r['metadata']["chunk_number"]}]{r["text"]}[{r['metadata']["chunk_number"]}|', end='')

            # print(f'{r['file_name']} - {r['chunk_number']}')
