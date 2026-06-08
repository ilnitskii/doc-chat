from app.chroma_repo import ChromaDocChat
from app.elastic_repo import ElasticDocChat
from app.types import VectorizedChunk
from app.embeddings import calc_relevance_logit


class HybridRetriever:
    """Гибридный поиск релевантных данных.

    Включает семантический поиск в векторной БД,
    поиск по ключевым словам в Elasticsearch."""

    def __init__(
            self,
            collection_name: str,
            chroma_chat: ChromaDocChat | None = None,
            elastic_chat: ElasticDocChat | None = None
    ) -> None:
        """Инициализирует поисковик по заданной коллекции."""
        self.chroma_chat = (ChromaDocChat(collection_name=collection_name)
                            if chroma_chat is None
                            else chroma_chat)
        self.elastic_chat = (ElasticDocChat(index_name=collection_name)
                             if elastic_chat is None
                             else elastic_chat)
        self.reranker = Reranker()

    def get(
        self,
        search_phrase: str,
        key_words: str,
        top_k: int
    ) -> list[list[VectorizedChunk]]:
        """Извлекает релевантные документы из коллекции."""
        results = []
        for engine in (self.chroma_chat, self.elastic_chat):
            search_query = (search_phrase
                            if isinstance(engine, ChromaDocChat)
                            else key_words)
            print(f'Search by {engine}. Query: {search_query}')
            collection_result: list[VectorizedChunk] = engine.search_records(
                search_query,
                top_k
            )
            results.append(collection_result)
        return results

    def retrieve_relevant(
            self,
            query_for_semantic_search: str,
            key_words: str,
            top_k: int = 10
    ) -> list[VectorizedChunk]:
        """Возвращает релевантные запросу чанки."""
        all_results = self.get(
            query_for_semantic_search, key_words, top_k)
        filtered_results = []
        for idx, results in enumerate(all_results, start=1):
            print(f'COLLECTION OF RESULTS {idx}')
            for rec in results:
                print((f'[{rec.metadata.file_name}.'
                       f'{rec.metadata.chunk_number}] {rec.text[:150]}'))

            ranked_chunks = self.reranker.rank(
                query_for_semantic_search, results)
            rrf_ranked_chunks = self.reranker.rrf(
                [results, ranked_chunks])
            filtered_results.append(
                self.reranker.filter_by_score(rrf_ranked_chunks))
        return self.reranker.merge_rankings(*filtered_results)


class Reranker:
    """Ранжирование результатов поиска."""

    def __init__(self):
        pass

    def rank(
        self,
        query: str,
        chunks: list[VectorizedChunk]
    ) -> list[VectorizedChunk]:
        """Ранжирует результаты поиска."""
        unique_chunks = {chunk.doc_id: chunk for chunk in chunks}
        chunks = list(unique_chunks.values())
        chunks = calc_relevance_logit(query, chunks)
        chunks.sort(key=lambda x: x.score, reverse=True)
        for chunk in chunks:
            print(f'{chunk.metadata.chunk_number} - [{chunk.score}]')
        return chunks

    def rrf(
        self,
        rankings: list[list[VectorizedChunk]],
        k: int = 60
    ) -> list[VectorizedChunk]:
        """Перекрестное объединение оценок релевантности.

        Используется метод Reciprocal Rank Fusion."""
        chunk_by_id = dict()
        rrf_score = {}
        for ranking in rankings:
            for rank, chunk in enumerate(ranking, start=1):
                chunk_by_id[chunk.doc_id] = chunk
                if chunk.doc_id not in rrf_score:
                    rrf_score[chunk.doc_id] = 0
                rrf_score[chunk.doc_id] += 1/ (k + rank)
        sorted_chunks_ids = sorted(
            rrf_score.items(), key=lambda x: x[1], reverse=True)

        sorted_chunks = [chunk_by_id.get(doc_item[0])
                         for doc_item in sorted_chunks_ids]

        print('Reciprocal Rank Fusion')
        for chunk in sorted_chunks:
            print(f'{chunk.metadata.chunk_number}')
        return sorted_chunks

    def filter_by_score(
        self,
        ranking: list[VectorizedChunk],
        threshold: float = 0.7,
        fallback_top_k: int = 3
    ) -> list[VectorizedChunk]:
        """Фильтрация выборки по пороговому значения."""
        result = [chunk for chunk in ranking if chunk.score > threshold]
        if not result:
            result = ranking[:fallback_top_k]
        print('Filtering')
        for chunk in result:
            print(f'{chunk.metadata.chunk_number} - [{chunk.score}]')
        return result

    def merge_rankings(
        self,
        ranking_1: list[VectorizedChunk],
        ranking_2: list[VectorizedChunk]
    ) -> list[VectorizedChunk]:
        """Объединяет ранжированные списки результатов отбора."""
        if not ranking_1:
            return ranking_2
        if not ranking_2:
            return ranking_1
        p1 = 0
        p2 = 0
        result = []
        unique_doc_id = set()
        while p1 < len(ranking_1) and p2 < len(ranking_2):
            if ranking_1[p1].score > ranking_2[p2].score:
                if ranking_1[p1].doc_id not in unique_doc_id:
                    result.append(ranking_1[p1])
                    unique_doc_id.add(ranking_1[p1].doc_id)
                p1 += 1
            else:
                if ranking_2[p2].doc_id not in unique_doc_id:
                    result.append(ranking_2[p2])
                    unique_doc_id.add(ranking_2[p2].doc_id)
                p2 += 1
            if p1 >= len(ranking_1):
                tail = ranking_2[p2:]
            if p2 >= len(ranking_2):
                tail = ranking_1[p1:]
        result.extend(tail)
        print('MERGE')
        for chunk in result:
            print(f'{chunk.metadata.chunk_number} - [{chunk.score}]')
        return result
