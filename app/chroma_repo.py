import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from app.embeddings import get_embedding
from app.types import VectorizedChunk, Chunk, MetaData


class ChromaDocChat:
    """Операции с ChromaDB."""

    def __init__(
            self,
            chroma_persistant_dir: str = "./chroma_db",
            collection_name: str = "documents_ru_clean"
    ):
        """Инициализирует ChromaDB в persistent режиме."""
        self.client = chromadb.PersistentClient(path=chroma_persistant_dir)
        self.collection_name = collection_name

        self.collection: chromadb.Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f' ChromaDB инициализирована: {chroma_persistant_dir}')

    def add_record(self, chunk: VectorizedChunk):
        """Добавляет запись."""
        self.collection.upsert(
                ids=chunk.doc_id,
                embeddings=chunk.embedding,
                metadatas=chunk.metadata.to_dict(),
                documents=chunk.text
            )

    def clear(self):
        """Очищает текущую коллекцию ChromaDB."""
        records = self.collection.get()
        ids = records.get('ids', [])
        if ids:
            self.collection.delete(ids=ids)
        print(f'ChromaDB коллекция очищена: {self.collection_name}')

    def search_records(
        self,
        query: str,
        top_k: int = 10
    ) -> List[VectorizedChunk]:
        """Cемантический поиск по векторной БД.

        Возвращает релевантные чанки с метаданными.
        """
        query_embedding = get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        chunks = []
        if results['documents'] and results['documents'][0]:
            for result_idx in range(len(results['ids'][0])):
                chunk = Chunk(
                    results['ids'][0][result_idx],
                    results['documents'][0][result_idx],
                    MetaData(
                        results['metadatas'][0][result_idx]['file_name'],
                        results['metadatas'][0][result_idx]['chunk_number'],
                        results['metadatas'][0][result_idx]['char_start'],
                        results['metadatas'][0][result_idx]['char_end']
                    )
                )
                vectorized_chank = VectorizedChunk(
                    chunk,
                    distance=(results['distances'][0][result_idx]
                              if results['distances']
                              else None))
                chunks.append(vectorized_chank)
        return chunks
