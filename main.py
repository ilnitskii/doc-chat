
from typing import List, Dict, Any
import hashlib
from app.llm_engine import OpenAICompatibleEngine, BaseAgent
from app.chroma_repo import ChromaDocChat
from app.elastic_repo import ElasticDocChat
from app.text_processing import (
    read_html_files,
    read_pdf_files,
    extract_text_from_html
)
from app.embeddings import get_embedding
from tqdm import tqdm
from app.types import Chunk, MetaData, VectorizedChunk

from app.retrieval import HybridRetriever

CHUNK_SIZE = 300
CHUNK_OVERLAP = 30


def is_end_of_sentence(word) -> bool:
    """Проверяет, что слово стоит в конце предложения."""
    stop_chars = {'.', '!', '?'}
    if word in stop_chars or word[-1] in stop_chars:
        return True
    return False


def chunk_text(text: str, filename: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """Разбивает текст на чанки с перекрытием.

    Возвращает список чанков с метаданными.
    """
    chunks = []

    words = text.split()
    chunk_words = []
    current_size = 0
    chunk_number = 1
    slice_flag = False

    for i, word in tqdm(enumerate(words)):
        chunk_words.append(word)
        current_size += 1

        if current_size >= chunk_size or i == len(words) - 1:
            slice_flag = True

        if slice_flag and is_end_of_sentence(word):
            chunk_text = ' '.join(chunk_words)

            chunk_hash = hashlib.md5(
                f"{filename}:{chunk_number}:{chunk_text[:50]}".encode()
            ).hexdigest()[:16]

            if i-len(chunk_words) < 0:
                char_start = 0
            else:
                char_start = sum([len(w)+1 for w in words[:i-len(chunk_words)]])
            char_end = sum([len(w)+1 for w in words[:i+1]])

            metadata = MetaData(filename, chunk_number, char_start, char_end)
            chunks.append(Chunk(chunk_hash, chunk_text, metadata))
            chunk_number += 1

            if overlap > 0 and i < len(words) - 1:
                while not is_end_of_sentence(chunk_words[len(chunk_words)-overlap-1]):
                    overlap += 1

                overlap_words = chunk_words[len(chunk_words)-overlap:]
                overlap = CHUNK_OVERLAP
                chunk_words = overlap_words
                current_size = len(overlap_words)
            else:
                chunk_words = []
                current_size = 0
            slice_flag = False

    return chunks


def process_file(file_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Полная обработка файла: извлечение текста, чанкинг."""
    text = file_data.get('text') or extract_text_from_html(file_data['content'])
    chunks = chunk_text(text, file_data['filename'])
    print(f"  - File {file_data['filename']}: {len(chunks)} chanks")
    return chunks


def load_embeddings_to_repos(
        chroma_chat: ChromaDocChat,
        elastic_chat: ElasticDocChat,
        chunks: List[Chunk]
) -> None:
    """Загружает эмбеддинги чанков в БД."""
    if not chunks:
        return
    print(f'Подготовка {len(chunks)} чанков в ChromaDB...')

    for chunk in tqdm(chunks):
        vectorized_chunk = VectorizedChunk(
            chunk,
            embedding=get_embedding(chunk.text)
        )
        chroma_chat.add_record(vectorized_chunk)
        elastic_chat.add_record(chunk)

    print('Эмбеддинги загружены в БД')


def do_indexing(
        chroma_chat: ChromaDocChat,
        elastic_chat: ElasticDocChat,
        docs_directory: str = "./html_docs"
):
    print('Индексация документов...')
    files_data = read_html_files(docs_directory) + read_pdf_files(docs_directory)
    if not files_data:
        print('HTML/PDF файлы не найдены!')
        return

    total_chunks = 0
    for file_data in files_data:
        print(f'Обработка файла {file_data['filename']}')
        chunks = process_file(file_data)
        load_embeddings_to_repos(chroma_chat, elastic_chat, chunks)
        total_chunks += len(chunks)

    print((f'Индексация завершена: {len(files_data)} файлов, '
           f'{total_chunks} чанков'))


def main():
    engine = OpenAICompatibleEngine()
    print(f'Список моделей на LLM-сервере: {engine.list_models()}')
    agent = BaseAgent(engine, model='qwen3.5-9b', temperature=0.1)
    # agent = BaseAgent(engine, model='google/gemma-4-e2b', temperature=0.1)

    collection_name = 'fiction_books'
    chroma_chat = ChromaDocChat(collection_name=collection_name)
    elastic_chat = ElasticDocChat(index_name=collection_name)

    retriever = HybridRetriever(collection_name, chroma_chat, elastic_chat)

    if input('Провести индексацию? [Y|y - да]:').lower() == 'y':
        do_indexing(chroma_chat, elastic_chat)

    print('Поиск по документам (введите "quit" для выхода)')

    while True:
        try:
            query = input('\nВаш вопрос: ').strip()

            if query.lower() in ['quit', 'exit', 'q']:
                break

            if not query:
                continue

            # msg_chroma = agent.build_messages(f'Сгенерируй поисковый запрос на английском языке для семантического поиска в локальной базе данных информации по следующему пользовательскому запросу: {query}. Ответ должен содержать только фразу на английском языке для поиска необходимой пользователю информации. Не выводи никакой другой информации.')
            msg_chroma = agent.build_messages(f'Сгенерируй поисковый запрос на русском языке для поиска в локальной базе данных информации по следующему пользовательскому запросу: {query}. Ответ должен содержать только фразу на русском языке для поиска необходимой пользователю информации. Не выводи никакой другой информации.')
            # msg_es = agent.build_messages(f'Сгенерируй до 3 ключевых слов на английском языке для поиска в локальной локальной базе данных информации по следующему пользовательскому запросу: {query}. Ответ должен содержать только список ключевых слов на английском языке, разделенных пробелом, для поиска необходимой пользователю информации. Не выводи никакой другой информации.')
            msg_es = agent.build_messages(f'Сгенерируй до 3 ключевых слов на русском языке для поиска в локальной локальной базе данных информации по следующему пользовательскому запросу: {query}. Ответ должен содержать только список ключевых слов на русском языке, разделенных пробелом, для поиска необходимой пользователю информации. Не выводи никакой другой информации.')

            query_for_semantic_search = agent.generate(
                msg_chroma).get('content')
            key_words = agent.generate(msg_es).get('content')

            print(f'Semantic search query: {query_for_semantic_search}')
            print(f'Elastic  search_query: {key_words}')

            relevant_chunks = retriever.retrieve_relevant(query_for_semantic_search, key_words)
            print(f'Итого найдено {len(relevant_chunks)} чанков: {[chunk.doc_id for chunk in relevant_chunks]}')

            search_result = '\n\n'.join([chunk.text for chunk in relevant_chunks])
            print('[AI Agent] working ...')
            msg = agent.build_messages(f'Пользователь задал вопрос: {query}.\n\nВ результате поиска была найдена следующая информация:\n{search_result}\n\nНа основе найденной информации дай ясный, четкий и понятный ответ пользователю.')
            answer = agent.generate(msg).get('content')
            print(f'[AI Agent] Result\n{answer}')

        except KeyboardInterrupt:
            print('Останов')
            break


if __name__ == "__main__":
    main()
