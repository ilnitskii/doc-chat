from pathlib import Path

import streamlit as st


DEFAULT_DOCS_DIR = './html_docs'
DEFAULT_COLLECTION = 'fiction_books'
DEFAULT_MODEL = 'qwen3.5-9b'
DEFAULT_LLM_HOST = 'http://127.0.0.1:1234'


st.set_page_config(page_title='Doc Chat', layout='wide')


@st.cache_resource(show_spinner='Загрузка моделей и подключение к БД...')
def get_runtime(
    collection_name: str,
    llm_host: str,
    model_name: str,
    temperature: float
):
    from app.chroma_repo import ChromaDocChat
    from app.elastic_repo import ElasticDocChat
    from app.llm_engine import BaseAgent, OpenAICompatibleEngine
    from app.retrieval import HybridRetriever

    engine = OpenAICompatibleEngine(host=llm_host)
    agent = BaseAgent(engine, model=model_name, temperature=temperature)
    chroma_chat = ChromaDocChat(collection_name=collection_name)
    elastic_chat = ElasticDocChat(index_name=collection_name)
    retriever = HybridRetriever(collection_name, chroma_chat, elastic_chat)
    return engine, agent, chroma_chat, elastic_chat, retriever


def save_uploaded_files(uploaded_files, docs_directory: str) -> list[str]:
    docs_path = Path(docs_directory)
    docs_path.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for uploaded_file in uploaded_files:
        filename = Path(uploaded_file.name).name
        filepath = docs_path / filename
        filepath.write_bytes(uploaded_file.getbuffer())
        saved_files.append(filename)

    return saved_files


def build_answer(agent, retriever, query: str, top_k: int):
    msg_chroma = agent.build_messages(
        'Сгенерируй поисковый запрос на русском языке для поиска в локальной '
        'базе данных информации по следующему пользовательскому запросу: '
        f'{query}. Ответ должен содержать только фразу на русском языке для '
        'поиска необходимой пользователю информации. Не выводи никакой другой '
        'информации.'
    )
    msg_es = agent.build_messages(
        'Сгенерируй до 3 ключевых слов на русском языке для поиска в локальной '
        'локальной базе данных информации по следующему пользовательскому '
        f'запросу: {query}. Ответ должен содержать только список ключевых слов '
        'на русском языке, разделенных пробелом, для поиска необходимой '
        'пользователю информации. Не выводи никакой другой информации.'
    )

    query_for_semantic_search = agent.generate(msg_chroma).get('content')
    key_words = agent.generate(msg_es).get('content')
    relevant_chunks = retriever.retrieve_relevant(
        query_for_semantic_search,
        key_words,
        top_k=top_k
    )
    search_result = '\n\n'.join([chunk.text for chunk in relevant_chunks])
    msg = agent.build_messages(
        f'Пользователь задал вопрос: {query}.\n\n'
        f'В результате поиска была найдена следующая информация:\n'
        f'{search_result}\n\n'
        'На основе найденной информации дай ясный, четкий и понятный ответ '
        'пользователю.'
    )
    answer = agent.generate(msg).get('content')
    return answer, relevant_chunks, query_for_semantic_search, key_words


def get_runtime_or_stop():
    try:
        return get_runtime(
            collection_name,
            llm_host,
            model_name,
            temperature
        )
    except Exception as error:
        st.error(f'Не удалось инициализировать систему: {error}')
        st.stop()


with st.sidebar:
    st.header('Настройки')
    docs_directory = st.text_input('Папка документов', DEFAULT_DOCS_DIR)
    collection_name = st.text_input('Коллекция', DEFAULT_COLLECTION)
    llm_host = st.text_input('LLM host', DEFAULT_LLM_HOST)
    model_name = st.text_input('Модель', DEFAULT_MODEL)
    temperature = st.slider('Temperature', 0.0, 1.0, 0.1, 0.1)
    top_k = st.slider('Top K', 1, 30, 10, 1)

    st.divider()
    uploaded_files = st.file_uploader(
        'Документы',
        type=['html', 'htm', 'pdf'],
        accept_multiple_files=True
    )
    if uploaded_files and st.button('Сохранить файлы'):
        saved_files = save_uploaded_files(uploaded_files, docs_directory)
        st.success(f'Сохранено файлов: {len(saved_files)}')

    if st.button('Индексировать'):
        from main import do_indexing

        _, _, chroma_chat, elastic_chat, _ = get_runtime_or_stop()
        with st.spinner('Индексация...'):
            do_indexing(chroma_chat, elastic_chat, docs_directory=docs_directory)
        st.success('Индексация завершена')

    if st.button('Проверить LLM'):
        engine, _, _, _, _ = get_runtime_or_stop()
        st.write(engine.list_models())


st.title('Doc Chat')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

query = st.chat_input('Ваш вопрос')
if query:
    st.session_state.messages.append({'role': 'user', 'content': query})
    with st.chat_message('user'):
        st.markdown(query)

    _, agent, _, _, retriever = get_runtime_or_stop()
    with st.chat_message('assistant'):
        with st.spinner('Поиск и генерация ответа...'):
            answer, chunks, semantic_query, key_words = build_answer(
                agent,
                retriever,
                query,
                top_k
            )
        st.markdown(answer)
        with st.expander('Поисковые запросы'):
            st.write({'semantic_query': semantic_query, 'key_words': key_words})
        with st.expander(f'Найденные чанки: {len(chunks)}'):
            for chunk in chunks:
                score = getattr(chunk, 'score', None)
                score_text = f' | score: {score:.3f}' if score is not None else ''
                st.markdown(
                    f'**{chunk.metadata.file_name} / '
                    f'chunk {chunk.metadata.chunk_number}{score_text}**'
                )
                st.write(chunk.text)

    st.session_state.messages.append({'role': 'assistant', 'content': answer})
