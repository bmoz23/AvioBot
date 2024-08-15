import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain.schema import SystemMessage, HumanMessage
from streamlit_chat import message
import os
import speech_recognition as sr
import time
import pyttsx3
from dotenv import load_dotenv, find_dotenv
import pyaudio

r = sr.Recognizer()

def record_audio():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.2)
        st.success("Listening...")
        audio = r.listen(source)
        return audio

def transcribe_audio(audio_data):
    try:
        if audio_data:
            text = r.recognize_google(audio_data)
            st.write(f"You said: {text}")
            return text
    except sr.RequestError as e:
        st.write(f"Could not request results; {e}")
    except sr.UnknownValueError:
        st.write("Unknown audio value")
    return None

def load_document(file):
    import os
    name, extension = os.path.splitext(file)
    
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        try:
            loader = PyPDFLoader(file_path=file)
            data = loader.load()
            return data
        except Exception as e:
            print(f"Error loading PDF file: {e}")
            return None
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        try:
            loader = Docx2txtLoader(file)
            data = loader.load()
            return data
        except Exception as e:
            print(f"Error loading DOCX file: {e}")
            return None
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader=TextLoader(file)
    else:
        print('Document format is not supported')
        return None

def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks=text_splitter.split_documents(data)
    return chunks

def insert_or_fetch_embeddings(index_name, chunks):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain.embeddings import OpenAIEmbeddings
    from pinecone import PodSpec

    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')  # OpenAI modelini doğru şekilde belirtin

    if index_name in pc.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings...', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('ok')
    else:
        print(f'Creating index {index_name} and embeddings...', end='')
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=PodSpec(environment = 'gcp-starter')
        )
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')

    return vector_store

def delete_pinecone_index(index_name='all'):
    import pinecone
    pc = pinecone.Pinecone()
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        print('Deleting all indexes...')
        for index in indexes:
            pc.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')

def ask_and_get_answer(vector_store, q, k=5):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    
    #system_message=SystemMessage(content="Your name is Aviobot. You are an assistant of pilot. You just answer the questions that are available on vectore database as vectore store. Other than this, just say 'I can only answer based on the data. I don't know.' ")
    llm = ChatOpenAI(model='gpt-4-turbo-preview', 
                     temperature=1,
                     )
    if vector_store is None:
        raise ValueError("Vector store is None, cannot proceed with the retrieval.")
    
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    # k means that it will return the k size most similar chunks to the user's query.
    
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

    answer = chain.run(q)

    return answer

def speak(text):
    import openai
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False

    # Hypothetical usage of openai.audio.speech.with_streaming_response.create
    with openai.audio.speech.with_streaming_response.create(
        model='tts-1',
        voice='nova',
        response_format='pcm',
        input=text,
    ) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            else:
                if max(chunk) > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True

    player_stream.stop_stream()
    player_stream.close()

def welcome_message():
    message = (
        "Welcome to AvioBot, your advanced co-pilot designed AI. "
        "I am here to assist you with real-time flight. "
        "Let's ensure a secure and efficient flight operation together."
    )
    speak(message)

if __name__ == "__main__":

    load_dotenv(find_dotenv("req.env"), override=True)
    st.image("img/bites.png", width=700)
    st.markdown("<h1 style='text-align: center; color: white;'>Welcome to AvioBot</h1>", unsafe_allow_html=True)

    # Display the welcome message once
    if 'welcome_message_displayed' not in st.session_state:
        welcome_message()
        st.session_state.welcome_message_displayed = True
    user_photo = "bites.png"

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'running' not in st.session_state:
        st.session_state.running = False


    with st.sidebar:
        uploaded_file = st.file_uploader('Add data:', type=['pdf', 'docx', 'txt'])
        index_name = st.text_input("Enter the index name for the vector database")
        add_data = st.button('Create')
        
        if uploaded_file and add_data:
            with st.spinner('Reading, processing, and transferring the file to the vector database...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data)
                delete_pinecone_index()
                vector_store = insert_or_fetch_embeddings(index_name, chunks)
                st.session_state.vs = vector_store
                st.success('File uploaded, vector database checked, and file embedded.')

        # Custom CSS for buttons
  # Custom CSS for buttons
    st.markdown(
        """
        <style>
        .button-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        .stButton button {
            height: 50px;
            width: 100px;
            font-size: 16px;
            font-weight: bolder; /* Make text bold */
            background-color: white; /* Green */
            color: #1C3D6C;
            border: none;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            transition-duration: 0.4s;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #61C51F;
            color: white;
            border: 2px solid #4CAF50;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Buttons
    col1, col2, col3, col4,col5, col6= st.columns(6)
    with col1:
        start_button = st.button("Speak")
    with col6:
        stop_button = st.button("Interrupt")

    container = st.container()

    if stop_button:
        st.session_state.running = False
        st.success("Interrupted successfully")

    if start_button:
        st.session_state.running = True

    if start_button and st.session_state.running:
        audio_data = record_audio()
        query_prompt = transcribe_audio(audio_data)
        if query_prompt:
            if 'vs' in st.session_state:
                vector_store = st.session_state.vs 

                # Add user message to session state
                st.session_state.messages.append({"role": "user", "content": query_prompt})

                # Display user message immediately
                message(query_prompt, is_user=True, key=f"user_{len(st.session_state.messages)}")

                # Get the answer
                full_answer = ask_and_get_answer(vector_store, query_prompt)

                # Display the full answer at once
                response_placeholder = st.empty()
                response_placeholder.markdown(
                    f'<div style="background-color:#f0f0f0;padding:10px;border-radius:5px;"><span style="color: #20D026;">{full_answer}</span></div>',
                    unsafe_allow_html=True
                )

                # Add assistant message to session state
                st.session_state.messages.append({"role": "assistant", "content": full_answer})
                speak(full_answer)


     # Custom CSS for message emojis
    st.markdown(
        """
        <style>
        .stMessageUser {
            display: flex;
            align-items: center;
        }
        .stMessageUser::before {
            content: "👨‍✈️"; /* Custom emoji for user messages */
            font-size: 1.5em;
            margin-right: 0.5em;
        }
        .stMessageBot {
            display: flex;
            align-items: center;
        }
        .stMessageBot::before {
            content: "🤖"; /* Custom emoji for bot messages */
            font-size: 1.5em;
            margin-right: 0.5em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )