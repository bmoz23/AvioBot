import pyaudio
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage
)
from streamlit_chat import message
import os
import speech_recognition as sr
import keyboard
import time
import keyboard
import speech_recognition as sr
import pyttsx3
import asyncio

# Ses tanıma için tanımlamalar
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
    from langchain_community.embeddings import OpenAIEmbeddings  # Doğru yolu kontrol edin
    from pinecone import PodSpec

    # Pinecone API anahtarınızı eklemeniz gerekebilir # Pinecone API anahtarınızı buraya ekleyin
    pc= pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')  # OpenAI modelini doğru şekilde belirtin

    pc = pinecone.Pinecone()
    
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
    pc =pinecone.Pinecone()
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
    
    llm = ChatOpenAI(model='gpt-4-turbo-preview', temperature=1)
    if vector_store is None:
        raise ValueError("Vector store is None, cannot proceed with the retrieval.")
    
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    # k equals 3 means that it will return the three most similar chunks to the user's query.
    
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

    answer =chain.run(q)

    return answer

def speak(text):
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv("req.env"), override=True)
    from openai import OpenAI
    openai=OpenAI()
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


if __name__ == "__main__":

    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv("req.env"), override=True)

    st.subheader("BİTES BiAI'a Hoş Geldiniz")

    # Profil fotoğrafınızı yükleyin
    user_photo = "biLogo.png"

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        uploaded_file = st.file_uploader('Verinizi yükleyiniz:', type=['pdf', 'docx', 'txt'])
        index_name = st.text_input("Vektör Veritabanında oluşturmak istediğiniz index ismini yazınız")
        add_data = st.button('Başla')
        
        if uploaded_file and add_data:
            with st.spinner('Dosya okunuyor, işleniyor ve vektör veritabanına aktarılıyor.'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data)
                delete_pinecone_index()
                vector_store = insert_or_fetch_embeddings(index_name, chunks)
                st.session_state.vs = vector_store
                st.success('Dosya Yüklendi, vektör veritabanı kontrol edildi, ve dosya gömüldü.')
    
        # Kullanıcıdan metin alarak chatbot'u çalıştırma
    user_input = st.text_input("Lütfen sorunuzu yazınız:")
    container=st.container(border=True)
    if user_input:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs 

            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Display user message immediately
            message(user_input, is_user=True, key=f"user_{len(st.session_state.messages)}")

            # Placeholder for the assistant's response
            response_placeholder = st.empty()

            # Initialize an empty response
            response = ""

            # Get the answer
            full_answer = ask_and_get_answer(vector_store, user_input)

            # Display the answer character by character
            for char in full_answer:
                response += char
                response_placeholder.markdown(f'<div style="background-color:#f0f0f0;padding:10px;border-radius:5px;"><span style="color: #20D026;">{response}</span></div>', unsafe_allow_html=True)
                time.sleep(0.05)

            # Update the placeholder with the final response
            response_placeholder.markdown(f'<div style="background-color:#f0f0f0;padding:10px;border-radius:5px;"><span style="color: #20D026;">{response}</span></div>', unsafe_allow_html=True)
                
            # Add assistant message to session state
            st.session_state.messages.append({"role": "assistant", "content": response})
            #speak(response)