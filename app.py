import streamlit as st
import numpy as np
import faiss
import tempfile
from gtts import gTTS
import whisper
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
# Load environment variables
load_dotenv()
audio_placeholder = st.empty()
# Load the GROQ API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key="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", model_name="llama-3.3-70b-versatile")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Initialize FAISS Index
index = faiss.IndexFlatL2(384)  # Adjust dimension based on model
stored_data = []

# Function to fetch job posting from URL
def fetch_job_posting(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

# Function to generate embeddings for text
def get_embeddings(text):
    """Generate vector embeddings for text"""
    return model([text])[0]

# Function to store embeddings in FAISS index
def store_in_vectordb(text, label):
    """Store text embeddings in FAISS with index mapping"""
    embedding = get_embeddings(text)
    index.add(np.array([embedding], dtype=np.float32))
    stored_data.append((text, label))

# Function to generate interview questions using GROQ API
def generate_interview_questions(job_desc, cv_text):
    """Generate interview questions using GROQ API with error handling"""
    try:
        prompt = f"""Based on the following job description: {job_desc} 
        and the candidate's CV: {cv_text}, generate 5 technical interview questions."""
        response = groq.ChatCompletion.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": "You are an AI interviewer."},
                      {"role": "user", "content": prompt}]
        )
        questions = response["choices"][0]["message"]["content"].split("\n")
        return questions
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []

# Function to convert text to speech using gTTS
def text_to_speech(text):
    """Convert text to speech using gTTS"""
    try:
        tts = gTTS(text)
        filename = tempfile.mktemp(suffix=".mp3")
        tts.save(filename)
        return filename
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return ""

# Function to convert speech to text using Whisper
def speech_to_text():
    """Convert speech to text using Whisper"""
    try:
        model = whisper.load_model("small")
        filename = tempfile.mktemp(suffix=".wav")
        st.audio(filename, format='audio/wav', start_time=0)
        result = model.transcribe(filename)
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing speech: {e}")
        return ""

# Recognize speech from audio
def recognize_speech(audio_file=None):
    r = sr.Recognizer()
    if audio_file:
        try:
            with sr.AudioFile(audio_file) as source:
                audio = r.record(source)
                st.write('Detected spech:',r.recognize_google(audio))
            return r.recognize_google(audio)
        except Exception as e:
            print(f"Error recognizing speech: {e}")
    return None

# Function to create vector embeddings
def create_vector_embedding(job_text, cv_text):
    if "vectors" not in st.session_state:
        # Initialize the Ollama Embeddings model
        st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
        
        # Initialize text splitter
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # Split job text and CV text into documents
        job_docs = st.session_state.text_splitter.split_text(job_text)
        cv_docs = st.session_state.text_splitter.split_text(cv_text)
        
        # Combine documents
        combined_docs = job_docs + cv_docs
        
        # Create embeddings and store in FAISS
        st.session_state.vectors = FAISS.from_texts(combined_docs, st.session_state.embeddings)

# Cleanup temporary files after processing
def cleanup_file(file_path):
    try:
        os.remove(file_path)
    except Exception as e:
        st.error(f"Error during cleanup: {e}")

# Initialize session state variables if not present
if 'current_question_index' not in st.session_state:
    st.session_state['current_question_index'] = 0

if 'questions' not in st.session_state:
    st.session_state['questions'] = []

if 'answers' not in st.session_state:
    st.session_state['answers'] = []

if 'interview_started' not in st.session_state:
    st.session_state['interview_started'] = False

# Streamlit UI
st.title("AI-Powered Interview Trainer with Voice Interaction")

# Input for job posting URL
job_url = st.text_input("Enter Job Posting URL:")

# File uploader for CV
cv_file = st.file_uploader("Upload your CV (PDF):", type="pdf")

if st.button("Process Job and CV"):
    if job_url and cv_file:
        # Fetch job posting text
        job_text = fetch_job_posting(job_url)
        
        # Extract text from CV
        with open("temp_cv.pdf", "wb") as f:
            f.write(cv_file.getbuffer())
        cv_loader = PyPDFLoader("temp_cv.pdf")
        cv_docs = cv_loader.load()
        cv_text = " ".join([doc.page_content for doc in cv_docs])
        os.remove("temp_cv.pdf")  # Clean up the temporary file
        
        # Create vector embeddings
        create_vector_embedding(job_text, cv_text)
        st.success("Job and CV processed successfully!")

# Define the prompt template for generating interview questions
interview_prompt = ChatPromptTemplate.from_template(
    """
    Generate exactly 5 interview questions based on the following job description and CV.
    Provide only the questions, one per line, without any additional text or explanations.
    <context>
    {context}
    <context>
    """
)

# Define the prompt template for evaluating answers
evaluation_prompt = ChatPromptTemplate.from_template(
    """
    Evaluate the following interview answers and provide a score out of 10 and recommended answers:
    <context>
    {context}
    <context>
    Questions:
    {questions}
    Answers:
    {answers}
    """
)

# Button to start the interview
if st.button("Start Interview"):
    if "vectors" in st.session_state:
        # Generate interview questions
        document_chain = create_stuff_documents_chain(llm, interview_prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({'input': ''})
        questions = response['answer'].strip().split("\n")  # Access the 'answer' key
        
        # Ensure only 5 questions are displayed
        questions = questions[:5]
        
        st.session_state['questions'] = questions
        st.session_state['answers'] = [""] * len(questions)  # Initialize answers list
        st.session_state['current_question_index'] = 0  # Track the current question index
        st.session_state['interview_started'] = True
    else:
        st.warning("Please process the job and CV first.")

# Sequential interview flow
if st.session_state.get('interview_started', False):
    current_question_index = st.session_state['current_question_index']
    
    # Ensure we haven't finished all questions
    if current_question_index < len(st.session_state['questions']):
        question = st.session_state['questions'][current_question_index]
        
        # Display the current question
        st.write(f"**Question {current_question_index + 1}:** {question}")
        
        # Convert question to speech
        audio_file = text_to_speech(question)
        if audio_file:
            st.audio(audio_file, format="audio/mp3")
        
        # Capture user's answer using voice (audio recorder)
        st.write("Please answer the question by speaking.")
        user_audio = audio_recorder()
        
        if user_audio:
            # Save the audio to a temporary file and process it
            with open("temp_audio.wav", "wb") as f:
                f.write(user_audio)
            
            # Recognize the speech from the audio
            recognized_answer = recognize_speech("temp_audio.wav")
            if recognized_answer:
                # Save the recognized answer in session state
                st.session_state['answers'][current_question_index] = recognized_answer
                st.session_state['current_question_index'] += 1  # Move to the next question
                st.success(f"Answer recorded for Question {current_question_index}. Proceeding to the next question...")
            
            # Clean up the temporary audio file
            cleanup_file("temp_audio.wav")
        
    # If all questions are answered, prompt for submission and evaluation
    if st.session_state['current_question_index'] >= len(st.session_state['questions']):
        st.write("### All questions answered!")
        if st.button("Submit Answers for Evaluation"):
            # After all answers are provided, trigger evaluation
            evaluation_chain = create_stuff_documents_chain(llm, evaluation_prompt)
            evaluation_response = evaluation_chain.invoke({
                'input': '',
                'context': st.session_state.vectors.as_retriever().get_relevant_documents(""),
                'questions': "\n".join(st.session_state['questions']),
                'answers': "\n".join(st.session_state['answers'])
            })
            
            st.write("### Evaluation Results")
            st.write(evaluation_response)  # Access the 'answer' key


