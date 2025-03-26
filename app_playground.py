import streamlit as st
import tempfile
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import base64
import time

# Initialize app
def initialize_app():
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    
    defaults = {
        'current_question_index': 0,
        'questions': [],
        'answers': [],
        'interview_started': False,
        'evaluation_complete': False,
        'processing': False,
        'job_analyzed': False,
        'vectors': None,
        'embeddings': None,
        'text_splitter': None,
        'difficulty': "Intermediate",
        'question_count': 5,
        'domain_focus': ["Technical", "Behavioral"],
        'chat_history': [],
        'job_url': "",
        'cv_file': None,
        'spoken_questions': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Enhanced CSS (adjusted for new UI elements)
def load_custom_css():
    st.markdown("""
        <style>
            .main { 
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 20px;
            }
            .sidebar .sidebar-content {
                background: #ffffff;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                padding: 20px;
            }
            .chat-container {
                background: white;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin: 20px 0;
                max-height: 400px;
                overflow-y: auto;
            }
            .chat-message {
                padding: 12px 16px;
                border-radius: 15px;
                margin: 8px 0;
                max-width: 85%;
            }
            .assistant-message {
                background: #e3f2fd;
                border-bottom-left-radius: 4px;
            }
            .user-message {
                background: #d1fae5;
                border-bottom-right-radius: 4px;
                margin-left: auto;
            }
            .question-card {
                background: #fff3e0;
                border-radius: 15px;
                padding: 20px;
                margin: 20px 0;
                border-left: 5px solid #fb8c00;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .control-panel {
                background: white;
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .stButton>button {
                background: linear-gradient(45deg, #4CAF50, #45a049);
                color: white;
                border-radius: 25px;
                padding: 10px 20px;
                border: none;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            .feedback-card {
                background: #fefcbf;
                border-radius: 15px;
                padding: 20px;
                margin: 15px 0;
                border-left: 5px solid #dd6b20;
            }
            .recording-container {
                background: white;
                border-radius: 50%;
                padding: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                position: fixed;
                bottom: 30px;
                right: 30px;
            }
            .spoken-answer {
                background: #f0f4f8;
                border-radius: 15px;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #4CAF50;
            }
        </style>
    """, unsafe_allow_html=True)

# Utility functions
def fetch_job_posting(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'img']):
            element.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return '\n'.join(chunk for chunk in chunks if chunk)
    except:
        return None

def process_cv(cv_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(cv_file.getbuffer())
        tmp_path = tmp_file.name
    try:
        cv_loader = PyPDFLoader(tmp_path)
        cv_docs = cv_loader.load()
        return " ".join(doc.page_content for doc in cv_docs)
    finally:
        os.remove(tmp_path)

def initialize_embeddings(job_text, cv_text):
    if not st.session_state.vectors:
        try:
            st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            job_docs = st.session_state.text_splitter.create_documents([job_text])
            cv_docs = st.session_state.text_splitter.create_documents([cv_text])
            st.session_state.vectors = FAISS.from_documents(
                job_docs + cv_docs, 
                st.session_state.embeddings
            )
            return True
        except:
            return False
    return True

def generate_interview_questions():
    interview_prompt = ChatPromptTemplate.from_template(
        """Generate exactly {count} {difficulty} level interview questions focusing on {domains}.
        Format requirements:
        - Each question must start with "Q:" followed by the question text
        Context: {context}"""
    )
    
    document_chain = create_stuff_documents_chain(
        ChatGroq(model_name="llama3-70b-8192"),
        interview_prompt
    )
    retriever = st.session_state.vectors.as_retriever(k=3)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({
        'input': '',
        'count': st.session_state.question_count,
        'difficulty': st.session_state.difficulty.lower(),
        'domains': ", ".join(st.session_state.domain_focus)
    })
    
    return [line[2:].strip() for line in response['answer'].split('\n') 
            if line.strip().startswith('Q:')][:st.session_state.question_count]

def text_to_speech(text):
    try:
        audio_bytes = BytesIO()
        tts = gTTS(text=text, lang='en', slow=False)
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        audio_data = audio_bytes.getvalue()
        b64 = base64.b64encode(audio_data).decode()
        audio_html = f"""
            <audio id="audio-player" autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error in text-to-speech: {str(e)}")

def recognize_speech(audio_file_path):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = r.record(source)
            text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.warning("Could not understand audio. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"Speech recognition error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error in speech recognition: {str(e)}")
        return None

def evaluate_answers():
    evaluation_prompt = ChatPromptTemplate.from_template(
        """You are an experienced hiring manager evaluating interview responses.
        For each question and answer pair, provide:
        1. A score from 1-10 (10 being best)
        2. Strengths of the answer
        3. Areas for improvement
        4. A sample ideal response
        5. Keywords that should have been included
        
        Format your response as follows for each question:
        ---
        Question: [question text]
        Given Answer: [user's actual answer]
        Score: X/10
        Strengths: [strengths]
        Improvements: [improvements]
        Sample Answer: [sample answer]
        Keywords: [keywords]
        ---
        
        Questions: {questions}
        Answers: {answers}
        Context: {context}"""
    )
    
    try:
        document_chain = create_stuff_documents_chain(
            ChatGroq(model_name="llama3-70b-8192"),
            evaluation_prompt
        )
        retriever = st.session_state.vectors.as_retriever(k=3)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({
            'input': '',
            'questions': "\n".join(f"{i+1}. {q}" for i, q in enumerate(st.session_state.questions)),
            'answers': "\n".join(f"{i+1}. {a}" for i, a in enumerate(st.session_state.answers))
        })
        
        return response['answer']
    except Exception as e:
        st.error(f"Error evaluating answers: {str(e)}")
        return "Error: Could not generate feedback."

# UI Components
def sidebar_controls():
    with st.sidebar:
        #st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.header("🎯 Interview Setup")
        
        st.session_state.job_url = st.text_input(
            "Job Posting URL",
            value=st.session_state.job_url,
            placeholder="https://example.com/job"
        )
        
        st.session_state.cv_file = st.file_uploader(
            "Upload CV (PDF)",
            type="pdf",
            accept_multiple_files=False
        )
        
        if st.button("Analyze JOB & CV"):
            if st.session_state.job_url and st.session_state.cv_file:
                with st.spinner("Analyzing..."):
                    job_text = fetch_job_posting(st.session_state.job_url)
                    if job_text:
                        cv_text = process_cv(st.session_state.cv_file)
                        if initialize_embeddings(job_text, cv_text):
                            st.session_state.job_analyzed = True
                            st.session_state.chat_history = [{
                                "role": "assistant",
                                "content": "Materials analyzed successfully! Ready to begin your interview practice."
                            }]
                            st.success("Ready to start!")
        
        st.markdown("---")
        st.subheader("Customize Interview")
        st.session_state.difficulty = st.selectbox(
            "Difficulty",
            ["Beginner", "Intermediate", "Advanced"],
            index=1
        )
        st.session_state.question_count = st.slider(
            "Questions",
            3, 10, 5
        )
        st.session_state.domain_focus = st.multiselect(
            "Focus Areas",
            ["Technical", "Behavioral", "Situational", "Company Culture"],
            default=["Technical", "Behavioral"]
        )
        
        if st.session_state.job_analyzed and st.button("Start Interview"):
            with st.spinner("Generating questions..."):
                questions = generate_interview_questions()
                if questions:
                    st.session_state.questions = questions
                    st.session_state.answers = [""] * len(questions)
                    st.session_state.interview_started = True
                    st.session_state.evaluation_complete = False
                    st.session_state.spoken_questions = {}
                    st.session_state.current_question_index = 0
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Question 1: {questions[0]}"
                    })
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_chat():
    #st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-bottom: 15px;">💬 AI Powered Interview</h3>', unsafe_allow_html=True)
    
    for msg in st.session_state.chat_history:
        role_class = "assistant-message" if msg["role"] == "assistant" else "user-message"
        st.markdown(
            f'<div class="chat-message {role_class}">'
            f'<strong>{"Interviewer" if msg["role"] == "assistant" else "You"}:</strong> {msg["content"]}'
            '</div>', 
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

def interview_session():
    current_idx = st.session_state.current_question_index
    total_questions = len(st.session_state.questions)
    current_question = st.session_state.questions[current_idx]
    
    if current_idx not in st.session_state.spoken_questions:
        text_to_speech(current_question)
        st.session_state.spoken_questions[current_idx] = True
    
    st.markdown(
        f'<div class="question-card">'
        f'<h3>Question {current_idx + 1}/{total_questions}</h3>'
        f'<p>{current_question}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="recording-container">', unsafe_allow_html=True)
    audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=44100, key=f"rec_{current_idx}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Placeholder for displaying the spoken answer
    spoken_answer_placeholder = st.empty()
    
    if audio_bytes:
        with st.spinner("Processing response..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(audio_bytes)
                tmp_audio_path = tmp_audio.name
            
            recognized_text = recognize_speech(tmp_audio_path)
            
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    os.remove(tmp_audio_path)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                    else:
                        st.warning(f"Could not delete temporary file {tmp_audio_path}.")
                except Exception as e:
                    st.error(f"Error deleting temporary file: {str(e)}")
                    break
            
            if recognized_text:
                # Display the spoken answer immediately
                spoken_answer_placeholder.markdown(
                    f'<div class="spoken-answer">'
                    f'<strong>Your Answer:</strong> {recognized_text}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Save", key=f"save_{current_idx}"):
                        st.session_state.answers[current_idx] = recognized_text
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": recognized_text
                        })
                        st.success("Answer saved!")
                        
                        # Automatically move to the next question or evaluate if it's the last question
                        if current_idx < total_questions - 1:
                            st.session_state.current_question_index += 1
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"Question {current_idx + 2}: {st.session_state.questions[current_idx + 1]}"
                            })
                            st.rerun()
                        else:
                            with st.spinner("Evaluating..."):
                                if not any("Interview complete! Here's your feedback..." in msg["content"] for msg in st.session_state.chat_history):
                                    st.session_state.chat_history.append({
                                        "role": "assistant",
                                        "content": "Interview complete! Here's your feedback..."
                                    })
                                evaluation = evaluate_answers()
                                if evaluation and "Error" not in evaluation:
                                    st.session_state.evaluation = evaluation
                                    st.session_state.evaluation_complete = True
                                    st.session_state.interview_started = False
                                    st.rerun()
                                else:
                                    st.error("Failed to generate feedback. Please try again.")
                with col2:
                    if st.button("🔄 Retry", key=f"retry_{current_idx}"):
                        spoken_answer_placeholder.empty()  # Clear the displayed answer
                        st.rerun()
    
    # Navigation (only Previous and Repeat buttons)
    col1, col2 = st.columns(2)
    with col1:
        if current_idx > 0 and st.button("← Previous"):
            st.session_state.current_question_index -= 1
            st.rerun()
    with col2:
        if st.button("🔊 Repeat"):
            text_to_speech(current_question)

def evaluation_results():
    st.markdown('<h2>📊 Performance Review</h2>', unsafe_allow_html=True)
    
    if not hasattr(st.session_state, 'evaluation') or not st.session_state.evaluation:
        st.error("No evaluation data available. Please complete the interview again.")
        return
    
    sections = st.session_state.evaluation.split('---')
    
    if len(sections) <= 1:
        st.warning("Feedback format is incorrect. Raw evaluation output:")
        st.text(st.session_state.evaluation)
        return
    
    for section in sections:
        if section.strip():
            lines = section.strip().split('\n')
            feedback_dict = {}
            current_key = None
            for line in lines:
                if line.strip():
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                        current_key = key.strip()
                        feedback_dict[current_key] = value.strip()
                    else:
                        if current_key:
                            feedback_dict[current_key] += f"\n{line.strip()}"
            
            st.markdown(
                f'<div class="feedback-card">'
                f'<h4>Question: {feedback_dict.get("Question", "N/A")}</h4>'
                f'<p><strong>Your Answer:</strong> {feedback_dict.get("Given Answer", "N/A")}</p>'
                f'<p><strong>Score:</strong> {feedback_dict.get("Score", "N/A")}</p>'
                f'<p><strong>Strengths:</strong> {feedback_dict.get("Strengths", "N/A")}</p>'
                f'<p><strong>Areas for Improvement:</strong> {feedback_dict.get("Improvements", "N/A")}</p>'
                f'<p><strong>Sample Answer:</strong> {feedback_dict.get("Sample Answer", "N/A")}</p>'
                f'<p><strong>Keywords:</strong> {feedback_dict.get("Keywords", "N/A")}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
    
    if st.button("New Practice Session"):
        st.session_state.interview_started = False
        st.session_state.evaluation_complete = False
        st.session_state.current_question_index = 0
        st.session_state.spoken_questions = {}
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "Ready for another round? Let's begin!"
        }]
        st.rerun()

# Main app
def main():
    initialize_app()
    load_custom_css()
    
    st.title("🤝 AI Interview Coach")
    st.markdown("Master your interview skills with AI-powered practice and feedback")
    
    sidebar_controls()
    display_chat()
    
    if st.session_state.interview_started:
        interview_session()
    elif st.session_state.evaluation_complete:
        evaluation_results()

if __name__ == "__main__":
    main()