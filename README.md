# SpeechVectorAI - Fueling your interview success with voice-powered AI
![alt text](image.png)

# AI-Powered Interview Trainer with Voice Interaction

This project provides an AI-powered interview training system that allows users to practice job interviews using their own CVs and job postings. It generates relevant interview questions based on a job description and CV, and evaluates answers using advanced language models. Additionally, the application integrates voice interaction, enabling users to record and submit spoken answers to interview questions.

## Features
- **Job Posting URL Input**: Fetches job description from the provided URL.
- **CV Upload**: Users can upload their CV (PDF format), and the system will process it to generate interview questions.
- **Voice Interaction**: Users can answer questions by speaking, and the system converts speech to text.
- **Speech to Text**: Uses Whisper and Google Speech Recognition to convert spoken answers into text.
- **Interview Question Generation**: Uses the GROQ API to generate technical interview questions based on the job description and CV.
- **Text to Speech**: Converts interview questions to speech using Google Text-to-Speech (gTTS).
- **FAISS Integration**: Stores document embeddings for fast retrieval of relevant information during the interview process.
- **Real-time Evaluation**: Evaluates answers based on pre-defined criteria and provides feedback.

## Requirements
- Python 3.10 or higher
- Streamlit
- FAISS
- gTTS (Google Text-to-Speech)
- Whisper (for speech-to-text conversion)
- LangChain
- requests
- BeautifulSoup4
- dotenv
- PyPDFLoader (for PDF processing)
- audio_recorder_streamlit (for recording audio)
- SpeechRecognition (for speech recognition)
- Groq API Key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ai-interview-trainer.git
   cd ai-interview-trainer

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Install the required dependencies:
    GROQ_API_KEY=your_groq_api_key

## Usage
1. Run the Streamlit application:
    ```bash
   streamlit run app.py

2. Open the application in your browser, where you will be able to:

    Enter a job posting URL.
    Upload your CV (PDF).
    Generate interview questions and start practicing by speaking your answers.

3. The system will generate interview questions based on your CV and the job description. You will answer the questions verbally, and the system will recognize and store   your answers.

4.Once all questions are answered, you can submit them for evaluation, and the system will provide feedback.


## Libraries and Tools
- Streamlit: For creating the interactive UI.
- LangChain: For handling document processing and generating interview questions.
- FAISS: For vector-based similarity search and storage.
- Whisper: For converting speech to text.
- gTTS: For converting text to speech.
- SpeechRecognition: For recognizing speech from audio files.
- BeautifulSoup: For web scraping job postings.

## License
This project is licensed under the MIT License.