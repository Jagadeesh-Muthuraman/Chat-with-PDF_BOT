import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Updated import for ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import tempfile
import hashlib
import json
from langdetect import detect, LangDetectException
import pycountry
from deep_translator import GoogleTranslator


# Initialize API key variables
groq_api_key = None
google_api_key = None

# Define path for permanent storage
STORAGE_DIR = "pdf_data"
PROCESSED_FILES_LOG = os.path.join(STORAGE_DIR, "processed_files.json")
MASTER_DB_DIR = os.path.join(STORAGE_DIR, "master_db")

# Create storage directory if it doesn't exist
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)
if not os.path.exists(MASTER_DB_DIR):
    os.makedirs(MASTER_DB_DIR)


# Language support utilities
def detect_language(text):
    """Detect the language of the input text"""
    try:
        # Use a longer text sample for more accurate detection
        sample_text = text[:2000] if len(text) > 2000 else text
        lang_code = detect(sample_text)
        language = pycountry.languages.get(alpha_2=lang_code)
        if language:
            return lang_code, language.name
        return lang_code, lang_code
    except LangDetectException:
        return "en", "English"  # Default to English if detection fails


def translate_text(text, target_lang="en", source_lang=None):
    """Translate text to the target language using deep_translator"""
    try:
        if not text.strip():
            return text
            
        # Use deep_translator instead of googletrans
        translator = GoogleTranslator(source=source_lang if source_lang else 'auto', target=target_lang)
        return translator.translate(text)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails


# Function to load the log of processed files
def load_processed_files_log():
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, "r") as f:
            return json.load(f)
    return {"files": []}


# Function to save the log of processed files
def save_processed_files_log(log_data):
    with open(PROCESSED_FILES_LOG, "w") as f:
        json.dump(log_data, f)


# Function to generate a hash for uploaded files to use as an identifier
def generate_files_hash(uploaded_files):
    """Generate a unique hash based on filenames and modification times"""
    hasher = hashlib.md5()
    for file in uploaded_files:
        hasher.update(file.name.encode())
        # Also use file size as part of the hash
        hasher.update(str(file.size).encode())
    return hasher.hexdigest()


# Function to check if files have been processed before
def is_file_processed(file_hash):
    log_data = load_processed_files_log()
    return file_hash in [entry["hash"] for entry in log_data["files"]]


# Function to save master vector database to disk
def save_master_db(vector_db):
    """Save the master vector database to disk"""
    # Save the vector store using FAISS's native method
    vector_db.save_local(MASTER_DB_DIR)
    return MASTER_DB_DIR


# Function to load master vector database from disk
def load_master_db(embeddings):
    """Load the master vector database from disk"""
    if os.path.exists(MASTER_DB_DIR) and os.listdir(MASTER_DB_DIR):
        try:
            # Load using FAISS's native method with allow_dangerous_deserialization=True
            vectors = FAISS.load_local(MASTER_DB_DIR, embeddings, allow_dangerous_deserialization=True)
            return vectors
        except Exception as e:
            st.sidebar.error(f"Error loading master database: {e}")
    return None


# Function to process new documents and update master DB
def process_documents(uploaded_files, embeddings):
    """Process uploaded documents and merge with existing data"""
    files_hash = generate_files_hash(uploaded_files)
    
    # Check if these exact files have been processed before
    if is_file_processed(files_hash):
        st.sidebar.info("These files have already been processed! Using existing data.")
        return load_master_db(embeddings), True
    
    # Process the new files
    all_docs = []
    file_names = []
    document_languages = []
    
    # Show progress bar
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    
    # Process each uploaded file
    for i, uploaded_file in enumerate(uploaded_files):
        file_names.append(uploaded_file.name)
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Load the PDF document
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()  # Load document content

        # Remove the temporary file
        os.remove(temp_file_path)

        # Detect language of the document (using more content for better detection)
        if docs:
            combined_text = " ".join([d.page_content for d in docs[:2]])[:5000]  # Use first 2 pages, up to 5000 chars
            lang_code, lang_name = detect_language(combined_text)
            document_languages.append((uploaded_file.name, lang_code, lang_name))
        
        # Add loaded documents to the list
        all_docs.extend(docs)
        
        # Update progress
        progress_bar.progress((i + 1) / total_files)
    
    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(all_docs)
    
    # Create a vector store for the new documents
    new_vectors = FAISS.from_documents(final_documents, embeddings)
    
    # Try to load existing master database
    master_vectors = load_master_db(embeddings)
    
    if master_vectors:
        # Merge the new vectors with the master vectors
        master_vectors.merge_from(new_vectors)
    else:
        # If no master DB exists yet, use the new vectors as the master
        master_vectors = new_vectors
    
    # Save the updated master database
    save_master_db(master_vectors)
    
    # Update the processed files log
    log_data = load_processed_files_log()
    log_data["files"].append({
        "hash": files_hash,
        "file_names": file_names,
        "date_processed": st.session_state.get("current_date", "Unknown date"),
        "languages": document_languages
    })
    save_processed_files_log(log_data)
    
    # Store detected languages in session state
    if "document_languages" not in st.session_state:
        st.session_state.document_languages = []
    st.session_state.document_languages.extend(document_languages)
    
    # Clear progress bar
    progress_bar.empty()
    
    return master_vectors, False


# Store current date for logging purposes
if "current_date" not in st.session_state:
    from datetime import datetime
    st.session_state.current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize language preferences
if "interface_language" not in st.session_state:
    st.session_state.interface_language = "en"
if "response_language" not in st.session_state:
    st.session_state.response_language = "en"
if "document_languages" not in st.session_state:
    st.session_state.document_languages = []
if "auto_detect_language" not in st.session_state:
    st.session_state.auto_detect_language = True

# Get list of available languages for dropdown
def get_language_options():
    # Common languages first
    common_languages = [
        ("en", "English"),
        ("es", "Spanish"),
        ("fr", "French"),
        ("de", "German"),
        ("zh-cn", "Chinese (Simplified)"),
        ("ja", "Japanese"),
        ("hi", "Hindi"),
        ("ar", "Arabic"),
        ("ru", "Russian"),
        ("pt", "Portuguese"),
    ]
    
    # Extended language list
    extended_languages = [
        ("af", "Afrikaans"),
        ("sq", "Albanian"),
        ("am", "Amharic"),
        ("hy", "Armenian"),
        ("az", "Azerbaijani"),
        ("eu", "Basque"),
        ("be", "Belarusian"),
        ("bn", "Bengali"),
        ("bs", "Bosnian"),
        ("bg", "Bulgarian"),
        ("ca", "Catalan"),
        ("ceb", "Cebuano"),
        ("ny", "Chichewa"),
        ("zh-tw", "Chinese (Traditional)"),
        ("co", "Corsican"),
        ("hr", "Croatian"),
        ("cs", "Czech"),
        ("da", "Danish"),
        ("nl", "Dutch"),
        ("eo", "Esperanto"),
        ("et", "Estonian"),
        ("tl", "Filipino"),
        ("fi", "Finnish"),
        ("fy", "Frisian"),
        ("gl", "Galician"),
        ("ka", "Georgian"),
        ("el", "Greek"),
        ("gu", "Gujarati"),
        ("ht", "Haitian Creole"),
        ("ha", "Hausa"),
        ("haw", "Hawaiian"),
        ("iw", "Hebrew"),
        ("hmn", "Hmong"),
        ("hu", "Hungarian"),
        ("is", "Icelandic"),
        ("ig", "Igbo"),
        ("id", "Indonesian"),
        ("ga", "Irish"),
        ("it", "Italian"),
        ("jw", "Javanese"),
        ("kn", "Kannada"),
        ("kk", "Kazakh"),
        ("km", "Khmer"),
        ("ko", "Korean"),
        ("ku", "Kurdish"),
        ("ky", "Kyrgyz"),
        ("lo", "Lao"),
        ("la", "Latin"),
        ("lv", "Latvian"),
        ("lt", "Lithuanian"),
        ("lb", "Luxembourgish"),
        ("mk", "Macedonian"),
        ("mg", "Malagasy"),
        ("ms", "Malay"),
        ("ml", "Malayalam"),
        ("mt", "Maltese"),
        ("mi", "Maori"),
        ("mr", "Marathi"),
        ("mn", "Mongolian"),
        ("my", "Myanmar (Burmese)"),
        ("ne", "Nepali"),
        ("no", "Norwegian"),
        ("ps", "Pashto"),
        ("fa", "Persian"),
        ("pl", "Polish"),
        ("ro", "Romanian"),
        ("sm", "Samoan"),
        ("gd", "Scots Gaelic"),
        ("sr", "Serbian"),
        ("st", "Sesotho"),
        ("sn", "Shona"),
        ("sd", "Sindhi"),
        ("si", "Sinhala"),
        ("sk", "Slovak"),
        ("sl", "Slovenian"),
        ("so", "Somali"),
        ("su", "Sundanese"),
        ("sw", "Swahili"),
        ("sv", "Swedish"),
        ("tg", "Tajik"),
        ("ta", "Tamil"),
        ("te", "Telugu"),
        ("th", "Thai"),
        ("tr", "Turkish"),
        ("uk", "Ukrainian"),
        ("ur", "Urdu"),
        ("uz", "Uzbek"),
        ("vi", "Vietnamese"),
        ("cy", "Welsh"),
        ("xh", "Xhosa"),
        ("yi", "Yiddish"),
        ("yo", "Yoruba"),
        ("zu", "Zulu"),
    ]
    
    # Combine common and extended languages
    all_languages = common_languages + extended_languages
    
    # Add languages detected in documents
    detected_langs = [(lang[1], lang[2]) for lang in st.session_state.document_languages]
    
    # Combine without duplicates
    for lang_code, lang_name in detected_langs:
        if lang_code not in [l[0] for l in all_languages]:
            all_languages.append((lang_code, lang_name))
    
    return all_languages

# Apply custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E88E5 !important;
        margin-bottom: 1rem !important;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        border-color: #1565C0;
    }
    .stProgress > div > div > div {
        background-color: #1E88E5;
    }
    .sidebar-header {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #1E88E5 !important;  
    }
    .success-message {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
        margin: 10px 0;
    }
    .stSidebar .stButton>button {
        width: 100%;
    }
    div[data-testid="stFileUploadDropzone"] {
        border: 2px dashed #1E88E5;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Improved sidebar layout
with st.sidebar:
    st.markdown('<p class="sidebar-header">🔍 About</p>', unsafe_allow_html=True)
    st.write(
        "Welcome to **Chat with PDF**! Upload documents and ask questions about their content directly."
    )

    st.markdown('<p class="sidebar-header">⚙️ Settings</p>', unsafe_allow_html=True)

    # Language settings section with better organization
    st.markdown("##### 🌐 Language Settings")
    
    # Auto-detect language toggle with better description
    st.session_state.auto_detect_language = st.checkbox(
        "Auto-detect and match question language", 
        value=st.session_state.auto_detect_language,
        help="When enabled, responses will match the language of your questions."
    )
    
    # If auto-detect is disabled, show language selection dropdown
    if not st.session_state.auto_detect_language:
        # Get language options
        language_options = get_language_options()
        
        # Create a dict for the dropdown
        language_dict = {name: code for code, name in language_options}
        
        # Show language selector
        selected_lang_name = st.selectbox(
            "Force responses in this language:",
            options=list(language_dict.keys()),
            index=0  # Default to English
        )
        st.session_state.response_language = language_dict[selected_lang_name]
        
        # Add a note about forced language
        st.info("Questions will still be processed correctly in any language, but responses will always be in the selected language.")
    
    # Show document language information in a cleaner format
    if st.session_state.document_languages:
        with st.expander("📚 Document Languages"):
            st.markdown("#### Detected Languages:")
            for doc_name, lang_code, lang_name in st.session_state.document_languages:
                st.markdown(f"- **{doc_name}**: {lang_name}")
    
    st.markdown("##### 🔑 API Keys")
    st.write(
        "Required API keys:\n"
        "- [Groq API Key](https://console.groq.com/keys)\n"
        "- [Google API Key](https://aistudio.google.com/app/apikey)"
    )

    # Input fields for API keys with better styling
    groq_api_key = st.text_input("Groq API key:", type="password", help="Required for LLM processing")
    google_api_key = st.text_input("Google API key:", type="password", help="Required for embeddings")

    # Clear chat history button with confirmation
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

    # Validate API key inputs and initialize components if valid
    if groq_api_key and google_api_key:
        # Set Google API key as environment variable
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Initialize ChatGroq with the provided Groq API key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

        # Define the chat prompt template with enhanced language forcing
        prompt = ChatPromptTemplate.from_template(
            """
            You are a multilingual AI assistant that answers questions based on document content. Your goal is to maintain a 
            conversational flow and remember previous questions and your answers to them.
            
            When you receive a follow-up question that doesn't explicitly mention the previous topic, assume it's related
            to the previous context. Use the conversation history to understand what the user is asking about.
            
            Current conversation history:
            {chat_history}
            
            Context from documents:
            {context}
            
            Current Question: {input}
            
            Answer the question using information from both the context and chat history.
            If the question is a follow-up, connect it to previous topics naturally.
            If information is not in the context, indicate this clearly.

            CRITICAL INSTRUCTION: You MUST respond ONLY in the language specified in the RESPONSE_LANGUAGE field below. 
            This is a strict requirement.
            
            RESPONSE_LANGUAGE: {response_language}
            """
        )

        # Display a list of processed files if available in a more compact format
        processed_files = load_processed_files_log()
        if processed_files["files"]:
            with st.expander("📚 Processed Documents", expanded=False):
                for idx, file_entry in enumerate(processed_files["files"]):
                    st.markdown(f"**Batch {idx+1}:** {', '.join(file_entry['file_names'])}")
                    st.markdown(f"*{file_entry['date_processed']}*")
                    
                    # Display language information if available in a more compact format
                    if "languages" in file_entry:
                        langs = set(lang for _, _, lang in file_entry["languages"])
                        if langs:
                            st.markdown(f"*Languages: {', '.join(langs)}*")
                    
                    st.markdown("---")

        # File uploader for multiple PDFs with better styling
        st.markdown("##### 📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Select PDF file(s)", type="pdf", accept_multiple_files=True
        )

        # Load existing master database on app startup if not already loaded
        if "vectors" not in st.session_state:
            master_vectors = load_master_db(embeddings)
            if master_vectors:
                st.session_state.vectors = master_vectors
                st.sidebar.success("✅ Previously processed documents loaded")
        
        # Process uploaded PDFs when the button is clicked
        if uploaded_files:
            if st.button("🔍 Process Documents"):
                with st.spinner("Processing documents... Please wait."):
                    master_vectors, already_processed = process_documents(uploaded_files, embeddings)
                    
                    # Store in session state
                    st.session_state.vectors = master_vectors
                    
                    if already_processed:
                        st.sidebar.success("✅ Documents were already in the database!")
                    else:
                        st.sidebar.success("✅ Documents successfully processed!")
    else:
        st.warning("Please enter both API keys to proceed.")

# Main area for chat interface with better styling
st.markdown('<h1 class="main-header">💬 Chat with PDF</h1>', unsafe_allow_html=True)

# Initialize session state for chat messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history with improved styling
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "original_text" in message and message["original_text"] != message["content"]:
            with st.expander("Show original text"):
                st.markdown(message["original_text"])

# Input field for user queries
if human_input := st.chat_input("Ask a question about your documents..."):
    # Detect input language with improved detection
    input_lang_code, input_lang_name = detect_language(human_input)
    
    # Store original input for reference
    original_input = human_input
    
    # Determine target language for response based on settings
    if st.session_state.auto_detect_language:
        target_lang = input_lang_code
        target_lang_name = input_lang_name
    else:
        target_lang = st.session_state.response_language
        language = pycountry.languages.get(alpha_2=target_lang)
        target_lang_name = language.name if language else target_lang
    
    # Add user message to chat history
    user_message = {
        "role": "user", 
        "content": human_input,
        "language": input_lang_code,
        "original_text": original_input
    }
    st.session_state.messages.append(user_message)
    
    with st.chat_message("user"):
        st.markdown(human_input)
    
    # Check if we need to translate for document search
    search_input = human_input
    if input_lang_code != "en":
        # Translate query to English for better vector search
        search_input = translate_text(human_input, target_lang="en", source_lang=input_lang_code)
    
    # Add the current message to the chat history for context
    st.session_state.chat_history.append({"role": "user", "content": search_input})

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        # Create and configure the document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever(
            search_kwargs={"k": 5}  # Retrieve more documents for better context
        )
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Get response from the assistant with chat history context
        with st.spinner("Thinking..."):
            # Format chat history for the prompt
            formatted_history = "\n".join([
                f"{msg['role'].title()}: {msg['content']}" 
                for msg in st.session_state.chat_history[-6:]  # Use last 6 messages for context
            ])
            
            # CRITICAL FIX: Pass the response language explicitly in the prompt variables
            response = retrieval_chain.invoke({
                "input": search_input,
                "chat_history": formatted_history,
                "response_language": target_lang_name  # Pass the actual language name
            })
            
            assistant_response = response["answer"]
            
            # Double-check response language and force translation if needed
            response_lang_code, _ = detect_language(assistant_response)
            
            # ALWAYS verify language match and force translate if needed
            if response_lang_code != target_lang:
                # Store original response for reference
                original_response = assistant_response
                # Force translate to target language
                assistant_response = translate_text(
                    assistant_response, 
                    target_lang=target_lang, 
                    source_lang=response_lang_code
                )
            else:
                original_response = assistant_response

        # Add response to chat history for context in future messages
        st.session_state.chat_history.append({"role": "assistant", "content": original_response})
        
        # Append and display assistant's response
        assistant_message = {
            "role": "assistant", 
            "content": assistant_response,
            "language": target_lang,
            "original_text": original_response if original_response != assistant_response else None
        }
        st.session_state.messages.append(assistant_message)
        
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
            # Show original response if translation occurred
            if original_response != assistant_response:
                with st.expander("Show original response"):
                    st.markdown(original_response)

    else:
        # Prompt user to upload and process documents if no vectors are available, with better styling
        st.warning("⚠️ Please upload and process documents before asking questions.")
        
        assistant_response = (
            "Please upload PDF documents and click 'Process Documents' before asking questions."
        )
        
        # Always translate the message to match the user's language
        if input_lang_code != "en":
            assistant_response = translate_text(
                assistant_response,
                target_lang=input_lang_code,
                source_lang="en"
            )
            
        st.session_state.messages.append({
            "role": "assistant", 
            "content": assistant_response,
            "language": input_lang_code
        })
        
        with st.chat_message("assistant"):
            st.markdown(assistant_response)