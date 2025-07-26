import streamlit as st
from datetime import datetime
import requests
import json
from PIL import Image

# Load logo image
try:
    initial_logo = Image.open("logo.png")  # Ensure logo.png is present in the working directory
except FileNotFoundError:
    st.error("🚨 logo.png not found! Please place the logo.png file in the same folder as this script.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="KAI",
    page_icon=initial_logo,
    layout="centered"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! How can I help you today?",
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }]
if "is_typing" not in st.session_state:
    st.session_state.is_typing = False
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "phi3"
if "available_models" not in st.session_state:
    st.session_state.available_models = []
if "model_selected_manually" not in st.session_state:
    st.session_state.model_selected_manually = False  # Track manual model selection

def get_available_models():
    """
    Fetch list of available models from Ollama.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get("models", []):
                model_name = model.get("name", "").split(":")[0]
                if model_name and model_name not in models:
                    models.append(model_name)
            return sorted(models)
        else:
            return []
    except Exception:
        return []

def detect_task_type(user_message):
    """
    Enhanced task detection with extensive keyword sets for improved intent recognition.
    """
    coding_keywords = [
        "code", "coding", "bug", "debug", "fix", "error", "exception", "syntax", "compile", "runtime", "crash",
        "function", "method", "script", "programming", "algorithm", "implement", "variable", "constant", "loop",
        "array", "list", "dictionary", "object", "class", "inheritance", "polymorphism", "framework", "library",
        "module", "import", "export", "package", "dependency", "build", "ci/cd", "refactor", "merge conflict", "git",
        "version control", "repository", "commit", "push", "pull", "branch", "statement", "line of code", "stack trace",
        "traceback", "configuration", "environment variable", "shell", "terminal", "command line", "bash", "linux",
        "docker", "container", "api", "request", "response", "rest", "graphql", "endpoint", "json", "xml",
        "serialization", "deserialization", "testing", "assert", "unittest", "pytest", "documentation", "comment",
        "ide", "vscode", "pycharm", "intellij", "deploy", "hotfix", "patch", "release", "sprint", "backlog", "ticket",
        "javascript", "typescript", "react", "node", "sql", "database"
    ]

    summarization_keywords = [
        "summarize", "summary", "brief", "abstract", "recap", "key points", "core idea", "essence", "gist", "in short",
        "compress", "shorten", "reduce", "condense", "bullet points", "paraphrase", "rewrite", "reword", "rephrase",
        "concise", "main points"
    ]

    math_keywords = [
        "math", "mathematics", "equation", "solve", "solution", "calculation", "compute", "integral", "derivative",
        "calculus", "statistics", "probability", "mean", "median", "standard deviation", "regression",
        "probability density", "matrix", "determinant", "eigenvalue", "algebra", "geometry", "triangle", "shape",
        "angle", "logic", "proof", "theorem", "induction", "discrete math", "graph theory", "number theory", 
        "optimization", "minimum", "maximum", "function plot", "differentiation", "summation", "variable substitution",
        "mathematical model"
    ]

    writing_keywords = [
        "write", "draft", "generate text", "text generation", "story", "tale", "article", "essay", "blog", "poem",
        "haiku", "limerick", "creative", "plot", "outline", "editorial", "headline", "lead", "introduction", "body",
        "conclusion", "edit", "proofread", "correct grammar", "punctuation", "spelling", "rewrite", "expand",
        "elaborate", "clarify", "tone", "style", "literary", "narrative", "dialogue", "translate", "translation"
    ]

    science_keywords = [
        "science", "scientific", "research", "experiment", "hypothesis", "theory", "observation", "results",
        "conclusion", "abstract", "analysis", "data", "variable", "statistics", "table", "chart", "graph",
        "systematic review", "chemical", "molecule", "reaction", "synthesis", "physics", "optics", "quantum",
        "biology", "cell", "genetics", "engineering", "mechanical", "electrical", "computer science", "algorithmic",
        "biomedical", "peer review", "publication", "citation", "journal", "pubmed", "dataset", "artificial intelligence",
        "neural network", "machine learning"
    ]

    msg = user_message.lower()
    # Safely check for code block presence (triple backticks)
    if "```" in msg:
        return "coding"

    # Priority checks (coding first)
    for kw in coding_keywords:
        if kw in msg:
            return "coding"
    for kw in math_keywords:
        if kw in msg:
            return "math"
    for kw in summarization_keywords:
        if kw in msg:
            return "summarization"
    for kw in writing_keywords:
        if kw in msg:
            return "writing"
    for kw in science_keywords:
        if kw in msg:
            return "science"

    return "general"

def get_best_model_for_task(user_message, available_models):
    """
    Choose the best model based on expanded task detection and rich model lists.
    """
    model_preferences = {
        "coding": [
            "qwen2.5-coder", "qoonscoder", "code-llama", "code-llama-instruct", "starcoder",
            "starcoderplus", "wizardcoder", "phind", "stablecode", "deepseek-coder", "replit code", "llamacoder"
        ],
        "math": [
            "deepseek-math", "wizardmath", "mathcoder", "gpt-neox-math", "phi3", "llama3", 
            "wizardlm", "pythia-math", "galactica", "sciphi"
        ],
        "summarization": [
            "mistral", "mistral-instruct", "falcon", "llama3", "gemma", "mixtral", "vicuna", 
            "yi", "gpt-4all", "openhermes", "orca", "nous hermes"
        ],
        "writing": [
            "writerlm", "poro", "openorca", "guanaco", "dolly", "belle", "alpaca"
        ],
        "science": [
            "galactica", "sciencellama", "sciphi", "biomedlm", "pubmedgpt", "chemcrow"
        ],
        "general": [
            "phi3", "llama3", "mistral", "falcon", "vicuna", "openhermes", "gpt-4all", 
            "alpaca", "dolly", "gemma"
        ]
    }

    task = detect_task_type(user_message)
    preferred_models = model_preferences.get(task, [])
    for pref in preferred_models:
        for model in available_models:
            if pref.lower() in model.lower():
                return model
    # Fallback: first available model or None
    return available_models if available_models else None

def get_bot_response(user_message, model_name):
    """
    Query Ollama local API for response from selected model.
    """
    try:
        ollama_url = "http://localhost:11434/api/generate"
        # Map shorthand model names if needed
        if model_name == "qwen3":
            model_name = "qwen3:0.6b"
        elif model_name == "qwen2.5-coder":
            model_name = "qwen2.5-coder:0.5b"
        elif model_name == "qwen2.5":
            model_name = "qwen2.5:0.5b"

        payload = {
            "model": model_name,
            "prompt": user_message,
            "stream": False
        }
        response = requests.post(
            ollama_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Sorry, I couldn't generate a response.").strip()
        else:
            return f"Error: Ollama API returned status code {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ Error: Cannot connect to Ollama. Please ensure Ollama is running on localhost:11434."
    except requests.exceptions.Timeout:
        return "⏱️ Error: Request timed out. The model might be taking too long to respond."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def format_timestamp(ts_str):
    try:
        dt = datetime.strptime(ts_str, "%H:%M:%S")
        return dt.strftime("%I:%M:%S %p")
    except:
        return ts_str

# UI Rendering
st.image(initial_logo, width=100)
st.markdown("*Ideas start here. Just Ask*")
st.markdown("---")
st.title(f"KAI ({st.session_state.selected_model.upper()})")
st.subheader("🗪 Chat")

for message in st.session_state.messages:
    role = message["role"]
    timestamp = format_timestamp(message.get("timestamp", ""))
    content = message["content"]
    if role == "user":
        sender = "👤 <b>You</b>"
        bg_color = "#31c469"
    else:
        sender = "🤖 <b>KAI</b>"
        bg_color = "#0474ea"
    st.markdown(
        f"""
        <div style="margin-top: 1rem; padding: 10px; background-color: {bg_color}; border-radius: 10px;">
            <div style="margin-bottom: 5px;">{sender} <span style="color: black; font-size: 0.85em;">({timestamp})</span></div>
            <div style="font-size: 1rem;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

if st.session_state.is_typing:
    st.markdown("**KAI** is thinking...")
    st.warning("Typing...")

st.markdown("---")
st.subheader("✒️ Your Message")

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Type your message:",
        placeholder="Ask me anything..."
    )
    send_button = st.form_submit_button("➤ Send Message", type="primary")

col1, col2 = st.columns(2)
with col1:
    clear_button = st.button("🚮 Clear Chat")
with col2:
    export_button = st.button("📤 Export Chat")

if send_button and user_input.strip():
    current_time = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user",
        "content": user_input.strip(),
        "timestamp": current_time
    })
    st.session_state.is_typing = True
    st.rerun()

if st.session_state.is_typing:
    user_message = st.session_state.messages[-1]["content"]

    # Only auto-select best model if no manual model selection is active
    if not st.session_state.model_selected_manually:
        best_model = get_best_model_for_task(user_message, st.session_state.available_models)
        if best_model is not None and best_model != st.session_state.selected_model:
            st.session_state.selected_model = best_model

    bot_response = get_bot_response(user_message, st.session_state.selected_model)
    current_time = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response,
        "timestamp": current_time
    })
    st.session_state.is_typing = False
    st.rerun()

if clear_button:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! How can I help you today?",
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }]
    st.session_state.is_typing = False
    st.session_state.model_selected_manually = False  # Reset manual flag on clear
    st.success("Chat cleared!")
    st.rerun()

if export_button:
    if len(st.session_state.messages) > 1:
        chat_content = "CHATBOT CONVERSATION\n" + "="*50 + "\n\n"
        for msg in st.session_state.messages:
            role = "You" if msg["role"] == "user" else "Bot"
            chat_content += f"[{msg['timestamp']}] {role}: {msg['content']}\n\n"
        st.download_button(
            label="📄 Download Chat History",
            data=chat_content,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    else:
        st.warning("No messages to export yet!")

# Sidebar UI
with st.sidebar:
    st.header("🤖 Model Selection")

    if st.button("🔄 Refresh Models", help="Fetch the latest available models"):
        st.session_state.available_models = get_available_models()
        st.rerun()

    if not st.session_state.available_models:
        st.session_state.available_models = get_available_models()

    # Add "Auto" as the first choice for auto-selection
    model_options = ["Auto"] + st.session_state.available_models

    # Determine which option should be the default selection
    if st.session_state.model_selected_manually:
        # If manually selected, set the index of selected manual model (if present)
        if st.session_state.selected_model in model_options:
            selected_index = model_options.index(st.session_state.selected_model)
        else:
            selected_index = 1  # fallback to first model after "Auto"
    else:
        # If automatic, set "Auto" selected
        selected_index = 0

    selected_model = st.selectbox(
        "Choose Model:",
        options=model_options,
        index=selected_index,
        help="Select which AI model to use or choose Auto for automatic selection"
    )

    # Handle selection behavior
    if selected_model == "Auto" and st.session_state.model_selected_manually:
        # Switch to auto selection
        st.session_state.model_selected_manually = False
        best_model = get_best_model_for_task("", st.session_state.available_models)
        if best_model:
            st.session_state.selected_model = best_model
        st.success(f"✅ Switched to Auto (best model: {st.session_state.selected_model})")
        st.rerun()
    elif selected_model != "Auto" and (selected_model != st.session_state.selected_model or not st.session_state.model_selected_manually):
        # Manual model selection enables manual mode
        st.session_state.selected_model = selected_model
        st.session_state.model_selected_manually = True
        st.success(f"✅ Switched to {selected_model}")
        st.rerun()

    st.info(f"**Current Model:** {st.session_state.selected_model}")

    # Show model size info if available
    try:
        model_info_response = requests.post(
            "http://localhost:11434/api/show",
            json={"name": st.session_state.selected_model},
            timeout=5
        )
        if model_info_response.status_code == 200:
            model_data = model_info_response.json()
            model_size = model_data.get("details", {}).get("parameter_size", "Unknown")
            st.caption(f"Parameters: {model_size}")
    except Exception:
        pass

    st.markdown("---")

    st.header("🔗 Ollama Status")
    try:
        health_check = requests.get("http://localhost:11434/api/tags", timeout=5)
        if health_check.status_code == 200:
            st.success("✅ Ollama is running")
            models_count = len(st.session_state.available_models)
            st.success(f"✅ {models_count} models available")
        else:
            st.error("❌ Ollama not responding")
    except Exception:
        st.error("❌ Ollama not running")
        st.info("Start Ollama with: `ollama serve`")

    st.markdown("---")

    st.header("📊 Chat Stats")
    total_messages = len(st.session_state.messages)
    user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
    bot_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
    st.metric("Total Messages", total_messages)
    st.metric("Your Messages", user_messages)
    st.metric("Bot Messages", bot_messages)

    st.markdown("---")

    st.header("📋 Available Models")
    if st.session_state.available_models:
        for i, model in enumerate(st.session_state.available_models, 1):
            if model == st.session_state.selected_model:
                st.write(f"{i}. **{model}** ← *Current*")
            else:
                st.write(f"{i}. {model}")
    else:
        st.write("No models found")

    st.markdown("---")

    st.header("⚙️ Configuration")
    st.write(f"**Active Model:** {st.session_state.selected_model}")
    st.write("**Timeout:** 60 seconds")

    st.header("✨ Features")
    st.write("✅ Multi-model support")
    st.write("✅ Dynamic model switching")
    st.write("✅ Real-time model info")
    st.write("✅ Auto-refresh models")
    st.write("✅ Connection monitoring")

    st.markdown("---")

    st.markdown("### About")
    st.markdown(
        "This AI-powered chatbot is designed to help you get intelligent answers "
        "through natural, human-like conversations. It assists with tasks, solves problems, "
        "and adapts to your needs — making work, learning, or support faster and easier."
    )
    st.markdown("### Contact")
    st.markdown("[Email Us](mailto:buddhadevkokkiligadda@gmail.com)")

st.markdown("---")
st.markdown("**Instructions:** Type a message and press Enter or click 'Send Message' to chat with KAI")
