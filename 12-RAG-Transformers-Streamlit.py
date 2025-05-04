import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from huggingface_hub import InferenceClient
import os


st.set_page_config(layout="wide")



my_initial_rag_text = f"""This is a RAG (Retrieval-Augmented Generation) chatbot application built with Streamlit that combines document context with LLM responses. Here's a breakdown of its main components:

1. Initial Setup:
- Uses Streamlit for the web interface
- Employs SentenceTransformer for generating embeddings
- Uses HuggingFace's InferenceClient for LLM interaction
- Has a default text about a training module on LLMs

2. State Management:
- Maintains several session state variables for:
  - RAG text (the context document)
  - LLM model selection
  - Chat message history
  - Embeddings and sentences
  - HuggingFace client

3. Interface Layout:
- Split into two columns (1:2 ratio):
  - Left column: Model selection dropdown and RAG text input
  - Right column: Chat interface and message history

4. Core Functionality:
- RAG Implementation:
  - Splits the context document into sentences
  - When a user asks a question, it:
    - Converts the question into embeddings
    - Finds the 3 most relevant sentences from the context
    - Adds these relevant pieces as context to the prompt

- Chat Interface:
  - Streams responses from the LLM
  - Maintains a chat history
  - Shows message history and augmented prompts
  - Uses different avatars for user and AI messages

5. Model Options:
- Offers three LLM choices:
  - Mistral-7B-Instruct-v0.3 (default)
  - Qwen2.5-72B-Instruct
  - Zephyr-7b-beta

The application essentially creates an intelligent chatbot that can answer questions while taking into account the context provided in the RAG text, making it particularly useful for domain-specific Q&A scenarios.

A disclaimer at the bottom reminds users about potential LLM inaccuracies and the need for verification of responses.
"""

# Check if the LLM model is not already in the session state
if "my_llm_model" not in st.session_state:
    # Set the default LLM model to "mistralai/Mistral-7B-Instruct-v0.3"
    st.session_state["my_llm_model"] = "mistralai/Mistral-7B-Instruct-v0.3"
# Check if the SPACE_ID environment variable is not already in the session state
if "my_space" not in st.session_state:
    st.session_state["my_space"] = os.environ.get("SPACE_ID") 

# Function to update the LLM model client
def update_llm_model():
    if st.session_state["my_space"]:
        # Initialize the client with the model if SPACE_ID is available
        st.session_state["client"] = InferenceClient(st.session_state["my_llm_model"])
    else:
        # Initialize the client with the model and token if SPACE_ID is not available
        st.session_state["client"] = InferenceClient(st.session_state["my_llm_model"], token=os.getenv("HF_TOKEN"))
        
# Check if the client is not already in the session state
if "client" not in st.session_state:
    update_llm_model()

# Check if the embeddings model is not already in the session state
if "embeddings_model" not in st.session_state:
    # We will use the all-MiniLM-L6-v2 model for embeddings
    st.session_state["embeddings_model"] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

MAXIMUM_TOKENS = 512

my_system_instructions = "You are a helpful assistant. Be brief and concise. Provide your answers in 100 words or less."

first_message = "Hello, how can I help you today?"

# Check if the chat messages are not already in the session state
if "my_chat_messages" not in st.session_state:
    # Initialize the chat messages list in the session state
    st.session_state["my_chat_messages"] = []
    # Add the system instructions to the chat messages
    st.session_state["my_chat_messages"].append({"role": "system", "content": my_system_instructions})

def delete_chat_messages():
    for key in st.session_state.keys():
        if key != "my_rag_text":
            del st.session_state[key]

augmented_prompt = ""

# Create two columns with a 1:2 ratio
column_1, column_2 = st.columns([1, 2])

# In the first column
with column_1:
    # Display a disclaimer about the potential inaccuracies of Large Language Models
    st.expander("Disclaimer", expanded=False).markdown("""This application and code (hereafter referred to as the 'Software') is a proof of concept at an experimental stage and is not intended to be used as a production environment. The Software is provided as is, wihtout any warranties of any kind, expressed or implied and the user assumes full responsibility for its use, implementation, and legal compliance.
                                                       
The developers of the Software shall not be liable for any damages, losses, claims, or liabilities arising from the Software, including but not limited to the usage of artificial intelligence and machine learning, related errors, third-party tool failures, security breaches, intellectual property violations, legal or regulatory non-compliance, deployment risks, or any indirect, incidental, or consequential damages.
                                                       
Large Language Models may provide wrong answers. Please verify the answers and comply with applicable laws and regulations.
                                                       
The user agrees to indemnify and hold harmless the developers of the Software from any related claims or disputes arising from the utilization of the Software by the user.

By using the Software, you agree to the terms and conditions of the disclaimer.""")
    
    # Add a selectbox for model selection
    st.selectbox("Select the model to use:",
                ["mistralai/Mistral-7B-Instruct-v0.3",
                 "Qwen/Qwen2.5-72B-Instruct", 
                "HuggingFaceH4/zephyr-7b-beta"], 
                key="my_llm_model", on_change=update_llm_model)
    
    # Add a text area for RAG text input
    st.text_area(label="Please enter your RAG text here:", value=my_initial_rag_text, height=500, key="my_rag_text", on_change=delete_chat_messages)

# Check if the sentences are not already in the session state
if "my_sentences" not in st.session_state:
    my_sentences_split = st.session_state["my_rag_text"].split("\n")
    st.session_state["my_sentences"] = []
    for my_sentence in my_sentences_split:
        if my_sentence.strip():
            st.session_state["my_sentences"].append(my_sentence.strip())

# Check if the embeddings are not already in the session state
if "my_embeddings" not in st.session_state:
    st.session_state["my_embeddings"] = st.session_state["embeddings_model"].encode(st.session_state["my_sentences"])
    
with column_2:
    # Create a container for the messages with a specified height
    messages_container = st.container(height=500)
    
    # Display the first message from the assistant
    messages_container.chat_message("ai", avatar=":material/robot_2:").markdown(first_message)
    
    # Iterate through the chat messages stored in the session state
    for message in st.session_state["my_chat_messages"]:
        if message["role"] == "user":
            # Display user messages with a specific avatar - https://fonts.google.com/icons
            messages_container.chat_message(message["role"], avatar=":material/psychology_alt:").markdown(message["content"])
        elif message["role"] == "assistant":
            # Display assistant messages with a specific avatar
            messages_container.chat_message(message["role"], avatar=":material/robot_2:").markdown(message["content"])

    # Check if there is a new prompt from the user
    if prompt := st.chat_input("you may ask here your questions"):
        
        # Encode the user's prompt to get its embedding
        my_question_embedding = st.session_state.embeddings_model.encode([prompt])
        
        # Calculate the cosine similarity between the prompt embedding and stored embeddings
        similarity_to_question = cosine_similarity(my_question_embedding, st.session_state.my_embeddings).flatten()
        
        # Number of sentences to keep based on similarity
        nof_keep_sentences = 10
        
        # Get the indices of the top similar sentences
        sorted_indices = similarity_to_question.argsort()[::-1][:nof_keep_sentences]
        
        # Retrieve the top similar sentences
        sorted_sentences = [st.session_state.my_sentences[i] for i in sorted_indices]
        
        # Construct the augmented prompt with the similar sentences
        augmented_prompt = "Here is the context:"
        for sentence in sorted_sentences:
            augmented_prompt += "\n\n" + 20*"-" + f"\n\n{sentence}"
        augmented_prompt += "\n\n" + 20*"-" + "\n\n" + "The user said:" + f"\n\n{prompt}"

        # Display the user's prompt in the chat container with a specific avatar
        messages_container.chat_message("user", avatar=":material/psychology_alt:").markdown(prompt) 
        # Append the augmented prompt to the chat messages in the session state
        st.session_state["my_chat_messages"].append({"role": "user", "content": augmented_prompt})
        # Create an empty container for the streaming response from the assistant
        with messages_container.chat_message("ai", avatar=":material/robot_2:"):
            response_placeholder = st.empty()
            response = ""
            # Stream the response from the assistant and update the placeholder
            for chunk in st.session_state["client"].chat.completions.create(messages=st.session_state["my_chat_messages"], stream=True, max_tokens=512):
                if chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
                    # Use markdown to update the response placeholder with the streamed content
                    response_placeholder.markdown(response)

        # Remove the last message from the chat messages in the session state
        st.session_state["my_chat_messages"].pop()
        # Append the user's original prompt to the chat messages in the session state
        st.session_state["my_chat_messages"].append({"role": "user", "content": prompt})
        # Append the assistant's response to the chat messages in the session state
        st.session_state["my_chat_messages"].append({"role": "assistant", "content": response})

    
    # Display the chat messages history
    st.write("Messages History:")
    st.json(st.session_state["my_chat_messages"], expanded=False)
    # Display the augmented prompt used for generating the response
    st.write("Augmented prompt:")
    st.json({"augmented_prompt": augmented_prompt}, expanded=False)
