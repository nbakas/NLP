import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from huggingface_hub import InferenceClient
import os
import numpy as np
from openai import OpenAI





st.set_page_config(layout="wide")


my_initial_rag_text = f"""This is a RAG (Retrieval-Augmented Generation) chatbot application built with Streamlit that combines document context with LLM responses. Here's a breakdown of its main components:

1. Initial Setup:
- Uses Streamlit for the web interface
- Employs SentenceTransformer for generating embeddings
- Uses HuggingFace's InferenceClient for LLM interaction

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
    st.session_state['my_llm_model'] = "mistralai/Mistral-7B-Instruct-v0.3"

# Check if the SPACE_ID environment variable is not already in the session state
if "my_space" not in st.session_state:
    st.session_state['my_space'] = os.environ.get("SPACE_ID") 

# Function to update the LLM model client
def update_llm_model():
    if st.session_state['my_llm_model'].startswith("gemini-"):
        # Initialize the client for gemini models. We use the OpenAI API to interact with gemini models.
        st.session_state['client'] = OpenAI(api_key = os.getenv("GOOGLE_API_KEY"),
                                            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/")
    elif st.session_state['my_llm_model'].startswith("gpt-"):
        # Initialize the client for openai models
        st.session_state['client'] = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
                                            # ,base_url = "https://eu.api.openai.com/" # gives error
    else:
        if st.session_state['my_space']:
            # Initialize the client with the model if SPACE_ID is available
            st.session_state['client'] = InferenceClient(st.session_state['my_llm_model'])
        else:
            # Initialize the client with the model and token if SPACE_ID is not available
            st.session_state['client'] = InferenceClient(st.session_state['my_llm_model'], token=os.getenv("HF_TOKEN"))
        
# Check if the client is not already in the session state
if "client" not in st.session_state:
    update_llm_model()

# Check if the embeddings model is not already in the session state
if "embeddings_model" not in st.session_state:
    # We will use the all-MiniLM-L6-v2 model for embeddings
    st.session_state['embeddings_model'] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

my_system_instructions = "You are a helpful assistant. Be brief and concise. Provide your answers in 100 words or less."

first_message = "Hello, how can I help you today?"

def delete_chat_messages():
    for key in st.session_state.keys():
        if key != "my_rag_text" and key != "my_system_instructions":
            del st.session_state[key]
    update_llm_model()
            

def create_sentences_rag():
    with rag_status_placeholder:
        # The pattern splits text at any of the punctuation marks .?!;: followed by one or more spaces, or at a newline character
        pattern = r'(?<=[.?!;:])\s+|\n'
        st.session_state['my_sentences'] = [sentence.strip() for sentence in re.split(pattern, st.session_state['my_rag_text']) if sentence.strip()]
        with st.spinner(f"Encoding {len(st.session_state['my_sentences'])} sentences..."):
            sentences_ids = [i for i in range(len(st.session_state['my_sentences']))]
            # Rolling window: include partial windows at end
            st.session_state['my_sentences_rag_ids'] = []
            st.session_state['my_sentences_rag'] = []
            for rolling_window_size in range(st.session_state['min_window_size'], st.session_state['max_window_size']+1):
                for i in range(0, len(st.session_state['my_sentences'])-rolling_window_size+1):
                    chunk = " ".join(st.session_state['my_sentences'][i:i+rolling_window_size]).strip()
                    if chunk:   
                        st.session_state['my_sentences_rag'].append(chunk)
                        st.session_state['my_sentences_rag_ids'].append(sentences_ids[i:i+rolling_window_size])
                        # print(f"*****{chunk}*****\n")

            st.session_state['my_embeddings'] = st.session_state['embeddings_model'].encode(st.session_state['my_sentences_rag'])
        st.success(f"{len(st.session_state['my_sentences_rag'])} chunks have been encoded!")




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
    model_list_all = [  'mistralai/Mistral-7B-Instruct-v0.3',
                        'Qwen/Qwen2.5-72B-Instruct', 
                        'HuggingFaceH4/zephyr-7b-beta']
    if os.getenv("GOOGLE_API_KEY"):
        model_list_all.append('gemini-2.5-flash-preview-05-20')
    if os.getenv("OPENAI_API_KEY"):
        model_list_all.append('gpt-4.1-nano-2025-04-14')
    st.selectbox("Select the model to use:",
                model_list_all, 
                key="my_llm_model", 
                on_change=update_llm_model)
    
    # Add a text are for the system instructions
    st.text_area(label="Please enter your system instructions here:", value=my_system_instructions, height=80, key="my_system_instructions", on_change=delete_chat_messages)
    
    # Placeholder right after text_area
    rag_status_placeholder = st.empty()
    # Add a text area for RAG text input
    st.text_area(label="Please enter your RAG text here:", value=my_initial_rag_text, height=200, key="my_rag_text", on_change=delete_chat_messages)

    # Add a slider for minimum window size
    st.slider("Minimum window size in original sentences", min_value=1, max_value=20, value=5, step=1, key="min_window_size", on_change=create_sentences_rag)

    # Add a slider for maximum window size
    st.slider("Maximum window size in original sentences", min_value=1, max_value=20, value=10, step=1, key="max_window_size", on_change=create_sentences_rag)
    
    # Add a slider for the similarity threshold
    st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="my_similarity_threshold")

    # Add a slider for the number of sentences to keep
    st.slider("Number of original chunks to keep", min_value=1, max_value=50, value=20, step=1, key="nof_keep_sentences")

    # Add a slider for the number of minimum sub prompts
    st.slider("Minimum number of words in sub prompt split", min_value=1, max_value=10, value=1, step=1, key="nof_min_sub_prompts")

    # Add a slider for the number of maximum sub prompts
    st.slider("Maximum number of words in sub prompt split", min_value=1, max_value=10, value=5, step=1, key="nof_max_sub_prompts")



# Check if the chat messages are not already in the session state
if "my_chat_messages" not in st.session_state:
    # Initialize the chat messages list in the session state
    st.session_state['my_chat_messages'] = []
    # Add the system instructions to the chat messages
    st.session_state['my_chat_messages'].append({"role": "system", "content": st.session_state['my_system_instructions']})


# print(100*"-")
# Check if the sentences are not already in the session state
if "my_sentences_rag" not in st.session_state:
    create_sentences_rag()




with column_2:
    # Create a container for the messages with a specified height
    messages_container = st.container(height=500)
    
    # Display the first message from the assistant
    messages_container.chat_message("ai", avatar=":material/robot_2:").markdown(first_message)
    
    # Iterate through the chat messages stored in the session state
    for message in st.session_state['my_chat_messages']:
        if message['role'] == "user":
            # Display user messages with a specific avatar - https://fonts.google.com/icons
            messages_container.chat_message(message['role'], avatar=":material/psychology_alt:").markdown(message['content'])
        elif message['role'] == "assistant":
            # Display assistant messages with a specific avatar
            messages_container.chat_message(message['role'], avatar=":material/robot_2:").markdown(message['content'])

    # Check if there is a new prompt from the user
    if prompt := st.chat_input("you may ask here your questions"):
        
        # Split the prompt into words
        split_prompt = prompt.split(" ")
        all_sub_prompts = []
        # Generate sub-prompts based on the specified range
        for jj in range(st.session_state['nof_min_sub_prompts'], st.session_state['nof_max_sub_prompts']+1):
            for ii in range(len(split_prompt)):
                # Create sub-prompt by joining words
                i_split = " ".join(split_prompt[ii:ii+jj]).strip()
                if i_split:
                    all_sub_prompts.append(i_split)
        
        similarities_to_question = np.zeros(len(st.session_state['my_embeddings']))
        for sub_prompt in all_sub_prompts:
            # Encode the user's prompt to get its embedding
            my_question_embedding = st.session_state.embeddings_model.encode([sub_prompt])
            
            # Calculate the cosine similarity between the prompt embedding and stored embeddings
            similarities_to_question += cosine_similarity(my_question_embedding, st.session_state['my_embeddings']).flatten()
        similarities_to_question /= len(all_sub_prompts)
        
        # Get the indices of the top similar sentences
        bottom_col1, bottom_col2 = st.columns([1, 1])
        sorted_indices_rag = similarities_to_question.argsort()[::-1]
        sorted_indices_sentences = []
        max_similarity = 0
        # for irag in range(st.session_state['nof_keep_sentences']):
        irag = 0
        while len(set(sorted_indices_sentences))<st.session_state['nof_keep_sentences'] and irag<len(sorted_indices_rag):
            sorted_indices_sentences.extend(st.session_state['my_sentences_rag_ids'][sorted_indices_rag[irag]])
            max_similarity = max(max_similarity, similarities_to_question[sorted_indices_rag[irag]])
            with bottom_col1:
                str_conf = f"Confidence: {similarities_to_question[sorted_indices_rag[irag]]:.5f}, Sentences IDs: {st.session_state['my_sentences_rag_ids'][sorted_indices_rag[irag]]}"
                with st.expander(f"Chunk: {str(irag+1)} {str_conf}"):
                    for idx in st.session_state['my_sentences_rag_ids'][sorted_indices_rag[irag]]:
                        st.write(f"{st.session_state['my_sentences'][idx]}")
            irag += 1

        sorted_indices_sentences = sorted(list(set(sorted_indices_sentences)))

        # Display the user's prompt in the chat container with a specific avatar
        messages_container.chat_message("user", avatar=":material/psychology_alt:").markdown(prompt) 
        
        # Create an empty container for the streaming response from the assistant
        with messages_container.chat_message("ai", avatar=":material/robot_2:"):
            response_placeholder = st.empty()
            if max_similarity > st.session_state['my_similarity_threshold']:
                # Construct the augmented prompt with the similar sentences
                augmented_prompt = "This is my context:" + "\n\n" + 20*"-" + "\n\n" 
                augmented_prompt += "\n".join([st.session_state['my_sentences'][idx] for idx in sorted_indices_sentences])
                augmented_prompt += "\n\n" + 20*"-" + "\n\n" + "If the above context is not relevant to the prompt, ignore the context and reply based only on the prompt."
                augmented_prompt += "\n\n" + 20*"-" + "\n\n" + "If the above context is relevant to the prompt, reply based on the context and the prompt."
                augmented_prompt += "\n\n" + 20*"-" + "\n\n" + "The prompt is:"
                augmented_prompt += "\n\n" + f"\n\n{prompt}"
                # Append the augmented prompt to the chat messages in the session state
                st.session_state['my_chat_messages'].append({"role": "user", "content": augmented_prompt})
                # Stream the response from the assistant and update the placeholder
                response = ""
                for chunk in st.session_state['client'].chat.completions.create(messages = st.session_state['my_chat_messages'], 
                                                                                model = st.session_state['my_llm_model'],
                                                                                stream = True, 
                                                                                max_tokens = 1024):
                    if chunk.choices[0].delta.content:
                        response += chunk.choices[0].delta.content
                        # Use markdown to update the response placeholder with the streamed content
                        response_placeholder.markdown(response)
                # Remove the last message from the chat messages in the session state
                st.session_state['my_chat_messages'].pop()
            else:
                augmented_prompt = ""
                response = f"I do not have enough information to reply. The maximum similarity found in the context is: {100*max_similarity:.2f}%."
                response_placeholder.markdown(response)

        # Append the user's original prompt to the chat messages in the session state
        st.session_state['my_chat_messages'].append({"role": "user", "content": prompt})
        # Append the assistant's response to the chat messages in the session state
        st.session_state['my_chat_messages'].append({"role": "assistant", "content": response})


        if len(st.session_state['my_chat_messages'])>10:
            # Keep the first message which is the system instructions, remove the 2nd and 3rd messages which are the first user and assistant messages
            st.session_state['my_chat_messages'] = st.session_state['my_chat_messages'][:1] + st.session_state['my_chat_messages'][3:]
    
            

        with bottom_col2:
            # Display the augmented prompt used for generating the response
            st.write("Augmented prompt:")
            st.json({"max_similarity": max_similarity, "augmented_prompt": augmented_prompt}, expanded=False)

            # Display the chat messages history
            st.write("Messages History All:")
            st.json(st.session_state['my_chat_messages'], expanded=False)

    
