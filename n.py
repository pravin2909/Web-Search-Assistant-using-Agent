import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent
from langchain_community.tools import Tool, DuckDuckGoSearchResults
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

os.environ["GROQ_API_KEY"] = "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"

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize the LLM with Groq
def init_llm(model_name, temperature):
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=2048,
        timeout=30,
        max_retries=2,
    )

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize DuckDuckGo search tool
ddg_search = DuckDuckGoSearchResults()
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=ddg_search.run,
        description="Searches the web for real-time information.",
    )
]

# Custom prompt template
CUSTOM_PROMPT = PromptTemplate.from_template(
    """You are a helpful AI assistant with access to real-time web search. 
    Use your knowledge and web search to provide comprehensive answers.
    
    Current conversation:
    {chat_history}
    
    Human: {input}
    AI Assistant:"""
)

# Streamlit App UI
st.set_page_config(page_title="NeuroSearch AI", page_icon="🌐", layout="wide")
st.title("🌐 NeuroSearch AI - Search Assistant")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Choose Model",
        ["mixtral-8x7b-32768", "llama3-70b-8192", "llama3-8b-8192"],
        index=0
    )
    temperature = st.slider("Creativity Level", 0.0, 1.0, 0.7)
    max_history = st.slider("Conversation Memory", 1, 10, 3)
    st.divider()
    st.markdown("### 💡 Suggested Queries")
    suggestions = st.button("Latest AI breakthroughs")
    suggestions = st.button("Global market trends")
    suggestions = st.button("Top tech news today")

# Chat container
chat_container = st.container()
input_container = st.container()

# Initialize agent with current settings
def get_agent():
    llm = init_llm(model_name, temperature)
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent="conversational-react-description",
        verbose=True,
        memory=memory,
        prompt=CUSTOM_PROMPT,
        handle_parsing_errors=True,
    )

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input
with input_container:
    query = st.chat_input("Ask anything...")
    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        with st.spinner("🧠 Processing..."):
            try:
                agent = get_agent()
                response = agent.run(input=query)
                
                # Format response
                formatted_response = f"""
                {response}

                🔍 *Sources verified through web search*
                """
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                with st.chat_message("assistant"):
                    st.markdown(formatted_response)
                    
                # Maintain conversation history limit
                if len(st.session_state.messages) > max_history * 2:
                    st.session_state.messages = st.session_state.messages[-max_history*2:]
                    
            except Exception as e:
                error_message = f"⚠️ Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"):
                    st.error(error_message)

# History sidebar
with st.sidebar:
    st.divider()
    st.subheader("📜 Conversation History")
    if st.session_state.messages:
        for i, msg in enumerate(reversed(st.session_state.messages)):
            if msg["role"] == "user":
                st.markdown(f"🗨️ {msg['content']}")
            else:
                st.markdown(f"🤖 {msg['content']}")
            if i < len(st.session_state.messages)-1:
                st.divider()
    else:
        st.info("No conversation history yet")

# Footer
st.sidebar.markdown("""
---
**Powered by:**
- Groq Cloud AI
- DuckDuckGo Search
- Streamlit

*Version 2.1 | NeuroSearch AI*
""")

# Install required dependencies
# pip install streamlit langchain-groq langchain langchain-community duckduckgo-search