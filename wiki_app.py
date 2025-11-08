import streamlit as st
import weaviate
from weaviate.util import generate_uuid5
import utils
from typing import List, Dict

# Initialize Streamlit app with a title
st.title("Presidential Election 2024 Q&A")
st.write("Ask questions about the Wikipedia content in our database!")

# Initialize connection to Weaviate
@st.cache_resource


# Function to perform RAG
def retrieve_and_generate(
    query: str,
    _client,
    num_chunks: int = 3
) -> Dict:
    """
    Retrieve relevant chunks and generate an answer using RAG.
    Args:
        query (str): The user's question
        _client: Weaviate client instance (underscore prefix to prevent Streamlit hashing)
        num_chunks (int): Number of chunks to retrieve
    Returns:
        Dict: Contains generated answer and source information
    """
    # Get the WikiChunk collection
    wiki_chunks = _client.collections.get("WikiChunk")

    # Perform hybrid search
    results = wiki_chunks.query.hybrid(
        query=query,
        limit=num_chunks,
        return_metadata=["distance"],
        return_properties=["title", "chunk", "chunk_number"]
    ).objects

    # Format context from retrieved chunks
    context = "\n\n".join([
        f"From '{obj.properties['title']}' (chunk {obj.properties['chunk_number']}):\n{obj.properties['chunk']}"
        for obj in results
    ])

    # Generate response
    generate_response = wiki_chunks.generate.near_text(
        query=query,
        grouped_task="Please answer the following question based on the context. If the answer cannot be found in the context, say so clearly. Always cite the specific parts of the context you used.\n\nQuestion: " + query
    )

    return {
        "answer": generate_response.generated,  # Correct attribute for generated text
        "sources": [
            {
                "title": obj.properties["title"],
                "chunk_number": obj.properties["chunk_number"],
                "distance": obj.metadata.distance
            }
            for obj in results
        ]
    }

# Initialize the Weaviate client
try:
    client = utils.connect_to_my_db()
    st.success("Successfully connected to Weaviate!")
except Exception as e:
    st.error(f"Failed to connect to Weaviate: {str(e)}")
    st.stop()

# Create the main query input
query = st.text_input("Enter your question:", placeholder="What would you like to know?")

# Add a slider for number of chunks to retrieve
num_chunks = st.slider("Number of chunks to retrieve:", min_value=1, max_value=5, value=3)

# Add a button to trigger the search
if st.button("Get Answer") and query:
    with st.spinner("Searching and generating response..."):
        try:
            # Perform RAG
            result = retrieve_and_generate(query, client, num_chunks)

            # Display the answer
            st.markdown("### Answer")
            st.write(result["answer"])

            # Display the sources
            st.markdown("### Sources Used")
            for source in result["sources"]:
                st.markdown(
                    f"""- **{source['title']}** (Chunk {source['chunk_number']})"""
                        )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)  # Show full traceback for debugging

# Add information sidebar
with st.sidebar:
    st.markdown("""
    ### About this App

    This Q&A system uses Retrieval Augmented Generation (RAG) to provide accurate answers based on the Wikipedia content in our database.

    The process works as follows:
    1. Your question is used to search relevant chunks of text
    2. The most relevant chunks are retrieved
    3. An AI model generates an answer based on these chunks

    ### Tips for better results:
    - Be specific in your questions
    - Adjust the number of chunks if you need more context
    - Check the sources to understand where the information comes from
    """)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

client.close()
