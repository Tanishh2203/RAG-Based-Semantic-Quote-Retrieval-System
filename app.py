import streamlit as st
import pandas as pd
import json
from collections import Counter

def create_streamlit_app():
    st.set_page_config(
        page_title="RAG Quote Retrieval System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("🎯 RAG-Based Semantic Quote Retrieval System")
    st.markdown("Find inspiring quotes using natural language queries!")
    
    # Initialize session state
    if 'rag_pipeline' not in st.session_state:
        with st.spinner("Loading the RAG system... This may take a moment."):
            from rag_system import QuoteDataProcessor, QuoteEmbeddingModel, RAGQuotePipeline
            processor = QuoteDataProcessor()
            if processor.load_data():
                df = processor.preprocess_data()
                embedding_model = QuoteEmbeddingModel()
                model = embedding_model.load_model()
                rag_pipeline = RAGQuotePipeline(model, df)
                rag_pipeline.setup_llm()
                st.session_state.rag_pipeline = rag_pipeline
                st.session_state.df = df
                st.success("✅ RAG system loaded successfully!")
            else:
                st.error("❌ Failed to load the quotes dataset")
                return
    
    st.header("🔍 Query Interface")
    st.subheader("💡 Example Queries:")
    example_queries = [
        "Quotes about insanity attributed to Einstein",
        "Motivational quotes tagged 'accomplishment'",
        "All Oscar Wilde quotes with humor",
        "Quotes about love and relationships",
        "Inspirational quotes by women authors"
    ]
    
    for i, example in enumerate(example_queries):
        if st.button(f"Try: {example}", key=f"example_{i}"):
            st.session_state.user_query = example
    
    user_query = st.text_input(
        "Enter your query:",
        value=st.session_state.get('user_query', ''),
        placeholder="e.g., 'Quotes about courage by famous authors'"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of quotes to retrieve:", 1, 10, 5)
    with col2:
        show_scores = st.checkbox("Show similarity scores", value=True)
    
    if st.button("🚀 Search Quotes", type="primary") and user_query:
        with st.spinner("Searching for relevant quotes..."):
            try:
                result = st.session_state.rag_pipeline.query(user_query, top_k)
                st.header("📋 Results")
                st.subheader("🤖 AI Response")
                st.write(result['response'])
                st.subheader("📚 Retrieved Quotes")
                if result['retrieved_quotes']:
                    for i, quote in enumerate(result['retrieved_quotes'], 1):
                        with st.expander(f"Quote {i} - {quote['author']}" + (f" (Score: {quote['score']:.3f})" if show_scores else "")):
                            st.markdown(f"**Quote:** *\"{quote['quote']}\"*")
                            st.markdown(f"**Author:** {quote['author']}")
                            if quote['tags']:
                                st.markdown(f"**Tags:** {', '.join(quote['tags'])}")
                            if show_scores:
                                st.markdown(f"**Similarity Score:** {quote['score']:.3f}")
                    st.subheader("💾 Download Results")
                    json_str = json.dumps(result, indent=2, default=str)
                    st.download_button(
                        label="Download as JSON",
                        data=json_str,
                        file_name=f"quote_search_results_{user_query[:20]}.json",
                        mime="application/json"
                    )
                else:
                    st.warning("No relevant quotes found for your query.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    if 'df' in st.session_state:
        st.sidebar.header("📊 Dataset Statistics")
        df = st.session_state.df
        st.sidebar.metric("Total Quotes", len(df))
        st.sidebar.metric("Unique Authors", df['author'].nunique())
        st.sidebar.subheader("🏆 Top Authors")
        top_authors = df['author'].value_counts().head(5)
        for author, count in top_authors.items():
            st.sidebar.text(f"{author}: {count} quotes")
        st.sidebar.subheader("🏷️ Popular Tags")
        all_tags = []
        for tags in df['tags']:
            all_tags.extend(tags)
        if all_tags:
            top_tags = Counter(all_tags).most_common(5)
            for tag, count in top_tags:
                st.sidebar.text(f"{tag}: {count}")
            
    st.sidebar.subheader("Visualizations")
    st.sidebar.image("quote_length_distribution.png", caption="Quote Length Distribution")
    st.sidebar.image("top_tags.png", caption="Top 10 Tags")

if __name__ == "__main__":
    create_streamlit_app()