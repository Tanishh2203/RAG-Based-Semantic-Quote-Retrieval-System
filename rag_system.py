import pandas as pd
import numpy as np
from datasets import load_dataset
import json
import re
from typing import List, Dict, Any
import logging
import warnings
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import faiss

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuoteDataProcessor:
    def __init__(self):
        self.dataset = None
        self.processed_data = None
    
    def load_data(self):
        """Load the English quotes dataset from HuggingFace"""
        logger.info("Loading english_quotes dataset...")
        try:
            self.dataset = load_dataset("Abirate/english_quotes")
            logger.info(f"Dataset loaded successfully. Size: {len(self.dataset['train'])}")
            return True
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"]', ' ', text)
        return text
    
    def preprocess_data(self):
        """Preprocess and clean the dataset"""
        logger.info("Preprocessing data...")
        df = pd.DataFrame(self.dataset['train'])
        df['quote'] = df['quote'].apply(self.clean_text)
        df['author'] = df['author'].fillna('Unknown').apply(self.clean_text)
        df['tags'] = df['tags'].apply(lambda x: [tag.strip() for tag in x] if isinstance(x, list) else [])
        df = df[df['quote'].str.len() > 0]
        df['combined_text'] = df.apply(
            lambda row: f"Quote: {row['quote']} | Author: {row['author']} | Tags: {', '.join(row['tags'])}", 
            axis=1
        )
        self.processed_data = df
        logger.info(f"Data preprocessing completed. Final size: {len(df)}")
        return df
    
    def save_processed_data(self, filepath: str = "processed_quotes.json"):
        """Save processed data to JSON"""
        if self.processed_data is not None:
            self.processed_data.to_json(filepath, orient='records', indent=2)
            logger.info(f"Processed data saved to {filepath}")

class QuoteEmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.is_fine_tuned = False
    
    def load_base_model(self):
        """Load the base sentence transformer model"""
        logger.info(f"Loading base model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def prepare_training_data(self, df: pd.DataFrame, sample_size: int = 1000):
        """Prepare training data for fine-tuning"""
        logger.info("Preparing training data for fine-tuning...")
        train_examples = []
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        for _, row in sample_df.iterrows():
            train_examples.append(InputExample(
                texts=[row['quote'], f"Author: {row['author']}"],
                label=1.0
            ))
            for tag in row['tags'][:2]:
                train_examples.append(InputExample(
                    texts=[row['quote'], f"Tag: {tag}"],
                    label=0.8
                ))
        logger.info(f"Created {len(train_examples)} training examples")
        return train_examples
    
    def fine_tune_model(self, df: pd.DataFrame, epochs: int = 1, batch_size: int = 16):
        """Fine-tune the sentence transformer model"""
        if self.model is None:
            self.load_base_model()
        logger.info("Starting model fine-tuning...")
        train_examples = self.prepare_training_data(df)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.CosineSimilarityLoss(self.model)
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            output_path='./fine_tuned_quote_model'
        )
        self.is_fine_tuned = True
        logger.info("Model fine-tuning completed!")
        return self.model
    
    def save_model(self, path: str = "./fine_tuned_quote_model"):
        """Save the fine-tuned model"""
        if self.model:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = "./fine_tuned_quote_model"):
        """Load a fine-tuned model"""
        try:
            self.model = SentenceTransformer(path)
            self.is_fine_tuned = True
            logger.info(f"Fine-tuned model loaded from {path}")
        except:
            logger.info("Fine-tuned model not found, loading base model...")
            self.load_base_model()
        return self.model

class RAGQuotePipeline:
    def __init__(self, model: SentenceTransformer, df: pd.DataFrame):
        self.model = model
        self.df = df
        self.index = None
        self.embeddings = None
        self.llm_client = None
    
    def create_embeddings(self):
        """Create embeddings for all quotes"""
        logger.info("Creating embeddings for all quotes...")
        texts = self.df['combined_text'].tolist()
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
        return self.embeddings
    
    def build_faiss_index(self):
        """Build FAISS index for similarity search"""
        if self.embeddings is None:
            self.create_embeddings()
        logger.info("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
        return self.index
    
    def setup_llm(self, use_ollama: bool = True):
        """Setup LLM for generation"""
        if use_ollama:
            try:
                import ollama
                self.llm_client = ollama.Client()
                logger.info("Ollama client initialized")
            except ImportError:
                logger.error("Ollama not installed. Using fallback response generator")
                self.llm_client = "fallback"
        else:
            self.llm_client = "fallback"
        return self.llm_client
    
    def retrieve_quotes(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant quotes based on query"""
        if self.index is None:
            self.build_faiss_index()
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.df):
                quote_data = self.df.iloc[idx]
                results.append({
                    'rank': i + 1,
                    'score': float(score),
                    'quote': quote_data['quote'],
                    'author': quote_data['author'],
                    'tags': quote_data['tags'],
                    'combined_text': quote_data['combined_text']
                })
        return results
    
    def generate_response(self, query: str, retrieved_quotes: List[Dict]) -> str:
        """Generate response using LLM"""
        context = "\n".join([
            f"Quote {i+1}: \"{quote['quote']}\" - {quote['author']} (Tags: {', '.join(quote['tags'])})"
            for i, quote in enumerate(retrieved_quotes[:3])
        ])
        prompt = f"""Based on the following quotes, provide a comprehensive answer to the user's query.
Query: {query}
Relevant Quotes:
{context}
Please provide a structured response that includes:
1. A brief summary addressing the query
2. The most relevant quotes with their authors
3. Any insights or connections between the quotes
Response:"""
        if self.llm_client and self.llm_client != "fallback":
            try:
                response = self.llm_client.generate(model='llama2', prompt=prompt)
                return response['response']
            except Exception as e:
                logger.error(f"Error with Ollama: {e}")
                return self._fallback_response(query, retrieved_quotes)
        else:
            return self._fallback_response(query, retrieved_quotes)
    
    def _fallback_response(self, query: str, retrieved_quotes: List[Dict]) -> str:
        """Fallback response when LLM is not available"""
        if not retrieved_quotes:
            return "No relevant quotes found for your query."
        response = f"Here are the most relevant quotes for '{query}':\n\n"
        for i, quote in enumerate(retrieved_quotes[:3], 1):
            response += f"{i}. \"{quote['quote']}\"\n"
            response += f"   - Author: {quote['author']}\n"
            if quote['tags']:
                response += f"   - Tags: {', '.join(quote['tags'])}\n"
            response += f"   - Relevance Score: {quote['score']:.3f}\n\n"
        return response
    
    def query(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """Main query function that combines retrieval and generation"""
        logger.info(f"Processing query: {user_query}")
        retrieved_quotes = self.retrieve_quotes(user_query, top_k)
        generated_response = self.generate_response(user_query, retrieved_quotes)
        return {
            'query': user_query,
            'response': generated_response,
            'retrieved_quotes': retrieved_quotes,
            'num_results': len(retrieved_quotes)
        }