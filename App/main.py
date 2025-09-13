import os
import json
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain + Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pinecone

# Import our custom modules
from feedback_utils import update_scores, log_feedback, get_score_stats
from hybrid_retriever import HybridRetriever

# Load environment variables
load_dotenv()

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "gcp-starter")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "vetios-index")
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# FastAPI app with CORS
app = FastAPI(
    title="VetIOS AI Agent", 
    version="1.0",
    description="AI-powered veterinary assistant with hybrid retrieval"
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for services
qa_chain = None
retriever = None
services_initialized = False
initialization_error = None

# Initialize components with error handling
def initialize_services():
    global qa_chain, retriever, services_initialized, initialization_error
    
    try:
        print("🚀 Initializing VetIOS services...")
        
        # Initialize Pinecone
        if not PINECONE_API_KEY:
            raise Exception("PINECONE_API_KEY not found in environment variables")
            
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        print("✅ Pinecone initialized")
        
        # Initialize hybrid retriever
        retriever = HybridRetriever(
            pinecone_index_name=PINECONE_INDEX,
            huggingface_model="sentence-transformers/all-MiniLM-L6-v2",
            pinecone_env=PINECONE_ENV
        )
        print("✅ Hybrid retriever initialized")
        
        # Initialize LLM
        if not HF_API_KEY:
            raise Exception("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
            
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=HF_API_KEY,
            model_kwargs={"temperature": 0.2, "max_new_tokens": 512}
        )
        print("✅ LLM initialized")
        
        # Create enhanced prompt
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are VetIOS, a veterinary medical assistant.\n\n"
                "Based on the following veterinary literature and sources:\n"
                "Context: {context}\n\n"
                "Question: {question}\n\n"
                "Provide a clear, evidence-based answer focused on veterinary medicine. "
                "If the question is not related to veterinary topics, politely redirect "
                "the user to ask about animal health, treatments, or veterinary procedures. "
                "Be concise but comprehensive."
            ),
        )
        
        # Create QA chain with hybrid retriever
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        
        services_initialized = True
        print("🎉 All VetIOS services initialized successfully!")
        
    except Exception as e:
        error_msg = f"Failed to initialize VetIOS services: {str(e)}"
        print(f"❌ {error_msg}")
        initialization_error = error_msg
        services_initialized = False

# Initialize services on startup
initialize_services()

# Pydantic models
class Query(BaseModel):
    question: str

class Feedback(BaseModel):
    question: str
    answer: str
    approved: bool
    sources: list
    user_comment: Optional[str] = None

# API Endpoints
@app.get("/")
def root():
    return {
        "message": "🐾 VetIOS AI Agent is live!", 
        "services_ready": services_initialized,
        "status": "healthy",
        "version": "1.0"
    }

@app.post("/ask")
def ask(query: Query):
    if not services_initialized:
        return {
            "answer": f"⚠️ AI services are currently unavailable. Error: {initialization_error or 'Services not initialized'}",
            "sources": [],
            "error": "services_not_initialized",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        print(f"🔍 Processing question: {query.question[:50]}...")
        
        # Get answer using QA chain with hybrid retrieval
        result = qa_chain({"query": query.question})
        answer = result["result"]
        
        # Process sources with enhanced metadata
        sources = []
        for doc in result.get("source_documents", []):
            source = {
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "url": doc.metadata.get("url", "N/A"),
                "title": doc.metadata.get("title", "Veterinary Resource"),
                "category": doc.metadata.get("category", "general"),
                "species": doc.metadata.get("species", "multiple"),
                "retrieval_method": doc.metadata.get("retrieval_method", "hybrid"),
                "relevance_score": round(doc.metadata.get("composite_score", 0), 3)
            }
            
            # Add DOI if available
            if doc.metadata.get("doi"):
                source["doi"] = doc.metadata["doi"]
            
            sources.append(source)
        
        print(f"✅ Generated answer with {len(sources)} sources")
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieval_info": {
                "total_sources": len(sources),
                "hybrid_retrieval": True,
                "question_length": len(query.question)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "answer": "❌ I encountered an error processing your question. Please try rephrasing or try again later.",
            "sources": [],
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/feedback")
def feedback(fb: Feedback):
    try:
        # Update document scores based on feedback
        updated_scores = update_scores(fb.sources, fb.approved)
        
        # Log detailed feedback
        log_success = log_feedback(
            question=fb.question,
            answer=fb.answer,
            sources=fb.sources,
            approved=fb.approved,
            user_comment=fb.user_comment
        )
        
        return {
            "status": "success",
            "message": "Thank you for your feedback! It helps improve VetIOS.",
            "sources_updated": len(fb.sources),
            "logged": log_success
        }
        
    except Exception as e:
        print(f"Feedback processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process feedback: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if services_initialized else "degraded",
        "services_initialized": services_initialized,
        "initialization_error": initialization_error,
        "components": {
            "pinecone": PINECONE_API_KEY is not None,
            "huggingface": HF_API_KEY is not None,
            "retriever": retriever is not None,
            "qa_chain": qa_chain is not None
        },
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0"
    }

@app.get("/retrieval/health")
def retrieval_health():
    if not retriever:
        return {"status": "error", "message": "Retriever not initialized"}
    
    try:
        health_status = retriever.health_check()
        stats = retriever.get_retrieval_stats()
        
        return {
            "status": "healthy" if services_initialized else "degraded",
            "health": health_status,
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/feedback/stats")
def feedback_stats():
    try:
        stats = get_score_stats()
        return {
            "status": "success",
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Admin endpoint for initializing knowledge base (use once)
@app.post("/admin/initialize-kb")
def initialize_knowledge_base():
    """One-time endpoint to initialize knowledge base with sample data"""
    try:
        from upsert_utils import VeterinaryDocumentUpserter
        
        print("🚀 Starting knowledge base initialization...")
        upserter = VeterinaryDocumentUpserter()
        documents = upserter.prepare_veterinary_documents()
        success = upserter.upsert_documents(documents)
        
        if success:
            return {
                "status": "success", 
                "message": f"Knowledge base initialized with {len(documents)} documents",
                "documents_count": len(documents)
            }
        else:
            return {"status": "failed", "message": "Failed to initialize knowledge base"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Health check endpoint for monitoring
@app.get("/ping")
def ping():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# This is crucial for Render deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting VetIOS API server on port {port}")
    print(f"🔍 Services initialized: {services_initialized}")
    uvicorn.run(app, host="0.0.0.0", port=port)
  
