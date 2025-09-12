# 🐾 VetIOS AI Agent

VetIOS is an intelligent veterinary AI assistant that provides expert guidance on animal health, treatments, and veterinary procedures. Built with FastAPI backend, React frontend, and powered by hybrid retrieval using Pinecone vector database, BM25 lexical search, and Hugging Face models.

---

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React UI      │───▶│  FastAPI API    │───▶│  Pinecone DB    │
│   (Frontend)    │    │   (Backend)     │    │ (Vector Store)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       ▲
                                ▼                       │
                       ┌─────────────────┐              │
                       │ Hugging Face    │              │
                       │    Models       │              │
                       └─────────────────┘              │
                                │                       │
                                ▼                       │
                       ┌─────────────────┐              │
                       │ Hybrid Retrieval│──────────────┘
                       │  BM25 + Vector  │
                       └─────────────────┘
```

---

## 🔧 Prerequisites

### Backend Requirements
- **Python 3.8+** (Render uses 3.8 by default)
- **Pinecone account** with API key and index
- **Hugging Face Hub account** with API token
- **Git** for version control

### Frontend Requirements
- **Node.js 16+** 
- **npm 8+** or yarn

---

## ⚙️ Environment Variables

### Backend Configuration
Create a `.env` file in your backend directory (copy from `.env.example`):

```bash
# Vector Database - Pinecone
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENV=gcp-starter
PINECONE_INDEX=vetios-index

# AI Models - Hugging Face
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
PORT=8000

# Data Sources (Optional)
NCBI_EMAIL=your-email@example.com

# AI Model Settings
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
AI_TEMPERATURE=0.2
MAX_TOKENS=512
```

### Frontend Configuration
Create a `.env.local` file in your frontend directory:

```bash
# Backend API Connection
VITE_BACKEND_URL=http://localhost:8000

# For production deployment
# VITE_BACKEND_URL=https://your-backend-name.onrender.com
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/vetios.git
cd vetios
```

### 2. Backend Setup
```bash
# Navigate to backend
cd "VetIOS App"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Initialize knowledge base (first time only)
python run_upsert.py

# Start the backend server
python main.py
```

Backend will be available at `http://localhost:8000`

### 3. Frontend Setup
```bash
# Navigate to frontend (in a new terminal)
cd ui

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local
# Edit .env.local with your backend URL

# Start development server
npm run dev  # For Vite
# OR
npm start    # For Create React App
```

Frontend will be available at `http://localhost:3000`

---

## 🌐 Production Deployment (Render)

### Backend Deployment

1. **Create Web Service on Render**
   - Connect your GitHub repository
   - Choose "Web Service"

2. **Configure Service Settings**
   ```
   Name: vetios-backend
   Environment: Python 3
   Root Directory: VetIOS App
   Build Command: pip install -r requirements.txt
   Start Command: python main.py
   ```

3. **Add Environment Variables**
   ```
   PINECONE_API_KEY=your-actual-key
   PINECONE_ENV=gcp-starter
   PINECONE_INDEX=vetios-index
   HUGGINGFACEHUB_API_TOKEN=your-actual-token
   NCBI_EMAIL=your-email@example.com
   ```

4. **Deploy and Note the URL** (e.g., `https://vetios-backend.onrender.com`)

### Frontend Deployment

1. **Create Static Site on Render**
   - Connect your GitHub repository
   - Choose "Static Site"

2. **Configure Site Settings**
   ```
   Name: vetios-frontend
   Root Directory: ui
   Build Command: npm install && npm run build
   Publish Directory: dist (Vite) or build (CRA)
   ```

3. **Add Environment Variables**
   ```
   VITE_BACKEND_URL=https://vetios-backend.onrender.com
   ```

4. **Deploy the Frontend**

---

## 📁 Project Structure

```
vetios/
├── VetIOS App/                 # 🐍 Backend (FastAPI)
│   ├── main.py                 # FastAPI application
│   ├── hybrid_retriever.py     # Hybrid search system
│   ├── feedback_utils.py       # User feedback processing
│   ├── upsert_utils.py         # Knowledge base setup
│   ├── run_upsert.py          # Quick setup script
│   ├── requirements.txt        # Python dependencies
│   ├── corpus.jsonl           # Document corpus (optional)
│   └── .env.example           # Environment template
├── ui/                        # ⚛️ Frontend (React)
│   ├── index.html             # HTML template
│   ├── vite.config.js         # Vite configuration
│   ├── package.json           # Node.js dependencies
│   ├── .env.example           # Environment template
│   └── src/
│       ├── index.js           # React entry point
│       ├── App.js             # Main App component
│       ├── index.css          # Global styles
│       └── components/
│           └── ChatBox.js     # Chat interface
└── README.md                  # This documentation
```

---

## 🔗 API Reference

### Core Endpoints

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|--------------|
| `GET` | `/` | Service status and health | None |
| `POST` | `/ask` | Submit veterinary question | `{"question": "string"}` |
| `POST` | `/feedback` | Submit user feedback | `{"question": "string", "answer": "string", "approved": boolean, "sources": []}` |
| `GET` | `/health` | Detailed health check | None |
| `GET` | `/retrieval/health` | Retrieval system status | None |
| `GET` | `/feedback/stats` | Feedback statistics | None |

### Example Usage

```javascript
// Ask a veterinary question
const response = await fetch('https://your-backend.onrender.com/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ 
    question: "What are the symptoms of kennel cough in dogs?" 
  })
});

const data = await response.json();
console.log(data.answer);
console.log(data.sources);
```

### Response Format

```json
{
  "answer": "Kennel cough symptoms include...",
  "sources": [
    {
      "title": "Canine Respiratory Diseases",
      "content": "Excerpt from source...",
      "url": "https://source-url.com",
      "retrieval_method": "hybrid",
      "relevance_score": 0.95
    }
  ],
  "retrieval_info": {
    "total_sources": 3,
    "hybrid_retrieval": true,
    "components_used": {
      "bm25": true,
      "semantic": true
    }
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

## 🛠️ Technology Stack

### Backend
- **FastAPI** - High-performance web framework
- **LangChain** - LLM application framework
- **Pinecone** - Vector database for semantic search
- **BM25** - Lexical search for keyword matching
- **Hugging Face** - AI models and embeddings
- **Uvicorn** - ASGI server

### Frontend
- **React 18** - UI library with hooks
- **Vite** - Fast build tool and dev server
- **CSS3** - Modern styling with gradients and animations

### AI & Search
- **Hybrid Retrieval** - Combines semantic and lexical search
- **User Feedback Loop** - Learns from user interactions
- **Document Chunking** - Optimized text processing

---

## 🧪 Testing & Development

### Run Tests
```bash
# Backend tests
cd "VetIOS App"
python -m pytest tests/  # If you add tests

# Frontend tests
cd ui
npm test
```

### Development Commands
```bash
# Backend development with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend development
npm run dev  # Vite
npm start    # Create React App

# Build for production
npm run build
```

### Health Checks
```bash
# Check backend health
curl http://localhost:8000/health

# Check retrieval system
curl http://localhost:8000/retrieval/health
```

---

## 📊 Features

### Core Features
- ✅ **AI-Powered Responses** - Contextual veterinary advice using LLMs
- ✅ **Hybrid Search** - Combines semantic (Pinecone) and lexical (BM25) search
- ✅ **Source Citations** - Transparent information sourcing with links
- ✅ **Real-time Chat** - Interactive conversation interface
- ✅ **User Feedback** - Learn from user interactions to improve results

### UI/UX Features
- ✅ **Responsive Design** - Works on desktop and mobile devices
- ✅ **Connection Status** - Visual indicator of backend connectivity  
- ✅ **Error Handling** - Graceful degradation and user-friendly messages
- ✅ **Loading States** - Clear feedback during processing
- ✅ **Auto-scroll** - Automatic scrolling to new messages

### Technical Features
- ✅ **Production Ready** - Comprehensive error handling and logging
- ✅ **Scalable Architecture** - Modular components for easy expansion
- ✅ **Environment Flexibility** - Easy configuration for different deployments
- ✅ **Performance Optimized** - Efficient document retrieval and processing

---

## 🚨 Troubleshooting

### Common Backend Issues

**Services Not Initialized**
```bash
# Check environment variables
python -c "import os; print('Pinecone:', os.getenv('PINECONE_API_KEY', 'Not Set'))"

# Check Pinecone connection
python -c "import pinecone; pinecone.init(api_key='your-key', environment='gcp-starter'); print(pinecone.list_indexes())"
```

**Knowledge Base Empty**
```bash
# Run the upsert script to populate
python run_upsert.py

# Check index stats
curl http://localhost:8000/retrieval/health
```

### Common Frontend Issues

**Can't Connect to Backend**
- Verify `VITE_BACKEND_URL` is set correctly
- Check backend is running and accessible
- Ensure CORS is enabled (included in enhanced main.py)

**Build Errors**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Check for TypeScript errors
npm run build
```

### Render Deployment Issues

**Backend Deployment Fails**
- Check build logs for missing dependencies
- Verify environment variables are set
- Ensure Python version compatibility

**Frontend Build Fails**
- Check Node.js version (should be 16+)
- Verify build command matches your setup
- Check environment variables are set

---

## 🎯 Roadmap

### Phase 1 (Current)
- [x] Basic chat interface
- [x] Hybrid retrieval system
- [x] User feedback collection
- [x] Production deployment

### Phase 2 (Planned)
- [ ] User authentication system
- [ ] Chat history persistence
- [ ] Advanced veterinary specializations
- [ ] Multi-language support

### Phase 3 (Future)
- [ ] Mobile app development
- [ ] Integration with veterinary databases
- [ ] Real-time collaborative features
- [ ] Advanced analytics dashboard

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes and test thoroughly**
4. **Commit with clear messages**
   ```bash
   git commit -m "Add amazing feature for veterinary diagnostics"
   ```
5. **Push and create a Pull Request**

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint/Prettier for JavaScript
- Add tests for new features
- Update documentation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/your-username/vetios/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/vetios/discussions)
- **Documentation**: This README and inline code comments
- **Email**: support@vetios.com

---

## 🙏 Acknowledgments

- **Pinecone** for vector database technology
- **Hugging Face** for AI models and transformers
- **LangChain** for LLM application framework
- **FastAPI** for the excellent web framework
- **React** for the frontend framework
- **Veterinary Community** for domain expertise

---

## ⭐ Star History

If you find VetIOS helpful, please consider giving it a star on GitHub!

---

**Made with ❤️ for veterinary professionals and pet owners worldwide**

*Empowering veterinary care through AI-powered assistance*
