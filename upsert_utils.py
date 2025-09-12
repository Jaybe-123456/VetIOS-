import os
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "gcp-starter")
INDEX_NAME = os.getenv("PINECONE_INDEX", "vetios-index")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

class VeterinaryDocumentUpserter:
    """Enhanced document upserter for VetIOS knowledge base."""
    
    def __init__(self, index_name: str = INDEX_NAME):
        self.index_name = index_name
        self.embeddings = None
        self.vectorstore = None
        self.text_splitter = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embeddings, Pinecone, and text splitter."""
        try:
            # Initialize Pinecone
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
            logger.info("✅ Pinecone initialized successfully")
            
            # Create embeddings model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✅ Embeddings model loaded")
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            logger.info(f"✅ Text splitter initialized (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    def create_index_if_not_exists(self, dimension: int = 384):
        """Create Pinecone index if it doesn't exist."""
        try:
            existing_indexes = pinecone.list_indexes()
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=dimension,  # all-MiniLM-L6-v2 produces 384-dim embeddings
                    metric="cosine",
                    metadata_config={"indexed": ["source", "category", "species", "urgency"]}
                )
                logger.info(f"✅ Created index '{self.index_name}'")
                
                # Wait for index to be ready
                import time
                time.sleep(10)
            else:
                logger.info(f"✅ Index '{self.index_name}' already exists")
                
        except Exception as e:
            logger.error(f"❌ Error creating index: {e}")
            raise
    
    def prepare_veterinary_documents(self) -> List[Document]:
        """Prepare comprehensive veterinary documents."""
        sample_docs = [
            {
                "title": "Canine Parvovirus (CPV)",
                "text": "Canine parvovirus (CPV) is a highly contagious viral disease affecting dogs, particularly puppies under 6 months old. The virus attacks rapidly dividing cells, primarily in the intestines and bone marrow. Clinical signs include severe vomiting, bloody diarrhea (often with a characteristic foul odor), lethargy, dehydration, and fever. The disease has a high mortality rate if left untreated. Diagnosis is typically made using ELISA tests or PCR. Treatment focuses on supportive care including IV fluid therapy, anti-nausea medication (maropitant), antibiotics to prevent secondary bacterial infections, and nutritional support. Prevention through vaccination is highly effective, with the core vaccine series starting at 6-8 weeks of age.",
                "category": "viral_diseases",
                "species": "canine",
                "source": "veterinary_manual",
                "urgency": "high"
            },
            {
                "title": "Feline Leukemia Virus (FeLV)",
                "text": "Feline leukemia virus (FeLV) is a retrovirus that affects cats worldwide and is one of the most important infectious diseases of cats. The virus suppresses the immune system and can cause various cancers, blood disorders, and other diseases. FeLV is transmitted through close contact between cats, including grooming, sharing food/water bowls, and bite wounds. Mother-to-kitten transmission can occur in utero or through nursing. Clinical signs vary but may include weight loss, poor appetite, recurring infections, gingivitis, diarrhea, and enlarged lymph nodes. Testing using ELISA or IFA is crucial for diagnosis. There is no cure for FeLV, but supportive care can help manage symptoms and secondary infections. Prevention includes vaccination of at-risk cats and avoiding exposure to infected cats.",
                "category": "viral_diseases",
                "species": "feline",
                "source": "veterinary_manual",
                "urgency": "moderate"
            },
            {
                "title": "Bovine Tuberculosis (bTB)",
                "text": "Bovine tuberculosis (bTB) is a chronic bacterial disease of cattle caused by Mycobacterium bovis. This zoonotic disease can affect humans and other animals. The infection primarily affects the lungs but can involve other organs including lymph nodes, liver, and udder. The disease is characterized by the formation of granulomatous lesions called tubercles. Clinical signs may include chronic cough, weight loss, decreased milk production, and enlarged lymph nodes. However, many infected cattle show no clinical signs. Diagnosis involves tuberculin skin testing (TST), interferon-gamma assays, and post-mortem examination. Control measures include regular testing programs, culling of positive animals, movement restrictions, and biosecurity measures. The disease has significant economic impact on the cattle industry.",
                "category": "bacterial_diseases",
                "species": "bovine",
                "source": "veterinary_manual",
                "urgency": "moderate"
            },
            {
                "title": "Rabies Prevention and Control",
                "text": "Rabies is a fatal viral disease that causes progressive inflammation of the brain (encephalitis) in mammals. The virus is typically transmitted through bite wounds from infected animals, as the virus is present in saliva. The incubation period varies from weeks to months depending on the location of the bite and viral load. Clinical signs progress through three phases: prodromal (behavioral changes, fever), furious (aggression, hypersalivation), and paralytic (weakness, paralysis, coma, death). There is no treatment once clinical signs appear, making prevention crucial. Vaccination of domestic animals is the primary control method. Post-exposure prophylaxis in humans involves immediate thorough wound cleaning, rabies immunoglobulin, and vaccination series. Wildlife reservoir species include bats, raccoons, skunks, and foxes depending on geographic region.",
                "category": "viral_diseases",
                "species": "multiple",
                "source": "veterinary_manual",
                "urgency": "critical"
            },
            {
                "title": "Equine Colic Management",
                "text": "Equine colic refers to abdominal pain in horses and represents a common veterinary emergency. Causes range from simple impaction and gas distention to life-threatening conditions like intestinal torsion, intussusception, or strangulating lipoma. Clinical signs include pawing, rolling, sweating, flank watching, absence of gut sounds, and elevated heart rate. Initial assessment should include vital signs, rectal examination, nasogastric intubation, and pain assessment. Treatment varies dramatically based on the underlying cause. Simple impactions may resolve with medical management including analgesics (flunixin meglumine), IV fluids, and mineral oil via nasogastric tube. Surgical intervention is required for strangulating lesions, severe impactions, or cases not responding to medical treatment. Prognosis depends on early recognition and appropriate intervention.",
                "category": "emergency_medicine",
                "species": "equine",
                "source": "emergency_protocols",
                "urgency": "critical"
            },
            {
                "title": "Feline Diabetes Mellitus Management",
                "text": "Feline diabetes mellitus is a common endocrine disorder characterized by insufficient insulin production or insulin resistance. Most cats develop Type 2 diabetes, often associated with obesity and pancreatic amyloidosis. Clinical signs include polydipsia (increased thirst), polyuria (increased urination), polyphagia (increased appetite), and weight loss. Diagnosis is based on persistent hyperglycemia and glucosuria. Treatment involves insulin therapy, typically using glargine or detemir insulin twice daily. Dietary management with high-protein, low-carbohydrate diets is crucial. Regular monitoring includes blood glucose curves and fructosamine levels. Some cats may achieve diabetic remission with proper management, especially if treated early. Complications include diabetic ketoacidosis, hypoglycemia from insulin overdose, and neuropathy.",
                "category": "endocrine_diseases",
                "species": "feline",
                "source": "internal_medicine",
                "urgency": "moderate"
            },
            {
                "title": "Kennel Cough Complex",
                "text": "Kennel cough, also known as canine infectious respiratory disease complex (CIRDC), is a highly contagious respiratory infection affecting dogs. Multiple pathogens can be involved including Bordetella bronchiseptica, parainfluenza virus, adenovirus, and mycoplasma. The condition spreads rapidly in environments where dogs are in close contact. Clinical signs include a harsh, dry cough often described as honking, retching, and production of white foam. Most dogs remain systemically well, but puppies and immunocompromised dogs may develop pneumonia. Diagnosis is often clinical, though PCR testing can identify specific pathogens. Treatment includes cough suppressants (dextromethorphan), antibiotics if bacterial involvement is suspected, and supportive care. Prevention involves vaccination with intranasal or injectable vaccines containing Bordetella and parainfluenza components.",
                "category": "respiratory_diseases",
                "species": "canine",
                "source": "veterinary_manual",
                "urgency": "low"
            },
            {
                "title": "Mastitis in Dairy Cattle",
                "text": "Mastitis is inflammation of the mammary gland and is the most economically important disease affecting dairy cattle. It can be caused by various bacteria including Streptococcus agalactiae, Staphylococcus aureus, Escherichia coli, and environmental streptococci. Clinical mastitis presents with visible abnormalities in milk (clots, discoloration) and mammary gland (swelling, heat, pain). Subclinical mastitis has no visible signs but results in elevated somatic cell counts. Diagnosis involves milk culture and sensitivity testing, somatic cell count analysis, and California Mastitis Test. Treatment depends on the causative organism and may include intramammary antibiotics, systemic antibiotics for severe cases, and supportive care. Prevention focuses on proper milking hygiene, teat dipping, dry cow therapy, and environmental management to reduce pathogen exposure.",
                "category": "bacterial_diseases",
                "species": "bovine",
                "source": "dairy_medicine",
                "urgency": "moderate"
            }
        ]
        
        documents = []
        for doc_data in sample_docs:
            # Split long documents into chunks
            text_chunks = self.text_splitter.split_text(doc_data["text"])
            
            for i, chunk in enumerate(text_chunks):
                metadata = {
                    "title": doc_data["title"],
                    "category": doc_data["category"],
                    "species": doc_data["species"],
                    "source": doc_data["source"],
                    "urgency": doc_data.get("urgency", "moderate"),
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "doc_id": f"{doc_data['title'].lower().replace(' ', '_')}_{i}"
                }
                
                documents.append(Document(page_content=chunk, metadata=metadata))
        
        logger.info(f"✅ Prepared {len(documents)} document chunks from {len(sample_docs)} source documents")
        return documents
    
    def upsert_documents(self, documents: List[Document], batch_size: int = 50) -> bool:
        """Upsert documents to Pinecone in batches."""
        try:
            # Create index if needed
            self.create_index_if_not_exists()
            
            # Process documents in batches
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for batch_num in range(0, len(documents), batch_size):
                batch_docs = documents[batch_num:batch_num + batch_size]
                current_batch = (batch_num // batch_size) + 1
                
                logger.info(f"Processing batch {current_batch}/{total_batches} ({len(batch_docs)} documents)")
                
                try:
                    # Extract texts and metadatas
                    texts = [doc.page_content for doc in batch_docs]
                    metadatas = [doc.metadata for doc in batch_docs]
                    
                    # Upsert to Pinecone
                    if not self.vectorstore:
                        # Create new vectorstore
                        self.vectorstore = Pinecone.from_texts(
                            texts=texts,
                            embedding=self.embeddings,
                            metadatas=metadatas,
                            index_name=self.index_name
                        )
                    else:
                        # Add to existing vectorstore
                        self.vectorstore.add_texts(
                            texts=texts,
                            metadatas=metadatas
                        )
                    
                    logger.info(f"✅ Batch {current_batch} upserted successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Error upserting batch {current_batch}: {e}")
                    return False
            
            logger.info(f"🎉 Successfully upserted {len(documents)} documents to index '{self.index_name}'")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in upsert process: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        try:
            index = pinecone.Index(self.index_name)
            stats = index.describe_index_stats()
            
            return {
                "index_name": self.index_name,
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}

def main():
    """Main function to run the upserting process."""
    try:
        logger.info("🚀 Starting VetIOS knowledge base creation...")
        
        # Initialize upserter
        upserter = VeterinaryDocumentUpserter()
        
        # Get sample documents
        documents = upserter.prepare_veterinary_documents()
        
        # Upsert documents
        success = upserter.upsert_documents(documents)
        
        if success:
            # Get and display index stats
            stats = upserter.get_index_stats()
            logger.info(f"📊 Index statistics: {stats}")
            logger.info("✅ Knowledge base creation completed successfully!")
        else:
            logger.error("❌ Failed to create knowledge base")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in main process: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
