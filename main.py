"""
Toyota Internal Knowledge Assistant
LangChain + ChromaDB + Ollama (local LLM) RAG Application
"""

import os
from pathlib import Path
from typing import List

# ─── LangChain Imports ────────────────────────────────────────────────────────
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document


# ─── Configuration ────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"   # Default Ollama URL
OLLAMA_MODEL    = "llama3"                   # Change to any model you've pulled (llama3, mistral, gemma, etc.)
EMBED_MODEL     = "nomic-embed-text"         # Ollama embedding model
CHROMA_DIR      = "./toyota_chroma_db"       # Persist ChromaDB here
COLLECTION_NAME = "toyota_knowledge_base"
DOCS_DIR        = "./toyota_docs"            # Put your Toyota docs here


# ─── Toyota-Specific Prompt Template ─────────────────────────────────────────

TOYOTA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are ToyotaAI, an expert internal knowledge assistant for Toyota Motor Corporation.
You help employees, engineers, and stakeholders with information about Toyota's products,
manufacturing processes (Toyota Production System / TPS), quality standards, vehicle specs,
service manuals, and company policies.

Use ONLY the context below to answer. If the context doesn't contain the answer,
say: "I don't have enough information in the knowledge base to answer that. 
Please consult the relevant Toyota department or official documentation."

Context:
{context}

Question: {question}

Answer (be concise, precise, and professional):"""
)


# ─── Sample Toyota Documents (used if no real docs exist) ─────────────────────

SAMPLE_TOYOTA_DOCS = [
    Document(
        page_content="""Toyota Production System (TPS) Overview:
        The Toyota Production System is an integrated socio-technical system that comprises its 
        management philosophy and practices. TPS organizes manufacturing and logistics, including 
        interaction with suppliers and customers. The two pillars of TPS are:
        1. Just-In-Time (JIT): Producing only what is needed, when it is needed, in the amount needed.
        2. Jidoka (Autonomation): Automation with a human touch – stopping when a problem occurs.
        The system is built on the foundation of Heijunka (production leveling), Standardized Work,
        and Kaizen (continuous improvement).""",
        metadata={"source": "TPS_Overview", "category": "Manufacturing", "dept": "Production"}
    ),
    Document(
        page_content="""Toyota Camry 2024 Specifications:
        Engine: 2.5L 4-cylinder Dynamic Force Engine
        Horsepower: 203 hp (base) / 225 hp (Hybrid)
        Transmission: 8-speed Direct-Shift Automatic
        Fuel Economy: 28 city / 39 highway MPG (base)
        Hybrid Economy: 51 city / 53 highway MPG
        Safety: Toyota Safety Sense 3.0 (TSS 3.0) standard
        Features: Pre-Collision System, Lane Departure Alert, Radar Cruise Control, Auto High Beams
        Wheelbase: 111.2 inches | Length: 192.7 inches | MSRP from: $27,315""",
        metadata={"source": "Camry_2024_Specs", "category": "Vehicle Specs", "model": "Camry"}
    ),
    Document(
        page_content="""Toyota Safety Sense (TSS) Technology:
        Toyota Safety Sense is a bundle of active safety features standard across most Toyota vehicles.
        TSS 3.0 includes:
        - Pre-Collision System with Pedestrian Detection (PCS): Detects vehicles, pedestrians, cyclists
        - Lane Departure Alert with Steering Assist (LDA): Warns and assists if drifting
        - Automatic High Beams (AHB): Automatically switches between high/low beams
        - Radar Cruise Control (RCC): Maintains set speed and distance from leading vehicle
        - Lane Tracing Assist (LTA): Keeps vehicle centered within detected lane
        - Emergency Steering Assist (ESA): Helps avoid collisions when obstacle detected""",
        metadata={"source": "TSS_Documentation", "category": "Safety Technology", "dept": "Engineering"}
    ),
    Document(
        page_content="""Toyota Quality Management - The Toyota Way:
        Toyota's quality philosophy is guided by The Toyota Way, established in 2001, with two pillars:
        1. Continuous Improvement (Kaizen): Challenge, Kaizen, Genchi Genbutsu (go and see)
        2. Respect for People: Respect, Teamwork
        Quality Control Tools used at Toyota:
        - Andon: Visual alert system for production line issues
        - Poka-Yoke: Error-proofing mechanisms
        - 5 Whys: Root cause analysis technique
        - A3 Report: One-page problem-solving methodology
        - PDCA Cycle: Plan-Do-Check-Act continuous improvement loop
        Toyota targets zero defects through these systematic quality assurance processes.""",
        metadata={"source": "Quality_Management", "category": "Quality", "dept": "Quality Assurance"}
    ),
    Document(
        page_content="""Toyota Hybrid Technology - Hybrid Synergy Drive (HSD):
        Toyota's Hybrid Synergy Drive is a full hybrid system that can run on electric only, 
        gasoline only, or a combination of both. Key components:
        - High-voltage NiMH or Lithium-Ion battery pack
        - Electric Motor/Generator (MG1 and MG2)
        - Power Split Device (Planetary Gear Set)
        - Gasoline Internal Combustion Engine
        The system uses regenerative braking to recharge batteries.
        Toyota has sold over 20 million hybrid vehicles globally.
        Current hybrid lineup includes: Prius, Camry Hybrid, RAV4 Hybrid, Highlander Hybrid,
        Sienna (standard hybrid), Venza, Crown, and more.""",
        metadata={"source": "Hybrid_Technology", "category": "Technology", "dept": "R&D"}
    ),
    Document(
        page_content="""Toyota Service Intervals and Maintenance Schedule:
        Toyota recommends the following standard maintenance intervals:
        - Oil Change: Every 10,000 miles or 12 months (synthetic oil)
        - Tire Rotation: Every 5,000 miles
        - Cabin Air Filter: Every 15,000 miles
        - Engine Air Filter: Every 30,000 miles
        - Spark Plugs (iridium): Every 60,000 miles
        - Brake Fluid: Every 45,000 miles
        - Coolant: Every 100,000 miles (first interval), then every 50,000 miles
        - Transmission Fluid: Every 60,000 miles
        Toyota Care Plan: Covers first 2 years / 25,000 miles of complimentary maintenance.
        ToyotaCare Plus: Extended maintenance plan available for purchase.""",
        metadata={"source": "Maintenance_Schedule", "category": "Service", "dept": "After-Sales"}
    ),
]


# ─── Core RAG Classes ─────────────────────────────────────────────────────────

class ToyotaDocumentLoader:
    """Loads Toyota documents from directory or uses sample data."""

    def __init__(self, docs_dir: str = DOCS_DIR):
        self.docs_dir = Path(docs_dir)

    def load_from_directory(self) -> List[Document]:
        """Load PDF and TXT files from the docs directory."""
        docs = []
        if not self.docs_dir.exists():
            print(f"[INFO] Docs directory '{self.docs_dir}' not found. Using sample documents.")
            return []

        # Load PDFs
        pdf_loader = DirectoryLoader(
            str(self.docs_dir), glob="**/*.pdf", loader_cls=PyPDFLoader
        )
        # Load TXT files
        txt_loader = DirectoryLoader(
            str(self.docs_dir), glob="**/*.txt", loader_cls=TextLoader
        )

        try:
            docs.extend(pdf_loader.load())
            docs.extend(txt_loader.load())
            print(f"[INFO] Loaded {len(docs)} documents from '{self.docs_dir}'")
        except Exception as e:
            print(f"[WARN] Error loading documents: {e}")

        return docs

    def get_documents(self) -> List[Document]:
        """Return real docs or fall back to sample docs."""
        docs = self.load_from_directory()
        if not docs:
            print("[INFO] Using built-in Toyota sample documents for demonstration.")
            return SAMPLE_TOYOTA_DOCS
        return docs


class ToyotaVectorStore:
    """Manages ChromaDB vector store with Ollama embeddings."""

    def __init__(self):
        print(f"[INFO] Initializing Ollama embeddings with model: {EMBED_MODEL}")
        self.embeddings = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        self.vectorstore = None

    def build(self, documents: List[Document]) -> None:
        """Chunk documents and build ChromaDB index."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split into {len(chunks)} chunks. Building ChromaDB vector store...")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
        )
        print(f"[INFO] Vector store built and persisted to '{CHROMA_DIR}'")

    def load(self) -> None:
        """Load an existing ChromaDB from disk."""
        print(f"[INFO] Loading existing ChromaDB from '{CHROMA_DIR}'")
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )

    def get_retriever(self, k: int = 4):
        """Return a retriever that fetches top-k relevant chunks."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call build() or load() first.")
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )


class ToyotaRAGApp:
    """Main Toyota RAG Application."""

    def __init__(self):
        self.loader      = ToyotaDocumentLoader()
        self.vector_mgr  = ToyotaVectorStore()
        self.qa_chain    = None

    def setup(self, force_rebuild: bool = False) -> None:
        """Initialize or load vector store, then build QA chain."""
        db_exists = Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir())

        if db_exists and not force_rebuild:
            self.vector_mgr.load()
        else:
            documents = self.loader.get_documents()
            self.vector_mgr.build(documents)

        # Initialize Ollama LLM
        print(f"[INFO] Connecting to Ollama model: {OLLAMA_MODEL}")
        llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,        # Low temperature for factual answers
            num_predict=512,        # Max tokens in response
        )

        # Build RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_mgr.get_retriever(k=4),
            chain_type_kwargs={"prompt": TOYOTA_PROMPT},
            return_source_documents=True,
        )
        print("[INFO] Toyota RAG application ready!\n")

    def ask(self, question: str) -> dict:
        """Ask a question and return the answer with sources."""
        if not self.qa_chain:
            raise RuntimeError("App not set up. Call setup() first.")

        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)

        result = self.qa_chain.invoke({"query": question})

        answer   = result["result"]
        sources  = result.get("source_documents", [])

        print(f"\nAnswer:\n{answer}")

        if sources:
            print(f"\nSources ({len(sources)} chunks retrieved):")
            seen = set()
            for doc in sources:
                src = doc.metadata.get("source", "Unknown")
                cat = doc.metadata.get("category", "")
                key = f"{src}|{cat}"
                if key not in seen:
                    seen.add(key)
                    print(f"  • [{cat}] {src}")

        return {"answer": answer, "sources": sources}

    def interactive_chat(self) -> None:
        """Run an interactive CLI chat session."""
        print("\n" + "🚗 " * 20)
        print("  TOYOTA INTERNAL KNOWLEDGE ASSISTANT")
        print("  Powered by LangChain + ChromaDB + Ollama")
        print("🚗 " * 20)
        print("\nType your question and press Enter. Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[INFO] Session ended.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye! Drive safely. 🚗")
                break

            self.ask(user_input)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Toyota RAG Knowledge Assistant")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the vector store")
    parser.add_argument("--query",   type=str, help="Run a single query and exit")
    args = parser.parse_args()

    app = ToyotaRAGApp()
    app.setup(force_rebuild=args.rebuild)

    if args.query:
        app.ask(args.query)
    else:
        app.interactive_chat()