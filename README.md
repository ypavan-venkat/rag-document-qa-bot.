🤖 RAG Document Q&A BotA professional-grade Retrieval-Augmented Generation (RAG) pipeline built to answer complex technical questions from research papers. This system ensures that answers are strictly grounded in provided documents, providing clear citations for every response.  🛠️ Tech StackLanguage: Python 3.11+LLM: Google Gemini 2.5 Flash (via langchain-google-genai)Orchestration: LangChain (LCEL)Vector Database: ChromaDBEmbeddings: HuggingFace all-MiniLM-L6-v2Document Loading: PyPDF  🏗️ Architecture OverviewThe system follows a standard five-stage RAG pipeline:Ingestion: Extracts text from PDF files located in the /data folder.  Chunking: Breaks large documents into smaller, manageable pieces.  Embedding: Converts text chunks into high-dimensional vectors.  Retrieval: Uses semantic search to find the top $k$ relevant chunks for a user query.  Generation: Passes context to the LLM to generate a grounded response with citations.  🧠 Technical DecisionsChunking Strategy: Recursive Character Text SplittingI chose the Recursive Character Text Splitter with a chunk size of 1000 tokens and an overlap of 200 tokens.  Why: Unlike simple fixed-size splitting, this method respects natural boundaries like paragraphs and sentences, which preserves the semantic context of complex technical definitions.  Vector Database: ChromaDBI implemented ChromaDB as the vector store because it is lightweight, open-source, and supports disk-based persistence. This allows the "Indexing" and "Querying" steps to remain separate—the bot doesn't need to re-process the PDFs every time it starts up.  Retrieval Tuning: $k=10$To handle dense academic papers, I configured the retriever to fetch the top 10 chunks. This ensures the LLM has enough context to explain math-heavy concepts that might be spread across multiple pages.  🚀 Setup InstructionsClone the Repo:Bashgit clone https://github.com/ypavan-venkat/rag-document-qa-bot.git
cd rag-document-qa-bot
Environment Setup:Create a .env file and add your Google API Key:PlaintextGOOGLE_API_KEY=your_actual_key_here
Install Dependencies:Bashpip install -r requirements.txt

4.  **Run Ingestion:**
    ```bash
    python ingest.py
    ```
5.  **Start the Bot:**
    ```bash
    python app.py
    ```

## 🧪 Example Queries
1.  *"What is the role of the scaling factor in Attention?"*
2.  *"Explain the difference between BERT and GPT."*
3.  *"How does LoRA minimize trainable parameters?"*
4.  *"What statistical divergence is used in GANs?"*
5.  **Grounding Test:** *"What is the recipe for chocolate cake?"* (Expected: Bot refuses to answer)[cite: 1].

## ⚠️ Known Limitations
*   **Math Rendering:** Basic PDF parsers occasionally struggle with complex Greek symbols in academic formulas[cite: 1].
*   **Local Hardware:** Initial embedding speed depends on local CPU performance[cite: 1].

---

