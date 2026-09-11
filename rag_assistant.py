import os
import pickle
import faiss
import psycopg2
import re
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import gc
import pdfplumber

load_dotenv() 

# Global variables for models
embedding_model = None

def load_embedding_model():
    global embedding_model
    
    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        gc.collect()
    
    return embedding_model

def extract_pdf_content(pdf_path):
    """Extract text content from PDF file"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_content = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
            return text_content.strip()
    except Exception as e:
        print(f"Error extracting content from {pdf_path}: {e}")
        return ""

def get_documents_from_db(case_id=None):
    """Get documents from database and extract actual PDF content"""
    # Print environment variable values for debugging
    # Debug logging intentionally minimal to avoid leaking credentials
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        sslmode="require"
    )
    
    documents = []

    with conn:
        with conn.cursor() as cur:
            # Get document metadata from database
            if case_id:
                cur.execute("""
                    SELECT document_title, metadata, file_path FROM documents
                    WHERE case_id = %s
                """, (case_id,))
            else:
                cur.execute("""
                    SELECT document_title, metadata, file_path FROM documents
                """)
            rows = cur.fetchall()

    # Process each document
    for row in rows:
        title, metadata, file_path = row
        
        # Fix file path to use correct sample_docs location
        if file_path and file_path.startswith('/sample_docs/'):
            # Convert from /sample_docs/filename.pdf to sample_docs/2024-PI-001/filename.pdf
            filename = os.path.basename(file_path)
            corrected_path = f"sample_docs/2024-PI-001/{filename}"
        else:
            corrected_path = file_path
        
        # Extract actual PDF content
        pdf_content = ""
        if corrected_path and os.path.exists(corrected_path):
            pdf_content = extract_pdf_content(corrected_path)
            print(f"Extracted content from: {title} ({len(pdf_content)} characters)")
        else:
            print(f"PDF not found: {corrected_path}")
        
        # Combine title + metadata + actual PDF content
        meta_str = str(metadata) if isinstance(metadata, dict) else str(metadata)
        full_content = f"{title}\n{meta_str}\n{pdf_content}"
        documents.append(full_content)
    
    return documents


# Build and save FAISS index
def build_faiss_index(documents):
    model = load_embedding_model()
    
    print("Encoding documents...")
    doc_embeddings = model.encode(documents, show_progress_bar=True)
    
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    faiss.write_index(index, "faiss_index.bin")

    # Save map for later lookup
    doc_id_map = {i: doc for i, doc in enumerate(documents)}
    with open("doc_id_map.pkl", "wb") as f:
        pickle.dump(doc_id_map, f)
    
    print("Index built successfully!")


# Retrieve top-k documents
def retrieve_relevant_docs(query, top_k=3):
    model = load_embedding_model()
    
    query_embedding = model.encode([query])

    index = faiss.read_index("faiss_index.bin")
    with open("doc_id_map.pkl", "rb") as f:
        doc_id_map = pickle.load(f)

    D, I = index.search(query_embedding, top_k)
    return [doc_id_map[i] for i in I[0]]


def extract_financial_info(documents):
    """Extract financial information from documents"""
    total_expenses = 0
    providers = []
    
    for doc in documents:
        # Look for expense information in metadata
        if isinstance(doc, str):
            # Try to extract JSON-like structures
            json_matches = re.findall(r'\{[^}]*\}', doc)
            for match in json_matches:
                try:
                    # Convert single quotes to double quotes for valid JSON
                    json_str = match.replace("'", '"')
                    data = json.loads(json_str)
                    if 'total_expenses' in data:
                        total_expenses += data['total_expenses']
                    if 'provider' in data:
                        providers.append(data['provider'])
                except:
                    pass
            
            # Look for numeric patterns that might be expenses
            expense_matches = re.findall(r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', doc)
            for match in expense_matches:
                try:
                    amount = float(match.replace(',', ''))
                    if amount > 100:  # Likely an expense if over $100
                        total_expenses += amount
                except:
                    pass
    print(total_expenses)
    print(providers)
    return total_expenses, providers

def generate_demand_letter_with_llm(query, retrieved_docs):
    """Generate a demand letter using LLM-like approach"""
    
    # Extract key information from documents
    total_expenses, providers = extract_financial_info(retrieved_docs)
    
    # Find liability information
    liability_info = ""
    for doc in retrieved_docs:
        if 'police' in doc.lower() and 'fault' in doc.lower():
            liability_info = doc
            break
    
    # Extract case ID
    case_id = "2024-PI-001"  # Default
    if "2024-PI-001" in query:
        case_id = "2024-PI-001"
    
    # Create context for LLM
    context = f"""
Case Information:
- Case ID: {case_id}
- Medical Expenses: ${total_expenses:,.2f}
- Medical Providers: {', '.join(set(providers)) if providers else 'Various providers'}
- Liability: {liability_info if liability_info else 'Police report indicates clear liability'}

Available Documents:
{chr(10).join([f"- {doc[:100]}..." for doc in retrieved_docs[:3]])}

Generate a professional demand letter that includes:
1. All medical expenses (${total_expenses:,.2f})
2. Lost wages (estimate based on typical recovery time)
3. Pain and suffering damages
4. References to Dr. Jones' medical findings
5. Police report liability determination
6. Proper legal citations and professional tone
"""
    
    # For now, return a structured response that mimics LLM output
    # In a real implementation, this would call an actual LLM API
    demand_letter = f"""
DEMAND LETTER
Case: {case_id}

Dear Claims Representative,

This letter serves as a formal demand for settlement of the above-referenced personal injury claim arising from the incident involving your insured.

MEDICAL EXPENSES AND TREATMENT:
Our client has incurred substantial medical expenses totaling ${total_expenses:,.2f} as a result of the injuries sustained in this incident. Medical treatment was provided by {', '.join(set(providers)) if providers else 'qualified medical professionals'}, including Dr. Michael Jones, whose medical findings document the extent and severity of our client's injuries.

LIABILITY DETERMINATION:
{liability_info if liability_info else 'The police report clearly establishes liability on the part of your insured, with fault determination indicating 100% responsibility on the part of the defendant.'}

DAMAGES CLAIMED:

1. Medical Expenses: ${total_expenses:,.2f}
   - Emergency medical treatment
   - Ongoing medical care and rehabilitation
   - Prescription medications
   - Medical equipment and supplies

2. Lost Wages: $15,000 (estimated)
   - Time missed from work due to injuries
   - Reduced earning capacity during recovery
   - Future lost wages if applicable

3. Pain and Suffering: $50,000
   - Physical pain and discomfort
   - Emotional distress
   - Loss of enjoyment of life
   - Permanent impairment (if any)

4. Property Damage: $5,000
   - Vehicle damage and related expenses

TOTAL DEMAND: ${total_expenses + 70000:,.2f}

LEGAL BASIS:
This demand is based on the following factors:
- Clear liability as established by law enforcement
- Substantial medical expenses and ongoing treatment needs
- Pain and suffering endured by our client
- Impact on quality of life and daily activities
- Precedent cases with similar injury patterns

We request a response to this demand within 30 days. If this matter cannot be resolved amicably, we will proceed with litigation to protect our client's rights and seek full compensation for all damages sustained.

Please direct all communications regarding this matter to our office.

Sincerely,

[Attorney Name]
[Law Firm Name]
[Address]
[Phone]
[Email]

CC: Client
"""
    
    return demand_letter.strip()


def generate_smart_response(query, retrieved_docs):
    """Generate a more intelligent response based on query type"""
    query_lower = query.lower()
    
    # Demand letter generation queries
    if any(word in query_lower for word in ['demand letter', 'demand', 'letter']):
        return generate_demand_letter_with_llm(query, retrieved_docs)
    
    # Provider-specific queries (put this first)
    elif "provider" in query_lower or "doctor" in query_lower:
        total_expenses, providers = extract_financial_info(retrieved_docs)
        if providers:
            return f"The medical providers mentioned are: {', '.join(set(providers))}"
        else:
            return "No specific providers were found in the documents."
    
    # Financial queries
    elif any(word in query_lower for word in ['expense', 'cost', 'amount', 'total', 'medical', 'bill', 'summary', 'damages']):
        total_expenses, providers = extract_financial_info(retrieved_docs)
        if total_expenses > 0:
            response = f"Based on the retrieved documents, the total medical expenses are ${total_expenses:,.2f}."
            if providers:
                response += f"\n\nProviders include: {', '.join(set(providers))}"
            return response
        else:
            return "I found relevant medical documents, but couldn't extract specific expense amounts."
    
    # Timeline queries
    elif any(word in query_lower for word in ['when', 'date', 'timeline', 'incident']):
        if retrieved_docs:
            return f"I found {len(retrieved_docs)} relevant documents that may contain timeline information. The most relevant document is:\n\n{retrieved_docs[0]}"
        return "I could not find timeline information in the documents."
    
    # General fallback
    else:
        if retrieved_docs:
            response = f"I found {len(retrieved_docs)} relevant documents for your query.\n\n"
            response += "Most relevant document:\n"
            
            # Format the document title and metadata more cleanly
            doc_parts = retrieved_docs[0].split('\n', 1)
            if len(doc_parts) > 1:
                title = doc_parts[0]
                metadata = doc_parts[1]
                
                # Clean up metadata display
                if metadata.startswith("{'") or metadata.startswith('{"'):
                    try:
                        # Convert to proper JSON and format nicely
                        import json
                        clean_metadata = metadata.replace("'", '"')
                        data = json.loads(clean_metadata)
                        formatted_metadata = "\n".join([f"  • {key.replace('_', ' ').title()}: {value}" for key, value in data.items()])
                        response += f"{title}\n{formatted_metadata}"
                    except:
                        response += retrieved_docs[0]
                else:
                    response += retrieved_docs[0]
            else:
                response += retrieved_docs[0]
            
            if len(retrieved_docs) > 1:
                response += f"\n\nOther relevant documents:\n"
                for i, doc in enumerate(retrieved_docs[1:], 2):
                    doc_parts = doc.split('\n', 1)
                    if len(doc_parts) > 1:
                        title = doc_parts[0]
                        metadata = doc_parts[1]
                        
                        # Clean up metadata display for other documents too
                        if metadata.startswith("{'") or metadata.startswith('{"'):
                            try:
                                clean_metadata = metadata.replace("'", '"')
                                data = json.loads(clean_metadata)
                                formatted_metadata = ", ".join([f"{key.replace('_', ' ').title()}: {value}" for key, value in data.items()])
                                response += f"{i}. {title} ({formatted_metadata})\n"
                            except:
                                response += f"{i}. {doc[:100]}...\n"
                        else:
                            response += f"{i}. {doc[:100]}...\n"
                    else:
                        response += f"{i}. {doc[:100]}...\n"
            
            return response
        else:
            return "No relevant documents found for your query."

def extract_medical_expenses_summary(pdf_path):
    summary_text = ""
    found_summary = False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            lines = text.splitlines()
            for line in lines:
                if "TOTAL MEDICAL EXPENSES SUMMARY" in line.upper():
                    found_summary = True
                    summary_text += line + "\n"
                    continue

                if found_summary:
                    # Stop if we hit another section header in ALL CAPS
                    if re.match(r"^[A-Z\s]+$", line.strip()) and not line.strip().isdigit():
                        return summary_text.strip()
                    summary_text += line + "\n"

    return summary_text.strip()

# Enhanced RAG response function
def generate_rag_response(query):
    print(f"Query: {query}")
    retrieved_docs = retrieve_relevant_docs(query)
    print(f"Retrieved {len(retrieved_docs)} documents")
    
    if retrieved_docs:
        print(f"Most relevant document: {retrieved_docs[0][:200]}...")
    
    return generate_smart_response(query, retrieved_docs)


# Optional: Only run this once to generate index
# if __name__ == "__main__":
#     try:
#         docs = get_documents_from_db()
#         print(f"Loaded {len(docs)} documents from DB.")
#         build_faiss_index(docs)
#         print("FAISS index built and saved.")

#     except Exception as e:
#         print(f"Error during execution: {e}")
#         import traceback
#         traceback.print_exc()

# Section-level RAG index (built from PDF sections) — now per-case
_current_sections_case_id = None

def _section_paths(case_id: str):
    base = f"faiss_sections_{case_id}.bin"
    mapf = f"section_id_map_{case_id}.pkl"
    cache_json = os.path.join("cache", case_id, "sections.json")
    return base, mapf, cache_json


def _ensure_cache_dir(case_id: str):
    cache_dir = os.path.join("cache", case_id)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _is_section_header(line: str) -> bool:
    if not line:
        return False
    stripped = line.strip()
    if len(stripped) < 3:
        return False
    # Explicit legal sections
    keywords = [
        "WAGE LOSS SUMMARY",
        "FUTURE WAGE IMPACT ANALYSIS",
        "TOTAL TIME OFF WORK",
        "TOTAL MEDICAL EXPENSES SUMMARY",
        "ESTIMATED REPAIR COST",
        "POLICE REPORT",
        "PROPERTY DAMAGE",
    ]
    if any(k in stripped.upper() for k in keywords):
        return True
    # All-caps header heuristic
    return stripped[:6].isupper()


def extract_pdf_sections(pdf_path: str) -> list[dict]:
    """Chunk a PDF into sections by headers; returns list of {title, text, source}."""
    sections = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            current_title = None
            current_lines = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                for raw_line in text.splitlines():
                    line = raw_line.rstrip()
                    if _is_section_header(line):
                        # flush previous
                        if current_title and current_lines:
                            sections.append({
                                "title": current_title.strip(),
                                "text": "\n".join(current_lines).strip(),
                                "source": os.path.basename(pdf_path)
                            })
                            current_lines = []
                        current_title = line.strip()
                    else:
                        current_lines.append(line)
            if current_title and current_lines:
                sections.append({
                    "title": current_title.strip(),
                    "text": "\n".join(current_lines).strip(),
                    "source": os.path.basename(pdf_path)
                })
    except Exception as e:
        print(f"Error extracting sections from {pdf_path}: {e}")
    return sections


def build_section_index_for_case(case_id: str) -> int:
    """Build FAISS index over section-level chunks for all PDFs in a case folder, using cached sections when available."""
    global _current_sections_case_id
    case_folder = os.path.join("sample_docs", case_id)
    if not os.path.isdir(case_folder):
        return 0
    index_path, map_path, cache_json = _section_paths(case_id)
    _ensure_cache_dir(case_id)

    all_sections: list[dict] = []
    # Load cached sections if present
    if os.path.exists(cache_json):
        try:
            with open(cache_json, "r") as f:
                all_sections = json.load(f)
        except Exception as e:
            print("Error loading cached sections.json:", e)
            all_sections = []
    if not all_sections:
        # Extract from PDFs and cache
        for fname in os.listdir(case_folder):
            if fname.lower().endswith('.pdf'):
                pdf_path = os.path.join(case_folder, fname)
                secs = extract_pdf_sections(pdf_path)
                all_sections.extend(secs)
        try:
            with open(cache_json, "w") as f:
                json.dump(all_sections, f)
        except Exception as e:
            print("Error writing cached sections.json:", e)

    if not all_sections:
        return 0

    model = load_embedding_model()
    texts = [f"{s['title']}\n{s['text']}" for s in all_sections]
    embeddings = model.encode(texts, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    # map id -> section dict
    id_map = {i: all_sections[i] for i in range(len(all_sections))}
    with open(map_path, "wb") as f:
        pickle.dump(id_map, f)

    _current_sections_case_id = case_id
    return len(all_sections)


def ensure_section_index(case_id: str):
    global _current_sections_case_id
    index_path, map_path, _ = _section_paths(case_id)
    if not (os.path.exists(index_path) and os.path.exists(map_path)):
        try:
            build_section_index_for_case(case_id)
        except Exception as e:
            print(f"Error building section index: {e}")
    else:
        _current_sections_case_id = case_id


def retrieve_relevant_sections(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve top-k sections for the most recently ensured case; returns list of section dicts."""
    try:
        if not _current_sections_case_id:
            print("retrieve_relevant_sections called before ensure_section_index; no case set")
            return []
        index_path, map_path, _ = _section_paths(_current_sections_case_id)
        model = load_embedding_model()
        q_emb = model.encode([query])
        index = faiss.read_index(index_path)
        with open(map_path, "rb") as f:
            id_map = pickle.load(f)
        D, I = index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            if int(idx) in id_map:
                results.append(id_map[int(idx)])
        return results
    except Exception as e:
        print(f"Error retrieving sections: {e}")
        return []