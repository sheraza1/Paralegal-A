"""
Advanced Citation Tracking System for Legal AI RAG
Handles page number extraction and detailed source references for legal compliance
"""

import pdfplumber
import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import os


@dataclass
class Citation:
    """Represents a legal citation with source tracking"""
    text: str
    source_document: str
    page_number: int
    line_number: Optional[int] = None
    context: str = ""
    citation_type: str = "legal"  # legal, medical, financial, etc.
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert citation to dictionary for storage"""
        return {
            'text': self.text,
            'source_document': self.source_document,
            'page_number': self.page_number,
            'line_number': self.line_number,
            'context': self.context,
            'citation_type': self.citation_type,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        """String representation for display"""
        return f"{self.text} (p.{self.page_number}, {self.source_document})"


class CitationTracker:
    """Advanced citation tracking system for legal documents"""
    
    def __init__(self):
        self.citations: List[Citation] = []
        self.page_content_map: Dict[str, Dict[int, str]] = {}
        
    def extract_page_numbers(self, pdf_path: str) -> Dict[int, str]:
        """Extract page numbers and content from PDF"""
        page_content = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        page_content[page_num] = text.strip()
                        
        except Exception as e:
            print(f"Error extracting pages from {pdf_path}: {e}")
            
        return page_content
    
    def find_legal_citations(self, text: str) -> List[str]:
        """Find legal citations in text"""
        # Common legal citation patterns
        patterns = [
            r'VTL\s*§\s*\d+',  # Vehicle and Traffic Law
            r'CVC\s*\d+',       # California Vehicle Code
            r'NYC\s*§\s*\d+',   # New York City Code
            r'U\.S\.C\.\s*\d+', # United States Code
            r'CFR\s*\d+',       # Code of Federal Regulations
            r'[A-Z]{2,4}\s*§\s*\d+',  # General state codes
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(matches)
            
        return list(set(citations))  # Remove duplicates
    
    def find_medical_citations(self, text: str) -> List[str]:
        """Find medical references in text"""
        patterns = [
            r'Dr\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+',  # Doctor names
            r'[A-Z][a-z]+\s+Hospital',             # Hospital names
            r'[A-Z][a-z]+\s+Medical\s+Center',     # Medical centers
            r'[A-Z][a-z]+\s+Clinic',               # Clinics
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
            
        return list(set(citations))
    
    def find_financial_citations(self, text: str) -> List[str]:
        """Find financial references in text"""
        patterns = [
            r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?',  # Dollar amounts
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*dollars',  # Written amounts
            r'[A-Z][a-z]+\s+Insurance\s+Company',  # Insurance companies
            r'Policy\s+#[A-Z0-9-]+',               # Policy numbers
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(matches)
            
        return list(set(citations))
    
    def extract_citations_from_page(self, page_content: str, page_num: int, 
                                   source_doc: str) -> List[Citation]:
        """Extract all types of citations from a single page"""
        citations = []
        
        # Find legal citations
        legal_citations = self.find_legal_citations(page_content)
        for citation in legal_citations:
            citations.append(Citation(
                text=citation,
                source_document=source_doc,
                page_number=page_num,
                citation_type="legal",
                context=self.get_context(page_content, citation)
            ))
        
        # Find medical citations
        medical_citations = self.find_medical_citations(page_content)
        for citation in medical_citations:
            citations.append(Citation(
                text=citation,
                source_document=source_doc,
                page_number=page_num,
                citation_type="medical",
                context=self.get_context(page_content, citation)
            ))
        
        # Find financial citations
        financial_citations = self.find_financial_citations(page_content)
        for citation in financial_citations:
            citations.append(Citation(
                text=citation,
                source_document=source_doc,
                page_number=page_num,
                citation_type="financial",
                context=self.get_context(page_content, citation)
            ))
        
        return citations
    
    def get_context(self, text: str, citation: str, context_chars: int = 100) -> str:
        """Get context around a citation"""
        try:
            index = text.find(citation)
            if index == -1:
                return ""
            
            start = max(0, index - context_chars)
            end = min(len(text), index + len(citation) + context_chars)
            
            return text[start:end].strip()
        except:
            return ""
    
    def process_document(self, pdf_path: str, document_title: str) -> List[Citation]:
        """Process a document and extract all citations with page numbers"""
        print(f"Processing document: {document_title}")
        
        # Extract page content
        page_content = self.extract_page_numbers(pdf_path)
        self.page_content_map[document_title] = page_content
        
        all_citations = []
        
        # Process each page
        for page_num, content in page_content.items():
            citations = self.extract_citations_from_page(content, page_num, document_title)
            all_citations.extend(citations)
            print(f"  Page {page_num}: Found {len(citations)} citations")
        
        # Add to global citations
        self.citations.extend(all_citations)
        
        return all_citations
    
    def get_citations_by_type(self, citation_type: str) -> List[Citation]:
        """Get citations filtered by type"""
        return [c for c in self.citations if c.citation_type == citation_type]
    
    def get_citations_by_document(self, document_title: str) -> List[Citation]:
        """Get citations from a specific document"""
        return [c for c in self.citations if c.source_document == document_title]
    
    def get_citations_by_page(self, document_title: str, page_number: int) -> List[Citation]:
        """Get citations from a specific page"""
        return [c for c in self.citations 
                if c.source_document == document_title and c.page_number == page_number]
    
    def search_citations(self, query: str) -> List[Citation]:
        """Search citations by text content"""
        query_lower = query.lower()
        return [c for c in self.citations 
                if query_lower in c.text.lower() or query_lower in c.context.lower()]
    
    def generate_citation_report(self, case_id: str) -> Dict:
        """Generate a comprehensive citation report"""
        report = {
            'case_id': case_id,
            'total_citations': len(self.citations),
            'by_type': {},
            'by_document': {},
            'legal_citations': [],
            'medical_citations': [],
            'financial_citations': []
        }
        
        # Count by type
        for citation in self.citations:
            citation_type = citation.citation_type
            if citation_type not in report['by_type']:
                report['by_type'][citation_type] = 0
            report['by_type'][citation_type] += 1
        
        # Count by document
        for citation in self.citations:
            doc = citation.source_document
            if doc not in report['by_document']:
                report['by_document'][doc] = 0
            report['by_document'][doc] += 1
        
        # Separate by type
        report['legal_citations'] = [c.to_dict() for c in self.get_citations_by_type('legal')]
        report['medical_citations'] = [c.to_dict() for c in self.get_citations_by_type('medical')]
        report['financial_citations'] = [c.to_dict() for c in self.get_citations_by_type('financial')]
        
        return report
    
    def save_citations(self, filepath: str):
        """Save citations to JSON file"""
        data = {
            'citations': [c.to_dict() for c in self.citations],
            'page_content_map': self.page_content_map
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_citations(self, filepath: str):
        """Load citations from JSON file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.citations = [Citation(**c) for c in data.get('citations', [])]
            self.page_content_map = data.get('page_content_map', {})


# Enhanced RAG integration
class CitationAwareRAG:
    """RAG system with citation tracking"""
    
    def __init__(self):
        self.citation_tracker = CitationTracker()
    
    def process_documents_with_citations(self, documents: List[Tuple[str, str, str]]):
        """Process documents and extract citations"""
        for doc_id, file_path, doc_title in documents:
            if file_path.endswith('.pdf'):
                self.citation_tracker.process_document(file_path, doc_title)
    
    def get_cited_response(self, query: str, context: str) -> str:
        """Generate response with proper citations"""
        # Find relevant citations
        relevant_citations = self.citation_tracker.search_citations(query)
        
        # Format response with citations
        response = f"Based on the analysis of the documents:\n\n{context}\n\n"
        
        if relevant_citations:
            response += "**Citations:**\n"
            for citation in relevant_citations[:5]:  # Top 5 citations
                response += f"- {citation}\n"
        
        return response


# Example usage
if __name__ == "__main__":
    # Initialize citation tracker
    tracker = CitationTracker()
    
    # Process sample documents with correct paths
    sample_docs = [
        ("sample_docs/2024-PI-001/medical_records_dr_jones.pdf", "Medical Records - Dr. Jones"),
        ("sample_docs/2024-PI-001/police_report_incident_789.pdf", "Police Report - SPD #789"),
        ("sample_docs/2024-PI-001/wage_statements_2024.pdf", "Wage Statements - Pacific Construction"),
        ("sample_docs/2024-PI-001/insurance_correspondence.pdf", "Insurance Communications - ABC Insurance")
    ]
    
    for pdf_path, doc_title in sample_docs:
        if os.path.exists(pdf_path):
            print(f"\nProcessing: {doc_title}")
            citations = tracker.process_document(pdf_path, doc_title)
            print(f"Found {len(citations)} citations in {doc_title}")
        else:
            print(f"Document not found: {pdf_path}")
    
    # Generate report
    report = tracker.generate_citation_report("2024-PI-001")
    print(f"\nCitation Report:")
    print(f"Total citations: {report['total_citations']}")
    print(f"By type: {report['by_type']}")
    print(f"By document: {report['by_document']}")
    
    # Save citations
    tracker.save_citations("citations.json")
    print(f"\nCitations saved to: citations.json") 