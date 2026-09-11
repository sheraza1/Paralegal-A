"""
Command Library for Legal AI RAG System
Provides easy-to-use functions for common operations
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import argparse

# Import our existing modules
from mcp_server import MCPServer
from rag_assistant import generate_rag_response, retrieve_relevant_docs, build_faiss_index, ensure_section_index, retrieve_relevant_sections
# Avoid importing heavy PDF generator at module import time
def _generate_demand_letter_pdf(case_id: str, output_path: str = "demand_letter.pdf"):
    from generate_report import generate_demand_letter_pdf
    return generate_demand_letter_pdf(case_id, output_path)


class LegalAIClient:
    """Main client class for Legal AI RAG System"""
    
    def __init__(self, case_id: str = "2024-PI-001"):
        """Initialize the client with a case ID"""
        self.case_id = case_id
        self.mcp = MCPServer()
        
    def get_case_summary(self) -> Dict:
        """Get comprehensive case summary"""
        try:
            case = self.mcp.get_case_details(self.case_id)
            parties = self.mcp.get_party_details(self.case_id)
            incident_details = self.mcp.get_incident_details(self.case_id)
            
            return {
                'case_id': self.case_id,
                'case_type': case.get('case_type', 'Unknown'),
                'status': case.get('status', 'Unknown'),
                'date_filed': case.get('date_filed', 'Unknown'),
                'case_summary': case.get('case_summary', 'No summary available'),
                'parties': parties,
                'incident_details': incident_details
            }
        except Exception as e:
            return {'error': f'Failed to get case summary: {e}'}
    
    def get_financial_summary(self) -> Dict:
        """Get comprehensive financial summary"""
        try:
            expense_breakdown = self.mcp.get_expense_breakdown(self.case_id)
            wage_info = self.mcp.get_wage_loss_info(self.case_id)
            pain_suffering = self.mcp.get_pain_suffering_estimate(self.case_id)
            
            return {
                'total_medical_expenses': expense_breakdown['total_medical'],
                'future_medical_expenses': expense_breakdown['future_medical'],
                'total_wage_loss': expense_breakdown['total_wage_loss'],
                'future_wage_loss': expense_breakdown['future_wages'],
                'pain_and_suffering': pain_suffering,
                'total_damages': (expense_breakdown['total_medical'] + 
                                expense_breakdown['future_medical'] + 
                                expense_breakdown['total_wage_loss'] + 
                                expense_breakdown['future_wages'] + 
                                pain_suffering),
                'expense_breakdown': expense_breakdown
            }
        except Exception as e:
            return {'error': f'Failed to get financial summary: {e}'}
    
    def get_medical_info(self) -> Dict:
        """Get comprehensive medical information"""
        try:
            medical_info = self.mcp.get_detailed_medical_info(self.case_id)
            medical_providers = self.mcp.get_medical_providers(self.case_id)
            injury_details = self.mcp.get_injury_details(self.case_id)
            
            return {
                'medical_providers': medical_providers,
                'injury_details': injury_details,
                'medical_info': medical_info,
                'total_expenses': sum([float(expense['amount']) for expense in 
                                     self.mcp.get_expense_breakdown(self.case_id)['medical_treatment']])
            }
        except Exception as e:
            return {'error': f'Failed to get medical info: {e}'}
    
    def get_liability_analysis(self) -> Dict:
        """Get liability analysis information"""
        try:
            police_info = self.mcp.get_police_report_details(self.case_id)
            incident_details = self.mcp.get_incident_details(self.case_id)
            
            return {
                'police_report': police_info,
                'incident_details': incident_details,
                'fault_determination': police_info[0][1].get('fault_determination', 'Unknown') if police_info else 'Unknown'
            }
        except Exception as e:
            return {'error': f'Failed to get liability analysis: {e}'}
    
    def search_documents(self, query: str, top_k: int = 3) -> List[str]:
        """Search documents using RAG"""
        try:
            return retrieve_relevant_docs(query, top_k)
        except Exception as e:
            return [f'Error searching documents: {e}']
    
    def ask_question(self, question: str) -> str:
        """Ask a question and get RAG response"""
        try:
            return generate_rag_response(question)
        except Exception as e:
            return f'Error getting response: {e}'
    
    def generate_demand_letter(self, output_path: str = "demand_letter.pdf") -> str:
        """Generate demand letter PDF"""
        try:
            _generate_demand_letter_pdf(self.case_id, output_path)
            return f"Demand letter generated successfully: {output_path}"
        except Exception as e:
            return f'Error generating demand letter: {e}'
    
    def rebuild_index(self) -> str:
        """Rebuild the FAISS index"""
        try:
            from rag_assistant import get_documents_from_db
            docs = get_documents_from_db()
            build_faiss_index(docs)
            return f"Index rebuilt successfully with {len(docs)} documents"
        except Exception as e:
            return f'Error rebuilding index: {e}'


# Command functions for easy access
def get_case_summary(case_id: str = "2024-PI-001") -> Dict:
    """Get case summary for a specific case"""
    client = LegalAIClient(case_id)
    return client.get_case_summary()


def get_financial_summary(case_id: str = "2024-PI-001") -> Dict:
    """Get financial summary for a specific case"""
    client = LegalAIClient(case_id)
    return client.get_financial_summary()


def get_medical_info(case_id: str = "2024-PI-001") -> Dict:
    """Get medical information for a specific case"""
    client = LegalAIClient(case_id)
    return client.get_medical_info()


def get_liability_analysis(case_id: str = "2024-PI-001") -> Dict:
    """Get liability analysis for a specific case"""
    client = LegalAIClient(case_id)
    return client.get_liability_analysis()


def search_documents(query: str, case_id: str = "2024-PI-001", top_k: int = 3) -> List[str]:
    """Search documents for a specific case"""
    client = LegalAIClient(case_id)
    return client.search_documents(query, top_k)


def ask_question(question: str, case_id: str = "2024-PI-001") -> str:
    """Ask a question about a specific case"""
    client = LegalAIClient(case_id)
    return client.ask_question(question)


def generate_demand_letter(case_id: str = "2024-PI-001", output_path: str = "demand_letter.pdf") -> str:
    """Generate demand letter for a specific case"""
    client = LegalAIClient(case_id)
    return client.generate_demand_letter(output_path)


def rebuild_index() -> str:
    """Rebuild the FAISS index"""
    client = LegalAIClient()
    return client.rebuild_index()


# Example usage functions
def demo_case_summary():
    """Demonstrate case summary functionality"""
    print("=== Case Summary Demo ===")
    summary = get_case_summary()
    print(f"Case ID: {summary.get('case_id')}")
    print(f"Case Type: {summary.get('case_type')}")
    print(f"Status: {summary.get('status')}")
    print(f"Summary: {summary.get('case_summary')}")


def demo_financial_summary():
    """Demonstrate financial summary functionality"""
    print("=== Financial Summary Demo ===")
    financial = get_financial_summary()
    print(f"Total Medical Expenses: ${financial.get('total_medical_expenses', 0):,.2f}")
    print(f"Total Wage Loss: ${financial.get('total_wage_loss', 0):,.2f}")
    print(f"Pain and Suffering: ${financial.get('pain_and_suffering', 0):,.2f}")
    print(f"Total Damages: ${financial.get('total_damages', 0):,.2f}")


def demo_rag_questions():
    """Demonstrate RAG question answering"""
    print("=== RAG Questions Demo ===")
    questions = [
        "What are the total medical expenses?",
        "Who are the medical providers?",
        "What are the incident details?",
        "What is the liability determination?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = ask_question(question)
        print(f"A: {answer[:200]}...")


def cli():
    parser = argparse.ArgumentParser(description="Legal AI CLI: query case sections via prebuilt FAISS index")
    parser.add_argument("statement", type=str, help="Natural-language statement or query to search relevant case sections")
    parser.add_argument("--case-id", dest="case_id", default="2024-PI-001", help="Case ID to use (default: 2024-PI-001)")
    parser.add_argument("--top-k", dest="top_k", type=int, default=5, help="Number of top sections to return (default: 5)")
    parser.add_argument("--out", dest="out", default=None, help="Optional path to save the retrieved sections as text")
    args = parser.parse_args()

    # Ensure per-case section-level FAISS index exists (uses cached files if present)
    ensure_section_index(args.case_id)

    # Retrieve from already generated index
    results = retrieve_relevant_sections(args.statement, top_k=args.top_k) or []

    # Pretty print
    lines = []
    lines.append(f"Case: {args.case_id}")
    lines.append(f"Query: {args.statement}")
    lines.append(f"Top {len(results)} sections:\n")
    for i, sec in enumerate(results, 1):
        title = sec.get("title", "<no title>")
        src = sec.get("source", "<unknown>")
        text = sec.get("text", "")
        snippet = (text[:600] + "...") if len(text) > 600 else text
        lines.append(f"[{i}] {title} (source: {src})\n{snippet}\n")

    output = "\n".join(lines)
    print(output)

    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    # If called as a script, use CLI mode to query the per-case section index
    cli() 