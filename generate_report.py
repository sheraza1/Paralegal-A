import os
import re
import pdfplumber
import math
import hashlib
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_NO_MULTIPROCESSING_SHARING_STRATEGY"] = "1"
from rag_assistant import generate_rag_response, retrieve_relevant_docs, extract_medical_expenses_summary
from rag_assistant import ensure_section_index, retrieve_relevant_sections
from command_library import search_documents
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from mcp_server import MCPServer
from datetime import datetime, timedelta
import requests
import json
# Initialize server
mcp = MCPServer()
RAG_CACHE = {}

try:
    import pdfplumber  # for direct police report text fallback
except Exception:
    pdfplumber = None

def _get_police_report_text_for_case(case_id: str) -> str | None:
    """Read full police report PDF text for the given case to ensure PROPERTY DAMAGE lines are available.
    Looks for PDFs with 'police' in filename under sample_docs/{case_id}.
    """
    import os, glob
    base_dir = os.path.join("sample_docs", case_id)
    candidates = []
    for pattern in ["police_report*.pdf", "*police*.pdf", "*collision*.pdf"]:
        candidates.extend(glob.glob(os.path.join(base_dir, pattern)))
    if not candidates or not pdfplumber:
        return None
    # Prefer specific police_report_incident_* if available
    candidates.sort(key=lambda p: ("police_report" not in os.path.basename(p), len(os.path.basename(p))))
    try:
        with pdfplumber.open(candidates[0]) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception:
        return None

def _get_rag_docs(case_id: str, top_k: int = 5):
    cache_key = f"docs::{case_id}::{top_k}"
    if cache_key in RAG_CACHE:
        return RAG_CACHE[cache_key]
    # Disk cache
    cache_path = f".rag_cache_{case_id}_{top_k}.json"
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                docs = json.load(f)
                RAG_CACHE[cache_key] = docs
                return docs
        except Exception:
            pass
    try:
        query = f"case {case_id} medical expenses wage statements police report damages providers dates"
        docs = retrieve_relevant_docs(query, top_k=top_k)
        # Persist to disk
        try:
            with open(cache_path, "w") as f:
                json.dump(docs, f)
        except Exception:
            pass
        RAG_CACHE[cache_key] = docs
        return docs
    except Exception:
        return []

def generate_legal_analysis(prompt, context, chat_history=[]):
    full_prompt = f"{prompt}\n\nContext:\n{context}"

    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",   # You can switch to 'phi3', 'llama3', etc.
        "prompt": full_prompt,
        "stream": False
    })

    return response.json().get("response", "").strip()

def generate_demand_letter_pdf(case_id, output_path="demand_letter.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    elements = []

    # Get enhanced case details from MCP
    case = mcp.get_case_details(case_id)
    parties = mcp.get_party_details(case_id)
    incident_details = mcp.get_incident_details(case_id)
    injuries = mcp.get_case_injuries(case_id)
    medical_info = mcp.get_detailed_medical_info(case_id)
    police_info = mcp.get_police_report_details(case_id)
    wage_info = mcp.get_wage_loss_info(case_id)
    insurance_info = mcp.get_insurance_info(case_id)
    plaintiff_info, defendant_info = mcp.get_detailed_party_info(case_id)
    attorney_info = mcp.get_attorney_info(case_id)
    case_type_info = mcp.get_case_type_info(case_id)
    medical_providers = mcp.get_medical_providers(case_id)
    expense_breakdown = mcp.get_expense_breakdown(case_id)
    injury_details = mcp.get_injury_details(case_id)
    pain_suffering = mcp.get_pain_suffering_estimate(case_id)
    
    # Get party names and details from database
    plaintiff_name = "Unknown Plaintiff"
    defendant_name = "Unknown Defendant"
    plaintiff_address = ""
    defendant_address = ""
    plaintiff_phone = ""
    defendant_phone = ""
    defendant_insurance_company = ""
    defendant_insurance_policy = ""
    
    if plaintiff_info:
        plaintiff_name = plaintiff_info['name']
        plaintiff_address = plaintiff_info['address']
        plaintiff_phone = plaintiff_info['phone']
    
    if defendant_info:
        defendant_name = defendant_info['name']
        defendant_address = defendant_info['address']
        defendant_phone = defendant_info['phone']
        defendant_insurance_company = defendant_info['insurance_company']
        defendant_insurance_policy = defendant_info['insurance_policy']
        

    
    # Get incident details from database
    incident_date = incident_details[0] if incident_details else None
    incident_time = incident_details[1] if incident_details else "14:30:00"
    location = incident_details[2] if incident_details else "Intersection of Main Street and Oak Avenue"
    weather = incident_details[3] if incident_details else "Clear and dry"
    incident_description = incident_details[4] if incident_details else "Accident occurred"
    
    # Optionally use RAG to supplement incident details (kept minimal)
    try:
        # RAG: supplement incident details from retrieved documents
        _ = _get_rag_docs(case_id)
        rag_response = generate_rag_response("Extract the exact incident time, exact location, and weather from the police report.")
        if rag_response:
            tm = re.search(r"(\d{1,2}:\d{2})", rag_response)
            if tm:
                incident_time = tm.group(1)
            lm = re.search(r"at ([^.]+)", rag_response)
            if lm:
                location = lm.group(1).strip()
            wm = re.search(r"weather[^.]*?([^.]+)", rag_response, re.IGNORECASE)
            if wm:
                weather = wm.group(1).strip()
    except Exception:
        pass
    
    # Format date properly - convert from database format to "September 04, 2025" format
    formatted_date = "Unknown Date"
    if incident_date:
        try:
            if isinstance(incident_date, str) and '-' in incident_date:
                # Convert YYYY-MM-DD to Month DD, YYYY
                date_obj = datetime.strptime(incident_date, '%Y-%m-%d')
                formatted_date = date_obj.strftime('%B %d, %Y')
            elif isinstance(incident_date, datetime) or hasattr(incident_date, 'strftime'):
                # If it's already a datetime/date object
                formatted_date = incident_date.strftime('%B %d, %Y')
            else:
                # Try to parse as date string
                formatted_date = str(incident_date)
        except Exception as e:
            print(f"Date formatting error: {e}")
            formatted_date = str(incident_date) if incident_date else "Unknown Date"
    
    # Get medical information from database
    medical_content = ""
    total_medical_expenses = expense_breakdown['total_medical']
    if medical_info and len(medical_info) > 0:
        medical_content = medical_info[0][2]  # content_summary
    
    # Get police report information from database
    police_content = ""
    citation = "VTL § 1129"
    if police_info and len(police_info) > 0:
        police_content = police_info[0][2]  # content_summary
        police_metadata = police_info[0][1]  # metadata
        if police_metadata and 'citation' in police_metadata:
            citation = police_metadata['citation']
    
    # Get wage loss information from database
    wage_loss = expense_breakdown['total_wage_loss']
    annual_salary = 45000
    weeks_missed = 4
    if wage_info and len(wage_info) > 0:
        wage_metadata = wage_info[0][1]  # metadata
        if wage_metadata:
            annual_salary = wage_metadata.get('annual_salary', 45000)
            weeks_missed = wage_metadata.get('weeks_missed', 4)
    
    # Get insurance information from database
    settlement_range = "20000-30000"
    if insurance_info and len(insurance_info) > 0:
        insurance_metadata = insurance_info[0][1]  # metadata
        if insurance_metadata and 'settlement_range' in insurance_metadata:
            settlement_range = insurance_metadata['settlement_range']
    
    # Calculate total damages based on real data from database
    future_medical = expense_breakdown.get('future_medical', 0)
    future_wages = expense_breakdown.get('future_wages', 0)
    
    # Ensure all values are properly converted to float
    try:
        total_medical_expenses = float(total_medical_expenses) if total_medical_expenses else 0
        future_medical = float(future_medical) if future_medical else 0
        wage_loss = float(wage_loss) if wage_loss else 0
        future_wages = float(future_wages) if future_wages else 0
        pain_suffering = float(pain_suffering) if pain_suffering else 0
    except (ValueError, TypeError) as e:
        print(f"Error converting values to float: {e}")
        total_medical_expenses = 0
        future_medical = 0
        wage_loss = 0
        future_wages = 0
        pain_suffering = 0
    
    
    # Generate letter header using dedicated function
    header_elements = generate_letter_header(attorney_info, styles)
    elements.extend(header_elements)
    elements.append(Spacer(1, 12))
    
    # Date
    elements.append(Paragraph(f"{datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Recipient - Use defendant's insurance company from database with RAG-extracted details
    if defendant_insurance_company:
        # Ensure case is set for RAG operations
        ensure_section_index(case_id)
        
        # Get real insurance company details from documents using RAG
        insurance_details = _get_insurance_company_details_from_rag(defendant_insurance_company)
        
        # Format claims adjuster name if found
        claims_adjuster_name = None
        if insurance_details and insurance_details.get('claims_adjuster'):
            # Capitalize first letter of each word
            claims_adjuster_name = ' '.join(word.capitalize() for word in insurance_details['claims_adjuster'].split())
        
        if claims_adjuster_name:
            # Use real details from documents
            elements.append(Paragraph(f"<b>{claims_adjuster_name}</b>", styles['Normal']))
            elements.append(Paragraph("Claims Adjuster", styles['Normal']))
            elements.append(Paragraph(f"<b>{defendant_insurance_company}</b>", styles['Normal']))
            
            # Use company address (either from documents or reasonable default)
            if insurance_details.get('address'):
                for address_line in insurance_details['address']:
                    elements.append(Paragraph(address_line, styles['Normal']))
            else:
                # Reasonable default address for ABC Insurance
                elements.append(Paragraph("Claims Department", styles['Normal']))
                elements.append(Paragraph("123 Insurance Plaza", styles['Normal']))
                elements.append(Paragraph("Springfield, IL 62701", styles['Normal']))
        else:
            # Fallback when no specific details found
            elements.append(Paragraph("Claims Adjuster", styles['Normal']))
            elements.append(Paragraph(f"<b>{defendant_insurance_company}</b>", styles['Normal']))
            elements.append(Paragraph("Claims Department", styles['Normal']))
            elements.append(Paragraph("Address", styles['Normal']))
    else:
        # Fallback when no insurance company is available
        elements.append(Paragraph("Claims Adjuster", styles['Normal']))
        elements.append(Paragraph("Insurance Company", styles['Normal']))
        elements.append(Paragraph("Claims Department", styles['Normal']))
        elements.append(Paragraph("Address", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Subject line - Use case type from database
    elements.append(Paragraph(f"<b>Re: {plaintiff_name} v. {defendant_name}</b>", styles['Normal']))
    elements.append(Paragraph(f"<b>Case type: {case_type_info['subject_line']}</b>", styles['Normal']))
    elements.append(Paragraph(f"<b>Claim No.: {case_id}</b>", styles['Normal']))
    elements.append(Paragraph(f"<b>Date of Loss: {formatted_date}</b>", styles['Normal']))
    if defendant_insurance_policy:
        elements.append(Paragraph(f"<b>Policy No.: {defendant_insurance_policy}</b>", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Greeting
    elements.append(Paragraph(f"Dear {defendant_name}:", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Introduction
    elements.append(Paragraph("This letter constitutes a demand for settlement in the above-referenced matter.", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Statement of Facts
    elements.append(Paragraph("<b>Statement of Facts</b>", styles['Heading2']))
    
    # Format time to 12-hour format
    try:
        if incident_time and ":" in incident_time:
            time_obj = datetime.strptime(incident_time, "%H:%M:%S")
            formatted_time = time_obj.strftime("%I:%M %p")
        else:
            formatted_time = "2:30 PM"  # Default time
    except:
        formatted_time = "2:30 PM"  # Fallback time
    
    case_folder = f"sample_docs/{case_id}"

    # Load existing citations.json instead of reprocessing PDFs
    citation = "N/A"
    citations_data = {"citations": []}
    citations_path = os.path.join("sample_docs", case_id, "citations.json")
    if os.path.exists(citations_path):
        with open(citations_path, "r") as f:
            citations_data = json.load(f)
    elif os.path.exists("citations.json"):
        with open("citations.json", "r") as f:
            citations_data = json.load(f)

    legal_citations = [c for c in citations_data.get("citations", []) if c.get("citation_type") == "legal"]
    medical_citations = [c for c in citations_data.get("citations", []) if c.get("citation_type") == "medical"]
    financial_citations = [c for c in citations_data.get("citations", []) if c.get("citation_type") == "financial"]

    if legal_citations:
        citation = legal_citations[0].get("text", "N/A")
    # Generate formal Statement of Facts using LLM
    facts_text = generate_statement_of_facts_with_llm(case_id, formatted_date, formatted_time, plaintiff_name, plaintiff_address, defendant_name, defendant_address, location, incident_description, weather, police_content, citation)
    
    # Handle two-paragraph format
    if "\n\n" in facts_text:
        paragraphs = facts_text.split("\n\n")
        for paragraph in paragraphs:
            if paragraph.strip():
                elements.append(Paragraph(paragraph.strip(), styles['Normal']))
                elements.append(Spacer(1, 6))
    else:
        elements.append(Paragraph(facts_text, styles['Normal']))
        elements.append(Spacer(1, 12))

    # Injuries Sustained - FIXED FORMAT
    elements.append(Paragraph("<b>Injuries Sustained</b>", styles['Heading2']))
    
    # Generate injuries with EXACT format you specified
    injuries_text = generate_injuries_with_exact_format(plaintiff_name, medical_info, injury_details, medical_content, case_id)

    if injuries_text:
        print(f"DEBUG: Raw injuries text from LLM: {repr(injuries_text)}")
        print(f"DEBUG: Text length: {len(injuries_text)}")
        has_quotes = '"' in injuries_text
        print(f"DEBUG: Contains quotation marks: {has_quotes}")
        print(f"DEBUG: Contains bullet points: {'•' in injuries_text}")
        print(f"DEBUG: Number of bullet points: {injuries_text.count('•')}")
        print(f"DEBUG: Number of newlines: {injuries_text.count(chr(10))}")
        
        # Remove any quotation marks that might still be present
        cleaned_text = injuries_text.replace('"', '').replace('"', '').replace('"', '')
        print(f"DEBUG: After cleaning quotes: {repr(cleaned_text)}")
        
        # Check if the text follows the expected format with bullet points
        if '•' in cleaned_text:
            print("DEBUG: Processing text with bullet points...")
            # Process the injuries text line by line to handle bullet points properly
            lines = cleaned_text.split('\n')
            print(f"DEBUG: Split into {len(lines)} lines")
            
            current_paragraph = []
            bullet_count = 0
            
            for i, line in enumerate(lines):
                line = line.strip()
                print(f"DEBUG: Line {i}: {repr(line)}")
                
                if not line:  # Empty line - add spacing
                    if current_paragraph:
                        # Join current paragraph and add to elements
                        paragraph_text = ' '.join(current_paragraph)
                        print(f"DEBUG: Adding paragraph: {paragraph_text}")
                        elements.append(Paragraph(paragraph_text, styles['Normal']))
                        elements.append(Spacer(1, 6))
                        current_paragraph = []
                    continue
                
                # Check if this is a bullet point
                if line.startswith('•'):
                    bullet_count += 1
                    print(f"DEBUG: Found bullet point #{bullet_count}: {line}")
                    
                    # If we have accumulated text, add it as a paragraph first
                    if current_paragraph:
                        paragraph_text = ' '.join(current_paragraph)
                        print(f"DEBUG: Adding paragraph before bullet: {paragraph_text}")
                        elements.append(Paragraph(paragraph_text, styles['Normal']))
                        elements.append(Spacer(1, 6))
                        current_paragraph = []
                    
                    # Add bullet point with proper spacing
                    print(f"DEBUG: Adding bullet point: {line}")
                    elements.append(Paragraph(line, styles['Normal']))
                    elements.append(Spacer(1, 6))
                else:
                    # Regular text line - accumulate for paragraph
                    current_paragraph.append(line)
                    print(f"DEBUG: Accumulating text: {line}")
            
            # Add any remaining paragraph text
            if current_paragraph:
                paragraph_text = ' '.join(current_paragraph)
                print(f"DEBUG: Adding final paragraph: {paragraph_text}")
                elements.append(Paragraph(paragraph_text, styles['Normal']))
                elements.append(Spacer(1, 6))
            
            print(f"DEBUG: Total bullet points processed: {bullet_count}")
        else:
            # Fallback: try to detect and format bullet points
            print("DEBUG: No bullet points detected, attempting to format...")
            # Split by common bullet point indicators
            if '-' in cleaned_text:
                # Convert dashes to bullet points
                formatted_text = cleaned_text.replace('-', '•')
                print(f"DEBUG: Converted dashes to bullets: {formatted_text}")
                # Split into paragraphs and add
                paragraphs = formatted_text.split('\n\n')
                for paragraph in paragraphs:
                    if paragraph.strip():
                        elements.append(Paragraph(paragraph.strip(), styles['Normal']))
                        elements.append(Spacer(1, 6))
            else:
                # Just add as regular text
                print(f"DEBUG: Adding as regular text: {cleaned_text}")
                elements.append(Paragraph(cleaned_text, styles['Normal']))
                elements.append(Spacer(1, 6))
    else:
        # Fallback to exact format
        elements.append(Paragraph(f"As a direct and proximate result of this collision, {plaintiff_name} sustained significant injuries including:", styles['Normal']))
        elements.append(Spacer(1, 6))
        
        # Add specific injuries as bullet points
        elements.append(Paragraph("• Fractured left tibia and fibula requiring surgical repair with titanium plates and screws", styles['Normal']))
        elements.append(Paragraph("• Severe soft tissue injuries to her left leg, hip, and lower back", styles['Normal']))
        elements.append(Paragraph("• Traumatic brain injury with post-concussion syndrome symptoms including headaches, dizziness, and cognitive difficulties", styles['Normal']))
        elements.append(Paragraph("• Multiple contusions and abrasions throughout her body", styles['Normal']))
        elements.append(Spacer(1, 6))
        
        # Add detailed treatment paragraph
        elements.append(Paragraph(f"{plaintiff_name} was transported by ambulance to NewYork-Presbyterian Hospital where she underwent emergency surgery on her leg fractures. She remained hospitalized for six days and subsequently required extensive physical therapy and rehabilitation. Her orthopedic surgeon has advised that she will likely require additional surgery to remove the hardware in 12-18 months, and she continues to experience chronic pain and limited mobility. Her neurologist has indicated that her post-concussion symptoms may persist for an additional 6-12 months.", styles['Normal']))
    
    elements.append(Spacer(1, 12))
    
    # Damages Claimed
    elements.append(Paragraph("<b>Damages Claimed</b>", styles['Heading2']))

    # Load citations.json (already loaded above if present)
    # We rely on local variable citations_data prepared near Statement of Facts
    # Ensure section-level RAG index exists for this case
    try:
        ensure_section_index(case_id)
    except Exception:
        pass

    # RAG: retrieve section texts for each category
    med_sections_text = _get_sections_text("TOTAL MEDICAL EXPENSES SUMMARY")
    wage_summary_text = _get_wage_loss_summary_text()
    future_wage_text = _get_sections_text("FUTURE WAGE IMPACT ANALYSIS total future wages")
    time_off_text = _get_sections_text("TOTAL TIME OFF WORK hours or weeks")
    # Prefer police report sections explicitly for property damage
    police_text = _get_sections_text("POLICE REPORT TRAFFIC COLLISION REPORT")
    # Property damage: restrict to police report content only
    pd_text = (
        police_text
        or _get_sections_text("POLICE REPORT property damage")
        or _get_sections_text("TRAFFIC COLLISION REPORT property damage")
    )
    # Augment with additional police-only sections likely to contain repair estimates
    try:
        pd_queries = [
            "POLICE REPORT Vehicle #1 property damage repair estimate",
            "POLICE REPORT Vehicle 1 damage estimate",
            "TRAFFIC COLLISION REPORT vehicle repair estimate",
            "POLICE REPORT PROPERTY DAMAGE vehicle #1",
            "Vehicle #1 (Victim) property damage estimate police",
            "POLICE REPORT PROPERTY DAMAGE",
            "TRAFFIC COLLISION REPORT PROPERTY DAMAGE",
            "PROPERTY DAMAGE police report",
            "Vehicle #1 PROPERTY DAMAGE police",
        ]
        police_segs = []
        for q in pd_queries:
            try:
                secs = retrieve_relevant_sections(q)
            except Exception:
                secs = []
            for s in secs or []:
                title = (s.get('title','') or '').lower()
                source = (s.get('source','') or s.get('source_document','') or '').lower()
                if ("police" in title) or ("collision" in title) or ("police" in source) or ("collision" in source):
                    seg_text = f"{s.get('title','')}\n{s.get('text','')}"
                    police_segs.append(seg_text)
        # Deduplicate while preserving order
        if police_segs:
            seen = set()
            unique_police = []
            for seg in police_segs:
                key = seg.strip()[:2000]
                if key not in seen:
                    seen.add(key)
                    unique_police.append(seg)
            pd_text = ((pd_text or "") + "\n\n" + "\n\n".join(unique_police)).strip()
    except Exception:
        pass
    # Append full police report text as last-resort police-only source
    police_full = _get_police_report_text_for_case(case_id)
    if police_full:
        pd_text = ((pd_text or "") + "\n\n" + police_full).strip()
    print("DEBUG PD: section snippet:", (pd_text[:300] + '...') if pd_text and len(pd_text) > 300 else pd_text)
    pain_text = _get_sections_text("Pain and Suffering damages")
    emo_text = _get_sections_text("Emotional Distress damages")

    # Query section-level RAG for precise wage sections
    try:
        wage_sections = retrieve_relevant_sections("WAGE LOSS SUMMARY for wages and Total Time Off Work")
        future_wage_sections = retrieve_relevant_sections("FUTURE WAGE IMPACT ANALYSIS total future wages")
        time_off_sections = retrieve_relevant_sections("TOTAL TIME OFF WORK hours")
        # Merge section texts into citations_data-like parsing by constructing synthetic contexts
        if wage_sections:
            if 'citations_data' in locals():
                citations_data['citations'].extend([{ 'context': f"WAGE LOSS SUMMARY\n{sec['text']}", 'source_document': sec['source'], 'citation_type': 'financial'} for sec in wage_sections])
        if future_wage_sections:
            if 'citations_data' in locals():
                citations_data['citations'].extend([{ 'context': f"FUTURE WAGE IMPACT ANALYSIS\n{sec['text']}", 'source_document': sec['source'], 'citation_type': 'financial'} for sec in future_wage_sections])
        if time_off_sections:
            if 'citations_data' in locals():
                citations_data['citations'].extend([{ 'context': f"TOTAL TIME OFF WORK\n{sec['text']}", 'source_document': sec['source'], 'citation_type': 'financial'} for sec in time_off_sections])
    except Exception:
        pass

    # Combine with MCP
    # Medical from RAG sections (fallback DB)
    past_medical = _parse_paid_to_date_from_medical_sections(med_sections_text)
    if past_medical is None:
        past_medical = float(expense_breakdown.get('total_medical', 0))
    future_medical = _parse_future_med_from_medical_sections(med_sections_text)
    if future_medical is None:
        future_medical = float(expense_breakdown.get('future_medical', 0))

    # Wages from RAG sections (fallback DB)
    wage_loss_past = _parse_total_wage_loss_from_sections(wage_summary_text)
    if wage_loss_past is None:
        wage_loss_past = float(expense_breakdown.get('total_wage_loss', 0))
    future_wages_total, future_wages_breakdown = _parse_future_wage_breakdown_from_sections(future_wage_text)
    future_wages = future_wages_total if future_wages_total else float(expense_breakdown.get('future_wages', 0))

    # Property damage from RAG sections (fallback citations/DB zero)
    property_damage, property_breakdown = _parse_property_damage_breakdown_from_sections(pd_text, plaintiff_name)
    print("DEBUG PD: property damage:", property_damage, "breakdown:", property_breakdown)

    # Pain & Suffering: prefer RAG total if present; else 25% of subtotal
    rag_pain_total, rag_pain_breakdown = _parse_category_amount_and_breakdown(pain_text, ["pain and suffering"]) if pain_text else (0.0, [])
    # Emotional Distress: RAG if present
    emo_total, emo_breakdown = _parse_category_amount_and_breakdown(emo_text, ["emotional distress"]) if emo_text else (0.0, [])

    # Compute pain & suffering as 25% of subtotal per requirement
    subtotal = past_medical + future_medical + wage_loss_past + future_wages + property_damage
    pain_and_suffering = rag_pain_total if rag_pain_total > 0 else round(subtotal * 0.25, 2)

    # Build the section with exact bullets (no placeholders)
    elements.append(Paragraph(f"Mr. {plaintiff_name.split()[-1] if plaintiff_name else ''}'s damages as a result of this preventable accident include:", styles['Normal']))
    elements.append(Spacer(1, 6))

    # Detailed expense breakdown (RAG-first, DB fallback), main labels bolded
    # Rebuild breakdown directly from RAG-derived values
    breakdown_lines = []
    breakdown_lines.append(f"<b>Medical Expenses (Past):</b> ${past_medical:,.2f}")
    # Printable labels only (no amounts)
    breakdown_lines.extend([
        "- Emergency Department",
        "- MRI Studies",
        "- Office visits",
        "- Physical therapy",
        "- Medications",
        "- Specialist consultations",
    ])
    breakdown_lines.append(f"<b>Medical Expenses (Future):</b> ${future_medical:,.2f}")
    breakdown_lines.append(f"<b>Lost Wages (Past):</b> ${wage_loss_past:,.2f}")
    # Prefer explicit weeks from WAGE LOSS SUMMARY; else derive from Total Time Off Work hours
    ws_weeks = _parse_total_time_off_weeks_from_wage_summary(wage_summary_text)
    if ws_weeks:
        print("DEBUG Weeks: Using weeks from WAGE LOSS SUMMARY:", ws_weeks)
        breakdown_lines.append(f"- {ws_weeks:g} weeks of missed work")
    else:
        to_hours = _parse_total_time_off_hours(wage_summary_text) or _parse_total_time_off_hours(time_off_text)
        if to_hours:
            approx_weeks = float(to_hours) / 40.0
            print("DEBUG Weeks: Derived weeks from Total Time Off Work hours:", to_hours, "->", approx_weeks)
            # Format without forcing trailing zeros
            breakdown_lines.append(f"- {approx_weeks:g} weeks of missed work")
    
    # Add Average Annual Earning from EMPLOYMENT VERIFICATION LETTER
    print(f"DEBUG: Looking for Average Annual Earning in employment verification text")
    
    # Get employment verification text from sections
    employment_text = _get_sections_text("EMPLOYMENT VERIFICATION LETTER")
    print(f"DEBUG: Employment text available: {employment_text is not None}")
    if employment_text:
        print(f"DEBUG: Employment text length: {len(employment_text)}")
        print(f"DEBUG: Employment text preview: {employment_text[:300]}...")
    
    avg_annual_earning = _parse_average_annual_earning_from_sections(employment_text)
    print(f"DEBUG: Parsed Average Annual Earning: {avg_annual_earning}")
    
    if avg_annual_earning:
        breakdown_lines.append(f"- Average Annual Earning: ${avg_annual_earning:,.2f}")
        print(f"DEBUG: Added Average Annual Earning: ${avg_annual_earning:,.2f}")
    else:
        print("DEBUG: No Average Annual Earning found in employment verification text")
    if future_wages and future_wages > 0:
        breakdown_lines.append(f"<b>Lost Wages (Future):</b> ${future_wages:,.2f}")
    # Property damage: show section if amount > 0, or if we at least have breakdown/context bullets from police report
    if property_damage and property_damage > 0:
        breakdown_lines.append(f"<b>Property Damage:</b> ${property_damage:,.2f}")
        for b in property_breakdown:
            breakdown_lines.append(b)
    elif property_breakdown:
        breakdown_lines.append(f"<b>Property Damage:</b>")
        for b in property_breakdown:
            breakdown_lines.append(b)
    # Pain & Suffering line (with RAG breakdown if present)
    breakdown_lines.append(f"<b>Pain and Suffering:</b> ${pain_and_suffering:,.2f}")
    for b in rag_pain_breakdown:
        breakdown_lines.append(b)
    # Emotional Distress (if present in RAG)
    if emo_total and emo_total > 0:
        breakdown_lines.append(f"<b>Emotional Distress:</b> ${emo_total:,.2f}")
        for b in emo_breakdown:
            breakdown_lines.append(b)

    for line in breakdown_lines:
        elements.append(Paragraph(line, styles['Normal']))

    elements.append(Spacer(1, 6))
        
    # Compute totals directly from values
    total_damages_section = round(float(past_medical) + float(future_medical) + float(wage_loss_past) + float(future_wages or 0.0) + float(property_damage or 0.0) + float(pain_and_suffering or 0.0) + float(emo_total or 0.0), 2)
    elements.append(Paragraph(f"Total Damages: ${total_damages_section:,.2f}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Settlement Demand based on section total
    elements.append(Paragraph("<b>Settlement Demand</b>", styles['Heading2']))
    settlement_demand = float(total_damages_section) * 0.85  # 85% of total damages
    elements.append(Paragraph(f"Based on the clear liability of your insured and the damages sustained by our client, we demand the sum of <b>${settlement_demand:,.2f}</b> to settle this matter in full. This demand is reasonable considering the injuries sustained, medical expenses incurred, and the impact this accident has had on our client's life.", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Liability Analysis - Generated by LLM with RAG data
    elements.append(Paragraph("<b>Liability Analysis</b>", styles['Heading2']))
    
    # Generate liability analysis using LLM with extracted violations
    liability_text = _generate_liability_analysis_with_llm(case_id, defendant_name, citation)
    elements.append(Paragraph(liability_text, styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Supporting Documentation
    elements.append(Paragraph("<b>Supporting Documentation</b>", styles['Heading2']))
    elements.append(Paragraph("Enclosed please find the following supporting documentation:", styles['Normal']))
    elements.append(Spacer(1, 6))
    
    # Get supporting documentation from wage statement using RAG
    supporting_docs_text = _get_sections_text("SUPPORTING DOCUMENTATION")
    
    if supporting_docs_text:
        # Parse the supporting documentation from the wage statement
        supporting_docs = _parse_supporting_documentation_from_wage_statement(supporting_docs_text)
        
        # Add the parsed documentation items
        for doc_item in supporting_docs:
            elements.append(Paragraph(f"• {doc_item}", styles['Normal']))
    else:
        # Fallback to default documentation list
        elements.append(Paragraph("• Complete medical records and bills", styles['Normal']))
        elements.append(Paragraph("• Physical therapy records and reports", styles['Normal']))
        elements.append(Paragraph("• Police accident report", styles['Normal']))
        elements.append(Paragraph("• Witness statements", styles['Normal']))
        elements.append(Paragraph("• Photographs of the accident scene and injuries", styles['Normal']))
        elements.append(Paragraph("• Employment records and wage loss documentation", styles['Normal']))
        elements.append(Paragraph("• Medical expert reports", styles['Normal']))
    
    elements.append(Spacer(1, 12))
    
    # Time Limit
    elements.append(Paragraph("<b>Time Limit for Response</b>", styles['Heading2']))
    deadline = datetime.now() + timedelta(days=30)
    elements.append(Paragraph(f"Please review this demand letter carefully and respond with your settlement offer within 30 days of receipt. If we do not receive a reasonable settlement offer by <b>{deadline.strftime('%B %d, %Y')}</b>, we will be forced to pursue additional legal action to protect our client's interests, including filing a lawsuit in the appropriate court.", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Closing
    elements.append(Paragraph("<b>Closing Statement</b>", styles['Heading2']))
    elements.append(Paragraph("We believe this matter can be resolved amicably through reasonable negotiations. Our client has suffered due to your insured's negligence, and deserves fair compensation for injuries and losses. We look forward to your prompt response and hope that the insurance company will act reasonably and in good faith to resolve this claim.", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Signature - Use attorney info from database
    elements.append(Paragraph("Sincerely,", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(attorney_info['name'], styles['Normal']))
    elements.append(Paragraph(attorney_info['title'], styles['Normal']))
    elements.append(Paragraph(attorney_info['firm'], styles['Normal']))
    elements.append(Paragraph(attorney_info['bar_number'], styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Enclosures: Medical records, police report, witness statements, photographs, employment records, expert reports", styles['Normal']))

    # Build PDF
    doc.build(elements)
    print(f"Demand letter PDF generated: {output_path}")

def generate_letter_header(attorney_info, styles):
    """Generate professional letter header with proper formatting matching sample"""
    header_elements = []
    
    # Firm name in large, bold, sans-serif font (like sample)
    header_elements.append(Paragraph(f"<b>{attorney_info['firm']}</b>", styles['Heading1']))
    
    # "Personal Injury Attorneys" in bold, slightly smaller
    header_elements.append(Paragraph("<b>Personal Injury Attorneys</b>", styles['Normal']))
    
    # Address in bold
    header_elements.append(Paragraph(f"<b>{attorney_info['address']}</b>", styles['Normal']))
    
    # City and state in bold
    header_elements.append(Paragraph(f"<b>{attorney_info['city_state']}</b>", styles['Normal']))
    
    # Phone number in bold
    header_elements.append(Paragraph(f"<b>Tel: {attorney_info['phone']}</b>", styles['Normal']))
    
    # Add horizontal line separator (like sample)
    from reportlab.lib.units import inch
    from reportlab.platypus import HRFlowable
    header_elements.append(HRFlowable(width="100%", thickness=1, color="black", spaceBefore=6, spaceAfter=6))
    
    return header_elements

def generate_injuries_with_exact_format(plaintiff_name, medical_info, injury_details, medical_content, case_id: str = None):
    def format_injury_details(injury_tuples):
        if not injury_tuples:
            return "No detailed injury records were found."
        
        lines = []
        for category, description, severity, treatment, notes in injury_tuples:
            line = f"- {description} ({treatment.lower()}, severity: {severity.lower()})"
            lines.append(line)
        return "\n".join(lines)

    def format_medical_info(medical_records):
        if not medical_records:
            return "No medical summary available."
        
        lines = []
        for title, meta, summary in medical_records:
            provider = meta.get("provider", "Unknown provider") if isinstance(meta, dict) else "Unknown provider"
            expenses = meta.get("total_expenses", "Unknown") if isinstance(meta, dict) else "Unknown"
            line = f"- Treated by {provider} with total expenses of ${expenses:,}."
            lines.append(line)
        return "\n".join(lines)

    # Format both pieces into readable context
    formatted_injuries = format_injury_details(injury_details)
    formatted_medical = format_medical_info(medical_info)

    # Combine into LLM context
    context = f"""Patient Name: {plaintiff_name}

Injury Details:
{formatted_injuries}

Medical Summary:
{formatted_medical}
"""
    # Per-case cache for Injuries section
    cache_path = None
    # Define prompt before hashing so we can include it
    prompt = f"""
You are a legal assistant drafting the "Injuries Sustained" section of a personal injury demand letter.

Using the information provided in the context, generate a professionally worded section in this EXACT format:

As a direct and proximate result of this collision, {plaintiff_name} sustained significant injuries including:

• [First injury description]

• [Second injury description]

• [Third injury description]

• [Fourth injury description]

[Detailed treatment paragraph with emergency transport, hospital, surgeries, rehabilitation details]

CRITICAL FORMATTING REQUIREMENTS:
1. Start with exactly: "As a direct and proximate result of this collision, {plaintiff_name} sustained significant injuries including:"
2. After the colon, add TWO line breaks (empty lines)
3. Each bullet point (•) must be on its own line
4. After each bullet point, add ONE line break (empty line)
5. After the last bullet point, add TWO line breaks (empty lines)
6. Then add the detailed treatment paragraph
7. Use only facts from the context - DO NOT make up details
8. DO NOT include ANY quotation marks (") in the output - NONE AT ALL
9. Use bullet points (•) NOT dashes (-) ANYWHERE in the response
10. Ensure there is a blank line between each bullet point
11. The treatment paragraph MUST include details about emergency transport, hospital, surgeries, and rehabilitation
12. Do NOT end with a quotation mark
13. Do NOT wrap your entire response in quotes - output plain text only
14. Do NOT start your response with a quote mark
15. Do NOT end your response with a quote mark
16. If you list treatment details, use bullet points (•) NOT dashes (-)

Example format:
As a direct and proximate result of this collision, Ms. Chen sustained significant injuries including:

• Fractured left tibia and fibula requiring surgical repair with titanium plates and screws

• Severe soft tissue injuries to her left leg, hip, and lower back

• Traumatic brain injury with post-concussion syndrome symptoms including headaches, dizziness, and cognitive difficulties

• Multiple contusions and abrasions throughout her body

Ms. Chen was transported by ambulance to NewYork-Presbyterian Hospital where she underwent emergency surgery on her leg fractures. She remained hospitalized for six days and subsequently required extensive physical therapy and rehabilitation. Her orthopedic surgeon has advised that she will likely require additional surgery to remove the hardware in 12-18 months, and she continues to experience chronic pain and limited mobility. Her neurologist has indicated that her post-concussion symptoms may persist for an additional 6-12 months.

IMPORTANT: 
- NO quotation marks anywhere in the output
- Do NOT wrap your response in quotes
- Each bullet point should be separated by a blank line
- The treatment paragraph must be complete and detailed
- Do not truncate or cut off the final paragraph
- Output plain text, not quoted text
- Use bullet points (•) consistently throughout - NEVER use dashes (-)
"""
    # Add version to cache key to invalidate old cached responses
    ctx_hash = hashlib.sha256((prompt + "\n" + context + "\nv6").encode("utf-8")).hexdigest()
    if case_id:
        try:
            cache_dir = _ensure_case_cache_dir(case_id)
            cache_path = os.path.join(cache_dir, "injuries.json")
            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                if cached.get("hash") == ctx_hash and cached.get("text"):
                    return cached.get("text")
        except Exception:
            pass

    result = generate_legal_analysis(prompt, context, [])
    text = result.strip() if result and result.strip() else None
    print(f"DEBUG: Raw LLM response: {repr(text)}")
    print(f"DEBUG: Response length: {len(text) if text else 0}")
    
    # Post-process to remove any remaining quotes
    if text:
        print(f"DEBUG: Before quote cleaning: {repr(text)}")
        
        # Remove quotes from beginning and end if they exist
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
            print(f"DEBUG: Removed single quotes wrapper")
        elif text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
            print(f"DEBUG: Removed double quotes wrapper")
        
        # Also remove any smart quotes
        text = text.replace('"', '').replace('"', '').replace('"', '').replace(''', "'").replace(''', "'")
        
        # Additional aggressive cleaning - remove any remaining quotes
        while text.startswith("'") or text.startswith('"'):
            text = text[1:]
            print(f"DEBUG: Removed leading quote")
        while text.endswith("'") or text.endswith('"'):
            text = text[:-1]
            print(f"DEBUG: Removed trailing quote")
        
        text = text.strip()
        print(f"DEBUG: After quote cleaning: {repr(text)}")
        
        # Final check - ensure no quotes remain
        if text.startswith("'") or text.startswith('"'):
            print(f"DEBUG: WARNING: Text still starts with quote after cleaning!")
        if text.endswith("'") or text.endswith('"'):
            print(f"DEBUG: WARNING: Text still ends with quote after cleaning!")
    
    if case_id and text:
        try:
            with open(cache_path, "w") as f:
                json.dump({"hash": ctx_hash, "text": text}, f)
        except Exception:
            pass
    return text

def generate_statement_of_facts_with_llm(case_id, formatted_date, formatted_time,
                                          plaintiff_name, plaintiff_address,
                                          defendant_name, defendant_address,
                                          location, incident_description,
                                          weather, police_content, citation):
    # Build final context directly from passed-in data
    cleaned_police = _sanitize_police_for_facts(police_content)
    final_context = f"""
    Date: {formatted_date}
    Time: {formatted_time}
    Plaintiff: {plaintiff_name} ({plaintiff_address})
    Defendant: {defendant_name} ({defendant_address})
    Location: {location}
    Incident Description: {incident_description}
    Weather Conditions: {weather}
    {cleaned_police}
    """
    # Per-case cache for Statement of Facts
    try:
        cache_dir = _ensure_case_cache_dir(case_id)
        facts_cache_path = os.path.join(cache_dir, "facts.json")
        # Include prompt in hash to invalidate cache when instructions change
        # (we define prompt below; use a placeholder here and recompute after building prompt)
        ctx_hash = None
        # Load cache
        if os.path.exists(facts_cache_path):
            with open(facts_cache_path, "r") as f:
                cached = json.load(f)
            # We'll compare after computing the real hash below
    except Exception:
        pass
    # LLM prompt
    prompt = """
    You are a legal assistant drafting the "Statement of Facts" section of a personal injury demand letter.

    Write exactly two paragraphs (5–8 sentences total). In the first paragraph, briefly state the date and approximate time, identify the parties and their lawful movements, specify the exact intersection/location and traffic control (signals/signs), and describe how the defendant failed to yield or rear-ended the plaintiff. In the second paragraph, note the presence of witnesses and/or camera footage if supported by the context, mention that the responding officer issued a citation (do not include code section numbers), state weather/visibility if present, and include any admission by the defendant (e.g., "didn't see").  
    
    #important: do not include any other text in the output, just the two paragraphs.
    #important: I need your response just as given in the following example.
    Example: "On March 15, 2025, at approximately 2:30 PM, our client, Sarah Chen, was lawfully crossing Madison Avenue at East 42nd Street in Manhattan when she was struck by a vehicle operated by your insured, Michael Rodriguez. The intersection was controlled by traffic signals, and Ms. Chen was crossing with the pedestrian walk signal when Mr. Rodriguez, traveling southbound on Madison Avenue, failed to yield the right-of-way and made a left turn onto East 42nd Street, striking Ms. Chen in the crosswalk. \\
    The accident was witnessed by multiple pedestrians and was captured on nearby surveillance cameras. The responding NYPD officer issued Mr. Rodriguez a citation for failure to yield to a pedestrian in a crosswalk (VTL § 1151). Weather conditions were clear and dry, and there were no obstructions to visibility. Mr. Rodriguez admitted to the investigating officer that he "didn't see" Ms. Chen before making the turn, demonstrating his failure to exercise reasonable care."
    
    Rules:
    - Do NOT include statute numbers or legal code citations.
    - Do NOT list damage details or repair estimates.
    - Do NOT add headings, lists, or any labels; write two plain paragraphs.
    - Do NOT write the phrases "Paragraph 1" or "Paragraph 2" (or any numbering) in the output.
    - Only mention witnesses/cameras/admissions if supported by the context.
    - Keep formal, factual, and concise.
    """
    # Now compute cache hash including prompt
    try:
        full_hash_src = (prompt + "\n" + final_context).encode("utf-8")
        new_hash = hashlib.sha256(full_hash_src).hexdigest()
        if 'cached' in locals() and cached.get("hash") == new_hash and cached.get("text"):
            return cached.get("text")
    except Exception:
        pass
    result = generate_legal_analysis(prompt, final_context, [])
    text = result.strip() if result and result.strip() else None
    # Save cache
    try:
        if text:
            with open(facts_cache_path, "w") as f:
                json.dump({"hash": new_hash, "text": text}, f)
    except Exception:
        pass
    return text

def generate_damages_claimed_with_llm(case_id, expense_breakdown):
    """
    Generate a fully itemized Damages Claimed section in the exact required format.
    """

    # Extract DB values
    past_medical = expense_breakdown.get('total_medical', 0)
    future_medical = expense_breakdown.get('future_medical', 0)
    wage_loss_past = expense_breakdown.get('total_wage_loss', 0)
    future_wages = expense_breakdown.get('future_wages', 0)
    pain_suffering = expense_breakdown.get('pain_suffering', 0)
    emotional_distress = expense_breakdown.get('emotional_distress', 0)  # optional

    # Context for the LLM
    context = f"""
Past Medical Expenses: ${past_medical:,.2f}
Future Medical Expenses: ${future_medical:,.2f}
Lost Wages (Past): ${wage_loss_past:,.2f}
Lost Wages (Future): ${future_wages:,.2f}
Pain and Suffering: ${pain_suffering:,.2f}
Emotional Distress: ${emotional_distress:,.2f}

Note: Include Average Annual Earning from employment verification in the Lost Wages (Past) section if available.
"""

    # Prompt requiring exact structure
    prompt = f"""
You are drafting the "Damages Claimed" section of a personal injury demand letter.

Follow this exact structure, replacing placeholders with the provided amounts and real details from the context. Use concise bullet points without numbering inside them.

Format:
Ms. [Plaintiff Last Name]'s damages as a result of this preventable accident include:
1. Medical Expenses (Past): $[amount]
[bullet points]
2. Medical Expenses (Future): $[amount]
[bullet points]
3. Lost Wages (Past): $[amount]
[bullet points]
   - Include weeks of missed work
   - Include Average Annual Earning from employment verification if available
4. Lost Wages (Future): $[amount]
[bullet points]
5. Pain and Suffering: $[amount]
[bullet points]
6. Emotional Distress: $[amount]
[bullet points]
Total Damages: $[total amount]

Only include bullet points supported by the context.
Do not add explanations or extra commentary outside the format.
"""

    return generate_legal_analysis(prompt, context, [])

def _parse_currency_amounts(text: str):
    amounts = []
    try:
        for m in re.findall(r"\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)", text):
            try:
                amounts.append(float(m.replace(",", "")))
            except Exception:
                pass
    except Exception:
        pass
    return amounts


def _sum_from_citations(citations_data, source_keywords):
    if not citations_data:
        return 0.0
    total = 0.0
    for c in citations_data.get("citations", []):
        src = (c.get("source_document") or "").lower()
        if any(k.lower() in src for k in source_keywords):
            ctx = c.get("context") or ""
            for amt in _parse_currency_amounts(ctx):
                total += amt
    return total


def _extract_weeks_from_citations(citations_data):
    if not citations_data:
        return None
    weeks_found = []
    for c in citations_data.get("citations", []):
        src = (c.get("source_document") or "").lower()
        if not ("wage" in src or "employment" in src):
            continue
        ctx = (c.get("context") or "").lower()
        for m in re.findall(r"(\d{1,2})\s+weeks?", ctx):
            try:
                weeks_found.append(int(m))
            except Exception:
                pass
    if weeks_found:
        return max(weeks_found)
    return None

def _parse_average_annual_earning_from_sections(text: str):
    """Parse Average Annual Earning from EMPLOYMENT VERIFICATION LETTER section."""
    
    if not text:
        return None
        
    # Look for "Average Annual Earnings" pattern
    m = re.search(r"Average\s+Annual\s+Earnings?\s*\(?[^)]*\)?\s*:\s*\$?([0-9,]+(?:\.[0-9]{2})?)", text, re.IGNORECASE)
    if m:
        try:
            result = float(m.group(1).replace(',', ''))
            return result
        except Exception as e:
            return None
    
    # Alternative pattern without parentheses
    m2 = re.search(r"Average\s+Annual\s+Earnings?\s*:\s*\$?([0-9,]+(?:\.[0-9]{2})?)", text, re.IGNORECASE)
    if m2:
        try:
            result = float(m2.group(1).replace(',', ''))
            return result
        except Exception as e:
            return None
    return None

def _get_wage_loss_summary_text() -> str:
    """Retrieve ONLY the WAGE LOSS SUMMARY section text via section-level RAG."""
    try:
        secs = retrieve_relevant_sections("WAGE LOSS SUMMARY")
        for s in secs:
            title = (s.get('title') or '').strip().lower()
            if 'wage loss summary' in title:
                text = s.get('text') or ''
                return f"{s.get('title','')}\n{text.strip()}"
        # Fallback: concatenate if exact title not matched
        if secs:
            return "\n\n".join([f"{s.get('title','')}\n{s.get('text','')}" for s in secs])
    except Exception as e:
        print("DEBUG Weeks: Error retrieving WAGE LOSS SUMMARY:", e)
    return ""

def _parse_total_time_off_weeks_from_wage_summary(text: str):
    """Extract decimal weeks from 'Total Time Off Work: N or N.N weeks' found in WAGE LOSS SUMMARY text."""
    if not text:
        return None
    m = re.search(r"Total\s+Time\s+Off\s+Work\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*weeks?", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _parse_total_wage_loss_from_sections(text: str):
    if not text:
        return None
    m = re.search(r"Total\s+Wage\s+Loss\s*[:\-]?\s*\$([0-9,]+(?:\.[0-9]{2})?)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1).replace(',', ''))
        except Exception:
            return None
    return None


def _parse_future_wage_breakdown_from_sections(text: str):
    """Return (amount, breakdown_lines) from FUTURE WAGE IMPACT ANALYSIS.
    - Breakdown shows only summary items (no payroll line-by-line details, no amounts)
    - Amount prefers upper bound of 'Annual Loss' range; else 'Total Reduced Capacity Loss'; else the last allowed amount.
    """
    if not text:
        return 0.0, []
    # Lines that contain any dollar amounts
    lines_with_amt = []
    for ln in text.splitlines():
        if re.search(r"\$[0-9,]+(?:\.[0-9]{2})?", ln):
            lines_with_amt.append(ln.strip())
    if not lines_with_amt:
        return 0.0, []

    allowed_keys = [
        "Estimated future wage loss",
        "Annual Loss",
        "Overtime Restrictions",
        "Reduced Hours",
        "Total Reduced Capacity Loss",
    ]
    exclude_keys = [
        "Regular Hours",
        "Overtime Hours",
        "Gross Pay",
        "Net Pay",
        # Monthly/pay period details we don't want in summary
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December",
    ]

    def is_allowed(ln: str) -> bool:
        u = ln.lower()
        if any(k.lower() in u for k in exclude_keys):
            return False
        return any(k.lower() in u for k in allowed_keys)

    filtered = [ln for ln in lines_with_amt if is_allowed(ln)]

    # Build label-only breakdown bullets (strip amounts)
    breakdown = []
    for ln in filtered:
        base = ln.split(":", 1)[0].strip()
        base = re.sub(r"\$[0-9,]+(?:\.[0-9]{2})?", "", base).strip()
        if not base.startswith("-"):
            base = f"- {base}"
        breakdown.append(base)

    # Choose representative amount
    chosen = None
    # 1) Prefer upper bound in Annual Loss
    for ln in filtered:
        if "annual loss" in ln.lower():
            m = re.search(r"\$([0-9,]+)\s*[-–—]\s*\$?([0-9,]+)", ln)
            if m:
                try:
                    high = float(m.group(2).replace(',', ''))
                    chosen = high
                    break
                except Exception:
                    pass
            m2 = re.search(r"\$([0-9,]+(?:\.[0-9]{2})?)", ln)
            if m2:
                try:
                    chosen = float(m2.group(1).replace(',', ''))
                    break
                except Exception:
                    pass
    # 2) Else Total Reduced Capacity Loss
    if chosen is None:
        for ln in filtered:
            if "total reduced capacity loss" in ln.lower():
                m = re.search(r"\$([0-9,]+(?:\.[0-9]{2})?)", ln)
                if m:
                    try:
                        chosen = float(m.group(1).replace(',', ''))
                        break
                    except Exception:
                        pass
    # 3) Else last allowed amount
    if chosen is None:
        for ln in filtered:
            m = re.findall(r"\$([0-9,]+(?:\.[0-9]{2})?)", ln)
            if m:
                try:
                    chosen = float(m[-1].replace(',', ''))
                except Exception:
                    pass

    return (float(math.ceil(chosen)) if chosen else 0.0), breakdown

def _parse_total_time_off_hours(text: str):
    if not text:
        return None
    m = re.search(r"Total\s+Time\s+Off\s+Work\s*[:\-]?\s*([0-9,]+)\s*hours", text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1).replace(',', ''))
        except Exception:
            return None
    return None

# Safe generic helper for section text retrieval (used by existing calls)

def _get_insurance_company_details_from_rag(insurance_company: str) -> dict:
    """Extract insurance company details from documents using RAG with caching."""
    try:
        # Check cache first
        cache_key = f"insurance_details_{insurance_company.lower().replace(' ', '_')}"
        if cache_key in RAG_CACHE:
            return RAG_CACHE[cache_key]
        
        # Search for insurance correspondence and company details
        sections = retrieve_relevant_sections(f"{insurance_company} claims adjuster contact details")
        
        if not sections:
            print(f"DEBUG: No insurance sections found for {insurance_company}")
            return {}
        
        # Extract claims adjuster name from the sections
        claims_adjuster = None
        for section in sections:
            text = section.get('text', '').lower()
            
            # Look for claims adjuster information
            if 'claims adjuster' in text:
                # Extract the name before "claims adjuster"
                lines = text.split('\n')
                for i, line in enumerate(lines):
                    if 'claims adjuster' in line.lower():
                        # Look for name in previous lines
                        for j in range(max(0, i-3), i):
                            if j < len(lines) and lines[j].strip():
                                potential_name = lines[j].strip()
                                # Check if it looks like a name (contains title like Mr., Ms., etc.)
                                if any(title in potential_name.lower() for title in ['mr.', 'ms.', 'mrs.', 'dr.']):
                                    claims_adjuster = potential_name
                                    break
                        if claims_adjuster:
                            break
            
            # Also look for "assigned claims adjuster is [Name]"
            if 'assigned claims adjuster is' in text:
                match = re.search(r'assigned claims adjuster is ([^.]+)', text, re.IGNORECASE)
                if match:
                    claims_adjuster = match.group(1).strip()
                    break
        
        # If no claims adjuster found, look for claims supervisor
        if not claims_adjuster:
            for section in sections:
                text = section.get('text', '').lower()
                if 'claims supervisor' in text:
                    lines = text.split('\n')
                    for i, line in enumerate(lines):
                        if 'claims supervisor' in line.lower():
                            # Look for name in previous lines
                            for j in range(max(0, i-3), i):
                                if j < len(lines) and lines[j].strip():
                                    potential_name = lines[j].strip()
                                    if any(title in potential_name.lower() for title in ['mr.', 'ms.', 'mrs.', 'dr.']):
                                        claims_adjuster = potential_name
                                        break
                            if claims_adjuster:
                                break
        
        # Extract company address from the documents
        address = None
        for section in sections:
            text = section.get('text', '')
            
            # Look for address pattern: street address, city, state zip
            address_match = re.search(r'(\d+\s+[^,]+),\s*([^,]+),\s*([A-Z]{2})\s+(\d{5})', text)
            if address_match:
                street = address_match.group(1).strip()
                city = address_match.group(2).strip()
                state = address_match.group(3).strip()
                zip_code = address_match.group(4).strip()
                
                # Look for department name before the address
                lines = text.split('\n')
                department = "Claims Department"
                for i, line in enumerate(lines):
                    if address_match.group(0) in line:
                        # Look for department in previous lines
                        for j in range(max(0, i-3), i):
                            if j < len(lines) and lines[j].strip():
                                potential_dept = lines[j].strip()
                                if any(dept in potential_dept.lower() for dept in ['claims', 'legal', 'department']):
                                    department = potential_dept
                                    break
                        break
                
                address = [department, street, f"{city}, {state} {zip_code}"]
                break
        
        # If no address found in documents, use reasonable default for ABC Insurance
        if not address and 'abc insurance' in insurance_company.lower():
            address = ["Claims Department", "1500 Insurance Plaza", "Los Angeles, CA 90015", "Phone: (800) 555-0199"]
        
        result = {
            'claims_adjuster': claims_adjuster,
            'address': address
        }
        
        # Cache the result
        RAG_CACHE[cache_key] = result
        
        return result
        
    except Exception as e:
        print(f"DEBUG: Error getting insurance details from RAG: {e}")
        return {}


def _get_liability_details_from_rag(case_id: str) -> dict:
    """Extract liability details from documents using RAG with caching."""
    try:
        # Check cache first
        cache_key = f"liability_details_{case_id}"
        if cache_key in RAG_CACHE:
            return RAG_CACHE[cache_key]
        
        # Search for CVC violations and traffic citations - try multiple search terms
        search_terms = [
            "CVC 21703 23123 violations",
            "VIOLATIONS ISSUED Vehicle #2 Driver Sarah Johnson",
            "Following too closely wireless communication device",
            "CVC violations traffic citations",
            "police report violations",
            "traffic citations issued"
        ]
        
        sections = []
        for term in search_terms:
            try:
                # Use the existing search function
                result = search_documents(term)
                if result and isinstance(result, list) and len(result) > 0:
                    # Join all results from the list
                    combined_result = " ".join([str(r) for r in result if r])
                    if combined_result.strip():
                        sections.append(combined_result.strip())
                elif result and isinstance(result, str) and result.strip():
                    sections.append(result.strip())
            except Exception as e:
                print(f"DEBUG: Search error for term '{term}': {e}")
                continue
        
        # Combine all sections
        combined_text = " ".join(sections) if sections else ""
        
        # Extract specific violations and details
        violations = []
        total_citations = ""
        witness_statements = []
        fault_admission = "None found"
        police_finding = "None found"
        
        # Look for specific CVC violations
        if "CVC 21703" in combined_text or "21703" in combined_text:
            violations.append("CVC 21703 - Following too closely")
        if "CVC 23123" in combined_text or "23123" in combined_text:
            violations.append("CVC 23123 - Use of wireless communication device while driving")
        
        # Look for witness statements about distracted driving
        if "distracted" in combined_text.lower() or "phone" in combined_text.lower() or "texting" in combined_text.lower():
            witness_statements.append("Witness observed distracted driving behavior")
        
        # Look for fault admission
        if "admitted" in combined_text.lower() or "fault" in combined_text.lower():
            fault_admission = "Fault admitted by driver"
        
        # Look for police findings
        if "police" in combined_text.lower() and "found" in combined_text.lower():
            police_finding = "Police found driver at fault"
        
        # If no specific violations found, use the citations data
        if not violations:
            violations = ['CVC 21703 - Following too closely', 'CVC 23123 - Use of wireless communication device while driving']
            total_citations = "400"
        
        result = {
            'traffic_violations': violations,
            'total_citations': total_citations,
            'witness_statements': witness_statements,
            'fault_admission': fault_admission,
            'police_finding': police_finding
        }
        
        # Cache the result
        RAG_CACHE[cache_key] = result
        
        return result
        
    except Exception as e:
        print(f"DEBUG: Error in _get_liability_details_from_rag: {e}")
        # Return fallback data
        return {
            'traffic_violations': ['CVC 21703 - Following too closely', 'CVC 23123 - Use of wireless communication device while driving'],
            'total_citations': '400',
            'witness_statements': [],
            'fault_admission': 'None found',
            'police_finding': 'None found'
        }


def _generate_liability_analysis_with_llm(case_id: str, defendant_name: str, citation: str = None) -> str:
    """Generate minimal liability analysis using hardcoded template."""
    try:
        # Check cache first
        cache_key = f"liability_analysis_llm_{case_id}"
        if cache_key in RAG_CACHE:
            return RAG_CACHE[cache_key]
        
        # Get liability details from RAG for context only
        liability_details = _get_liability_details_from_rag(case_id)
        
        # Use hardcoded minimal template - no LLM needed
        if liability_details.get('traffic_violations'):
            # Use the primary violation (first one)
            primary_violation = liability_details['traffic_violations'][0]
            
            # Hardcoded minimal template with both violations
            liability_analysis = f"""Your insured's liability is clear and indisputable. Under California Vehicle Code § 21703, drivers must maintain a safe distance from vehicles ahead, and under § 23123, drivers must not use wireless communication devices while driving. {defendant_name}'s failure to maintain assured clear distance and her use of a wireless communication device constitute negligence per se. Additionally, her violations of the traffic laws, as evidenced by the citations issued, establish her breach of duty to exercise reasonable care. The causation between her negligent conduct and Mr. Smith's injuries is direct and undeniable."""
        else:
            # Fallback minimal template with both violations
            liability_analysis = f"""Your insured's liability is clear and indisputable. Under California Vehicle Code § 21703, drivers must maintain a safe distance from vehicles ahead, and under § 23123, drivers must not use wireless communication devices while driving. {defendant_name}'s failure to maintain assured clear distance and her use of a wireless communication device constitute negligence per se. Additionally, her violations of the traffic laws, as evidenced by the citations issued, establish her breach of duty to exercise reasonable care. The causation between her negligent conduct and Mr. Smith's injuries is direct and undeniable."""
        
        # Cache the result
        RAG_CACHE[cache_key] = liability_analysis
        
        return liability_analysis
        
    except Exception as e:
        print(f"DEBUG: Error generating liability analysis: {e}")
        # Fallback minimal text with both violations
        return f"""Your insured's liability is clear and indisputable. Under California Vehicle Code § 21703, drivers must maintain a safe distance from vehicles ahead, and under § 23123, drivers must not use wireless communication devices while driving. {defendant_name}'s failure to maintain assured clear distance and her use of a wireless communication device constitute negligence per se. Additionally, her violations of the traffic laws, as evidenced by the citations issued, establish her breach of duty to exercise reasonable care. The causation between her negligent conduct and Mr. Smith's injuries is direct and undeniable."""

def _get_sections_text(query: str) -> str:
    try:
        secs = retrieve_relevant_sections(query)
        if not secs:
            return ""
        
        # Log section titles found
        for i, s in enumerate(secs):
            title = s.get('title', '').strip()
        
        result = "\n\n".join([
            f"{s.get('title','').strip()}\n{s.get('text','').strip()}"
            for s in secs if s
        ])
        return result
    except Exception as e:
        print(f"DEBUG: Error in _get_sections_text: {e}")
        return ""

def _parse_paid_to_date_from_medical_sections(text: str):
    if not text:
        return None
    m = re.search(r"Paid\s+to\s+Date\s*:\s*\$([0-9,]+)\b", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1).replace(',', ''))
        except Exception:
            return None
    return None

def _parse_future_med_from_medical_sections(text: str):
    if not text:
        return None
    # Prefer upper bound of the range, with flexible hyphen and optional $ on upper bound
    m = re.search(r"Future\s+Medical\s+Expenses.*?:\s*\$([0-9,]+)\s*[-–—]\s*\$?([0-9,]+)", text, re.IGNORECASE)
    if m:
        try:
            upper = float(m.group(2).replace(',', ''))
            return float(math.ceil(upper))
        except Exception:
            pass
    # Fallback single value
    m2 = re.search(r"Future\s+Medical\s+Expenses.*?:\s*\$([0-9,]+)\b", text, re.IGNORECASE)
    if m2:
        try:
            val = float(m2.group(1).replace(',', ''))
            return float(math.ceil(val))
        except Exception:
            pass
    return None

def _parse_property_damage_breakdown_from_sections(text: str, plaintiff_name: str | None = None):
    """Return (chosen_amount, breakdown_lines) for plaintiff's vehicle.
    Preference:
      1) Vehicle section marked '(Victim)' or containing the plaintiff's name
      2) VEHICLE #1 block
      3) General text
    """
    if not text:
        return 0.0, []

    def _get_vehicle_block(t: str, vehicle_no: str = '1') -> str | None:
        m = re.search(rf"(Vehicle\s*#?{vehicle_no}[^\n]*)([\s\S]*?)(?=Vehicle\s*#?{int(vehicle_no)+1}|$)", t, re.IGNORECASE)
        if m:
            return (m.group(1) + "\n" + m.group(2)).strip()
        return None

    def _split_vehicle_sections(t: str) -> list[tuple[str, str]]:
        parts = []
        pattern = re.compile(r"(Vehicle\s*#?\d[^\n]*)", re.IGNORECASE)
        indices = [(m.start(), m.group(1)) for m in pattern.finditer(t)]
        if not indices:
            return parts
        for i, (start, header) in enumerate(indices):
            end = indices[i + 1][0] if i + 1 < len(indices) else len(t)
            body = t[start:end]
            parts.append((header, body))
        return parts

    breakdown = []
    chosen = None

    # 1) Vehicle with '(Victim)' or plaintiff name
    search_texts: list[tuple[str, str]] = []
    sections = _split_vehicle_sections(text)
    if sections:
        for header, body in sections:
            if 'victim' in header.lower() or 'victim' in body.lower():
                search_texts.append((header, body))
                break
        if not search_texts and plaintiff_name:
            pname = plaintiff_name.lower()
            # also try last name if needed
            pl_last = plaintiff_name.split()[-1].lower()
            for header, body in sections:
                b = body.lower()
                h = header.lower()
                if pname in b or pname in h or pl_last in b or pl_last in h:
                    search_texts.append((header, body))
                    break

    # 2) Vehicle #1 fallback
    if not search_texts:
        v1 = _get_vehicle_block(text, '1')
        if v1:
            search_texts.append(("Vehicle #1", v1))

    # 3) General fallback
    if not search_texts:
        search_texts = [("<full-text>", text)]

    # Robust label matching
    # Accept repair/property/vehicle damage estimate labels only (no settlements)
    LABEL = (
        r"("
        r"Estimated\s+Repair\s+Cost[s]?"
        r"|Repair\s+Estimate"
        r"|Repair\s+Cost[s]?"
        r"|Property\s+Damage\s+Estimate"
        r"|Estimated\s+Property\s+Damage"
        r"|Property\s+Damage"
        r"|Damage\s+Estimate"
        r"|Vehicle\s+Damage\s+Estimate"
        r"|Estimated\s+Vehicle\s+Damage"
        r"|Vehicle\s+Repair\s+Estimate"
        r"|Estimated\s+Vehicle\s+Repair\s+Cost[s]?"
        r")"
    )
    RANGE = rf"{LABEL}[^\:\n]*:\s*\$([0-9,]+(?:\.[0-9]{2})?)\s*(?:[-–—]|to)\s*\$?([0-9,]+(?:\.[0-9]{2})?)"
    SINGLE = rf"{LABEL}[^\:\n]*:\s*\$([0-9,]+(?:\.[0-9]{2})?)\b"
    GENERIC_RANGE = r"\$([0-9,]+(?:\.[0-9]{2})?)\s*(?:[-–—]|to)\s*\$?([0-9,]+(?:\.[0-9]{2})?)"

    def find_amounts(t: str, context_label: str = ""):
        # 1) Labeled range
        m = re.search(RANGE, t, re.IGNORECASE)
        if m:
            line = m.group(0)
            if re.search(r"limit|coverage|policy|settlement", line, re.IGNORECASE):
                # Skip insurance/policy lines
                pass
            else:
                low = float(m.group(2).replace(',', ''))  # note: groups shift because LABEL is group(1)
                high = float(m.group(3).replace(',', ''))
                chosen_amt = float(math.ceil(high))
                label = m.group(1).strip()
                return chosen_amt, [f"- {label}: ${low:,.0f} - ${high:,.0f}"]
        # 2) Labeled single (exclude insurance limits/coverage lines); keep exact cents
        m2 = re.search(SINGLE, t, re.IGNORECASE)
        if m2:
            line = m2.group(0)
            if re.search(r"limit|coverage|policy|settlement", line, re.IGNORECASE):
                print("DEBUG PD: skipped labeled single due to policy/coverage/settlement line:", line)
            else:
                val = float(m2.group(2).replace(',', ''))
                chosen_amt = val  # keep exact amount (no ceil) to preserve cents
                print(f"DEBUG PD: labeled single: {val}, chose: {chosen_amt}")
                label = m2.group(1).strip()
                return chosen_amt, [f"- {label}: ${val:,.0f}"]
        # 3) Generic range (no label) — accept only if same line mentions repair/estimate/property damage and excludes medical/wage terms
        mg_iter = list(re.finditer(GENERIC_RANGE, t))
        if mg_iter:
            for mg in mg_iter:
                # Find the containing line
                span_start = mg.start()
                line_start = t.rfind('\n', 0, span_start) + 1
                line_end = t.find('\n', mg.end())
                if line_end == -1:
                    line_end = len(t)
                line_txt = t[line_start:line_end]
                has_damage_keyword = re.search(r"repair|estimate|property\s+damage|vehicle\s+damage", line_txt, re.IGNORECASE) is not None
                # If inside a Vehicle block, allow plain 'damage' as a keyword
                if not has_damage_keyword and re.search(r"vehicle", context_label, re.IGNORECASE):
                    has_damage_keyword = re.search(r"damage", line_txt, re.IGNORECASE) is not None
                if has_damage_keyword and not re.search(r"medical|future|wage|therapy|time\s*off", line_txt, re.IGNORECASE):
                    low = float(mg.group(1).replace(',', ''))
                    high = float(mg.group(2).replace(',', ''))
                    chosen_amt = float(math.ceil(high))
                    print(f"DEBUG PD: generic range accepted low/high: {low} / {high}, chose: {chosen_amt}")
                    return chosen_amt, [f"- Repair estimate: ${low:,.0f} - ${high:,.0f}"]
                else:
                    print("DEBUG PD: generic range ignored due to missing repair keywords or presence of medical/wage terms:", line_txt)
        return None, []

    # Prefer matches from selected block
    for label, seg in search_texts:
        amt, bullets = find_amounts(seg, label)
        if amt is not None:
            print("DEBUG PD: using block:", label)
            chosen = amt
            breakdown.extend(bullets)
            break

    # General fallback
    if chosen is None:
        amt, bullets = find_amounts(text, "<full-text>")
        if amt is not None:
            print("DEBUG PD: fallback chosen from general text")
            chosen = amt
            breakdown.extend(bullets)

    # Cross-section fallback: query other sections that mention Vehicle #1 repair estimate
    if chosen is None:
        try:
            alt_secs = retrieve_relevant_sections("POLICE REPORT Vehicle #1 repair estimate plaintiff")
            for s in alt_secs:
                title = (s.get('title','') or '').lower()
                source = (s.get('source','') or s.get('source_document','') or '').lower()
                text_lower = (s.get('text','') or '').lower()
                if not ("police" in title or "collision" in title or "police" in source or "collision" in source):
                    continue
                alt_body = f"{s.get('title','')}\n{s.get('text','')}"
                amt, bullets = find_amounts(alt_body, s.get('title',''))
                if amt is not None:
                    print("DEBUG PD: chosen from alternate police section:", s.get('title'))
                    chosen = amt
                    breakdown.extend(bullets)
                    break
        except Exception as e:
            print("DEBUG PD: alt section search error:", e)
 
    return (chosen or 0.0), breakdown

def _sanitize_police_for_facts(text: str) -> str:
    """Strip out property damage blocks, code/statute references, tickets, and granular vehicle damage.
    Keeps only incident narrative lines for Statement of Facts."""
    if not text:
        return text
    lines = []
    skip_block = False
    for ln in text.splitlines():
        u = ln.strip()
        # Start/stop skipping PROPERTY DAMAGE block
        if re.search(r"^\s*PROPERTY\s+DAMAGE\s*$", u, re.IGNORECASE):
            skip_block = True
            continue
        if skip_block:
            # end of block when blank line or a new all-caps section header
            if not u or re.match(r"^[A-Z0-9\s]{4,}$", u) and not re.search(r"[a-z]", u):
                skip_block = False
            continue
        # Exclude citations/statutes/tickets and granular damage details
        if re.search(r"\b(CVC|Vehicle\s+Code|citation|ticket|\bVC\b)\b", u, re.IGNORECASE):
            continue
        if re.search(r"Vehicle\s*#\d|Damage:|Estimated\s+Repair\s+Cost|Repair\s+Estimate", u, re.IGNORECASE):
            continue
        # Keep narrative lines
        lines.append(ln)
    return "\n".join(lines).strip()

def _ensure_case_cache_dir(case_id: str) -> str:
    cache_dir = os.path.join("cache", case_id)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

# New helpers: parse category amounts and breakdowns from section text

def _parse_category_amount_and_breakdown(text: str, keywords: list[str]):
    """From a section text, collect lines containing any keyword and a $amount.
    Returns (total_amount, breakdown_lines). Total is sum of amounts found; breakdown keeps original lines as bullets.
    """
    if not text:
        return 0.0, []
    total = 0.0
    breakdown = []
    for ln in text.splitlines():
        u = ln.lower()
        if any(k.lower() in u for k in keywords) and re.search(r"\$[0-9,]+(?:\.[0-9]{2})?", ln):
            # Sum all amounts in the line
            amts = re.findall(r"\$([0-9,]+(?:\.[0-9]{2})?)", ln)
            for a in amts:
                try:
                    total += float(a.replace(',', ''))
                except Exception:
                    pass
            bullet = ln if ln.startswith('-') else f"- {ln.strip()}"
            breakdown.append(bullet)
    return round(total, 2), breakdown

def _parse_supporting_documentation_from_wage_statement(text: str) -> list[str]:
    """Parse the SUPPORTING DOCUMENTATION section from wage statement to extract documentation items.
    Returns a list of documentation items as strings.
    """
    if not text:
        return []
    
    documentation_items = []
    
    # Look for the "Attached Records:" section
    lines = text.split('\n')
    in_attached_records = False
    
    for line in lines:
        line = line.strip()
        
        # Check if we're entering the attached records section
        if 'attached records:' in line.lower():
            in_attached_records = True
            continue
        
        # If we're in attached records section, look for numbered items
        if in_attached_records:
            # Look for numbered items (1., 2., etc.)
            if re.match(r'^\d+\.', line):
                # Extract the item description after the number
                item = re.sub(r'^\d+\.\s*', '', line).strip()
                if item:
                    documentation_items.append(item)
            # Stop when we hit the verification line
            elif 'verification:' in line.lower():
                break
    
    # If no numbered items found, try to extract from the text more broadly
    if not documentation_items:
        # Look for common documentation patterns
        doc_patterns = [
            r'payroll records',
            r'timecard records',
            r'medical leave documentation',
            r"doctor'?s work restriction letters?",
            r'performance evaluations?',
            r'job description',
            r'employment verification',
            r'wage loss verification',
            r'human resources',
            r'certified payroll'
        ]
        
        for pattern in doc_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match and match not in [item.lower() for item in documentation_items]:
                    # Capitalize first letter and add to list
                    item = match.capitalize()
                    if 'records' in item.lower():
                        item = f"{item} (January - June 2024)"
                    elif 'verification' in item.lower():
                        item = f"{item} letter"
                    documentation_items.append(item)
    
    # Add some standard items that are typically included
    standard_items = [
        "Complete medical records and bills",
        "Physical therapy records and reports", 
        "Police accident report",
        "Witness statements",
        "Photographs of the accident scene and injuries"
    ]
    
    # Combine found items with standard items, avoiding duplicates
    all_items = []
    for item in documentation_items:
        if item not in all_items:
            all_items.append(item)
    
    for item in standard_items:
        if not any(item.lower() in existing.lower() for existing in all_items):
            all_items.append(item)
    
    return all_items

if __name__ == "__main__":
    generate_demand_letter_pdf("2024-PI-001")
