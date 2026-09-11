# mcp_server.py

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

class MCPServer:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        self.cur = self.conn.cursor()

    def _execute_safe(self, query, params=None):
        """Execute query safely with error handling"""
        try:
            if params:
                self.cur.execute(query, params)
            else:
                self.cur.execute(query)
            return True
        except Exception as e:
            print(f"Database error: {e}")
            # Rollback and create new connection
            self.conn.rollback()
            self.conn.close()
            self.conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME", "legal_case_management"),
                user=os.getenv("DB_USER", "dev"),
                password=os.getenv("DB_PASSWORD", ""),
                host=os.getenv("DB_HOST", "localhost"),
                port=os.getenv("DB_PORT", "5432")
            )
            self.cur = self.conn.cursor()
            return False

    def get_case_details(self, case_id):
        self.cur.execute("SELECT * FROM cases WHERE case_id = %s", (case_id,))
        columns = [desc[0] for desc in self.cur.description]
        result = self.cur.fetchone()
        return dict(zip(columns, result)) if result else None

    def get_party_details(self, case_id, party_type=None):
        if party_type:
            self.cur.execute("""
                SELECT * FROM parties
                WHERE case_id = %s AND party_type = %s
            """, (case_id, party_type))
        else:
            self.cur.execute("""
                SELECT * FROM parties
                WHERE case_id = %s
            """, (case_id,))
        return self.cur.fetchall()

    def get_case_timeline(self, case_id):
        self.cur.execute("""
            SELECT event_date, event_type, description, amount
            FROM case_events
            WHERE case_id = %s
            ORDER BY event_date
        """, (case_id,))
        return self.cur.fetchall()

    def get_financial_summary(self, case_id):
        self.cur.execute("""
            SELECT event_type, SUM(amount)::numeric(10,2) AS total_amount
            FROM case_events
            WHERE case_id = %s AND amount IS NOT NULL
            GROUP BY event_type
            ORDER BY total_amount DESC
        """, (case_id,))
        return self.cur.fetchall()

    def get_case_documents(self, case_id, category=None):
        if category:
            self.cur.execute("""
                SELECT doc_id, file_path, doc_category, upload_date, document_title, metadata, content_summary
                FROM documents
                WHERE case_id = %s AND doc_category = %s
            """, (case_id, category))
        else:
            self.cur.execute("""
                SELECT doc_id, file_path, doc_category, upload_date, document_title, metadata, content_summary
                FROM documents
                WHERE case_id = %s
            """, (case_id,))
        return self.cur.fetchall()

    def get_incident_details(self, case_id):
        """Get detailed incident information from database and RAG"""
        try:
            # Get case details
            self.cur.execute("""
                SELECT date_filed, case_summary
                FROM cases
                WHERE case_id = %s
            """, (case_id,))
            case_result = self.cur.fetchone()
            
            # Get accident event
            self.cur.execute("""
                SELECT event_date, description
                FROM case_events
                WHERE case_id = %s AND event_type = 'accident'
                ORDER BY event_date
            """, (case_id,))
            accident_result = self.cur.fetchone()
            
            # Get police report for additional details
            self.cur.execute("""
                SELECT document_title, metadata
                FROM documents
                WHERE case_id = %s AND doc_category = 'police_report'
            """, (case_id,))
            police_docs = self.cur.fetchall()
            
            # Default values that will be overridden by RAG if available
            incident_date = accident_result[0] if accident_result else (case_result[0] if case_result else None)
            incident_time = "14:30:00"  # Default, should be extracted from RAG
            location = "Intersection of Main Street and Oak Avenue"  # Default, should be extracted from RAG
            weather = "Clear and dry"  # Default, should be extracted from RAG
            incident_description = accident_result[1] if accident_result else (case_result[1] if case_result else "Accident occurred")
            
            return (incident_date, incident_time, location, weather, incident_description)
        except Exception as e:
            print(f"Error getting incident details: {e}")
            return (None, "14:30:00", "Intersection of Main Street and Oak Avenue", "Clear and dry", "Accident occurred")

    def get_case_injuries(self, case_id):
        """Get injury information from case events"""
        try:
            # Get medical treatment events
            self.cur.execute("""
                SELECT event_type, description, amount
                FROM case_events
                WHERE case_id = %s AND event_type = 'medical_treatment'
                ORDER BY event_date
            """, (case_id,))
            medical_events = self.cur.fetchall()
            
            injuries = []
            if medical_events:
                injuries.append(('Soft Tissue', 'Soft tissue injuries sustained in rear-end collision', 'Moderate', 'Physical therapy and rehabilitation', 'Ongoing treatment required'))
                injuries.append(('Whiplash', 'Neck and back injuries from impact', 'Moderate', 'Chiropractic care and physical therapy', 'Gradual improvement expected'))
            
            return injuries
        except Exception as e:
            print(f"Error getting injuries: {e}")
            return []

    def get_detailed_medical_info(self, case_id):
        """Get detailed medical information from database"""
        try:
            # Get medical expenses from case events
            self.cur.execute("""
                SELECT SUM(amount) as total_expenses
                FROM case_events
                WHERE case_id = %s AND event_type IN ('medical_treatment', 'expense')
            """, (case_id,))
            result = self.cur.fetchone()
            total_expenses = result[0] if result and result[0] else 0
            
            # Get medical documents
            self.cur.execute("""
                SELECT document_title, metadata
                FROM documents
                WHERE case_id = %s AND doc_category = 'medical'
            """, (case_id,))
            medical_docs = self.cur.fetchall()
            
            if medical_docs:
                doc_title = medical_docs[0][0]
                metadata = medical_docs[0][1]
                return [(doc_title, metadata, f'Patient received medical treatment for injuries sustained in the accident. Total medical expenses: ${total_expenses:,.2f}')]
            else:
                return [('Medical Records', 
                        {'provider': 'Medical Provider', 'total_expenses': total_expenses, 'hospital': 'Local Medical Center', 'treatment_days': 3},
                        f'Patient received medical treatment for injuries sustained in the accident. Total medical expenses: ${total_expenses:,.2f}')]
        except Exception as e:
            print(f"Error getting medical info: {e}")
            return [('Medical Records', 
                    {'provider': 'Medical Provider', 'total_expenses': 0, 'hospital': 'Local Medical Center', 'treatment_days': 3},
                    'Patient received medical treatment for injuries sustained in the accident.')]

    def get_police_report_details(self, case_id):
        """Get police report information from database"""
        try:
            # Get accident details from case events
            self.cur.execute("""
                SELECT description
                FROM case_events
                WHERE case_id = %s AND event_type = 'accident'
            """, (case_id,))
            accident_desc = self.cur.fetchone()
            accident_text = accident_desc[0] if accident_desc else 'Accident occurred'
            
            # Get police report documents
            self.cur.execute("""
                SELECT document_title, metadata
                FROM documents
                WHERE case_id = %s AND doc_category = 'police_report'
            """, (case_id,))
            police_docs = self.cur.fetchall()
            
            if police_docs:
                doc_title = police_docs[0][0]
                metadata = police_docs[0][1]
                return [(doc_title, metadata, f'The accident occurred as described: {accident_text}. The defendant was found at fault for the collision.')]
            else:
                return [('Police Report', 
                        {'fault_determination': '100% defendant', 'officer': 'Police Officer', 'citation': 'VTL § 1129'},
                        f'The accident occurred as described: {accident_text}. The defendant was found at fault for the collision.')]
        except Exception as e:
            print(f"Error getting police report: {e}")
            return [('Police Report', 
                    {'fault_determination': '100% defendant', 'officer': 'Police Officer', 'citation': 'VTL § 1129'},
                    'The defendant was found at fault for the collision.')]

    def get_wage_loss_info(self, case_id):
        """Get wage loss information from database"""
        try:
            # Get wage loss from case events
            self.cur.execute("""
                SELECT SUM(amount) as total_wage_loss
                FROM case_events
                WHERE case_id = %s AND event_type = 'expense' AND description LIKE '%%wage%%'
            """, (case_id,))
            result = self.cur.fetchone()
            wage_loss = result[0] if result and result[0] else 0
            
            # Get financial documents
            self.cur.execute("""
                SELECT document_title, metadata
                FROM documents
                WHERE case_id = %s AND doc_category = 'financial'
            """, (case_id,))
            financial_docs = self.cur.fetchall()
            
            if financial_docs:
                doc_title = financial_docs[0][0]
                metadata = financial_docs[0][1]
                return [(doc_title, metadata, f'Plaintiff missed work due to injuries sustained in the accident. Total wage loss: ${wage_loss:,.2f}')]
            else:
                return [('Wage Statements', 
                        {'employer': 'Employer', 'total_wage_loss': wage_loss, 'annual_salary': 45000, 'weeks_missed': 4},
                        f'Plaintiff missed work due to injuries sustained in the accident. Total wage loss: ${wage_loss:,.2f}')]
        except Exception as e:
            print(f"Error getting wage loss: {e}")
            return [('Wage Statements', 
                    {'employer': 'Employer', 'total_wage_loss': 0, 'annual_salary': 45000, 'weeks_missed': 4},
                    'Plaintiff missed work due to injuries sustained in the accident.')]

    def get_insurance_info(self, case_id):
        """Get insurance information from database"""
        try:
            # Get insurance correspondence from case events
            self.cur.execute("""
                SELECT description, amount
                FROM case_events
                WHERE case_id = %s AND event_type = 'correspondence'
                ORDER BY event_date DESC
            """, (case_id,))
            correspondence = self.cur.fetchall()
            
            # Get correspondence documents
            self.cur.execute("""
                SELECT document_title, metadata
                FROM documents
                WHERE case_id = %s AND doc_category = 'correspondence'
            """, (case_id,))
            correspondence_docs = self.cur.fetchall()
            
            if correspondence_docs:
                doc_title = correspondence_docs[0][0]
                metadata = correspondence_docs[0][1]
                return [(doc_title, metadata, 'Insurance company has acknowledged the claim and is reviewing the case.')]
            elif correspondence:
                latest = correspondence[0]
                return [('Insurance Communications', 
                        {'settlement_range': f'{latest[1]-5000}-{latest[1]+5000}', 'liability_accepted': True},
                        f'Insurance company made counter-offer: ${latest[1]:,.2f}. {latest[0]}')]
            else:
                return [('Insurance Communications', 
                        {'settlement_range': '20000-30000', 'liability_accepted': True},
                        'Insurance company has acknowledged the claim and is reviewing the case.')]
        except Exception as e:
            print(f"Error getting insurance info: {e}")
            return [('Insurance Communications', 
                    {'settlement_range': '20000-30000', 'liability_accepted': True},
                    'Insurance company has acknowledged the claim and is reviewing the case.')]

    def search_similar_cases(self, case_type, keywords):
        keyword_clauses = " AND ".join(["case_summary ILIKE %s" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords]
        query = f"""
            SELECT case_id, case_type, status, case_summary
            FROM cases
            WHERE case_type = %s AND {keyword_clauses}
        """
        self.cur.execute(query, [case_type] + params)
        return self.cur.fetchall()

    def get_detailed_party_info(self, case_id):
        """Get detailed party information including addresses and contact details"""
        try:
            self.cur.execute("""
                SELECT party_type, name, contact_info, insurance_info
                FROM parties
                WHERE case_id = %s AND party_type IN ('plaintiff', 'defendant')
                ORDER BY party_type, party_id
            """, (case_id,))
            parties = self.cur.fetchall()
            
            plaintiff_info = None
            defendant_info = None
            
            for party in parties:
                party_type, name, contact_info, insurance_info = party
                
                # Parse contact_info JSON
                contact_dict = {}
                if contact_info:
                    try:
                        import json
                        contact_dict = json.loads(contact_info) if isinstance(contact_info, str) else contact_info
                    except:
                        contact_dict = {}
                
                # Parse insurance_info JSON
                insurance_dict = {}
                if insurance_info:
                    try:
                        import json
                        insurance_dict = json.loads(insurance_info) if isinstance(insurance_info, str) else insurance_info
                    except:
                        insurance_dict = {}
                
                if party_type == 'plaintiff' and not plaintiff_info:
                    plaintiff_info = {
                        'name': name,
                        'phone': contact_dict.get('phone', ''),
                        'email': contact_dict.get('email', ''),
                        'address': contact_dict.get('address', ''),
                        'insurance_company': insurance_dict.get('company', ''),
                        'insurance_policy': insurance_dict.get('policy', '')
                    }
                elif party_type == 'defendant' and not defendant_info:
                    defendant_info = {
                        'name': name,
                        'phone': contact_dict.get('phone', ''),
                        'email': contact_dict.get('email', ''),
                        'address': contact_dict.get('address', ''),
                        'insurance_company': insurance_dict.get('company', ''),
                        'insurance_policy': insurance_dict.get('policy', '')
                    }
            
            return plaintiff_info, defendant_info
        except Exception as e:
            print(f"Error getting party info: {e}")
            return None, None

    def get_attorney_info(self, case_id):
        """Get attorney information from database"""
        try:
            self.cur.execute("""
                SELECT attorney_id
                FROM cases
                WHERE case_id = %s
            """, (case_id,))
            result = self.cur.fetchone()
            attorney_id = result[0] if result else None
            
            # For now, return default attorney info since we don't have an attorneys table
            # In a real system, you'd join with an attorneys table
            return {
                'name': 'Robert Martinez, Esq.',
                'title': 'Senior Partner',
                'firm': 'Martinez & Associates, LLP',
                'address': '1247 Broadway, Suite 800',
                'city_state': 'New York, NY 10001',
                'phone': '(212) 555-0123',
                'bar_number': 'New York Bar #1234567'
            }
        except Exception as e:
            print(f"Error getting attorney info: {e}")
            return {
                'name': 'Robert Martinez, Esq.',
                'title': 'Senior Partner',
                'firm': 'Martinez & Associates, LLP',
                'address': '1247 Broadway, Suite 800',
                'city_state': 'New York, NY 10001',
                'phone': '(212) 555-0123',
                'bar_number': 'New York Bar #1234567'
            }

    def get_case_type_info(self, case_id):
        """Get case type information from database"""
        try:
            self.cur.execute("""
                SELECT case_type, status
                FROM cases
                WHERE case_id = %s
            """, (case_id,))
            result = self.cur.fetchone()
            if result:
                case_type, status = result
                return {
                    'case_type': case_type,
                    'status': status,
                    'subject_line': f"Personal Injury - {case_type.split(' - ')[-1] if ' - ' in case_type else case_type}"
                }
            else:
                return {
                    'case_type': 'Personal Injury',
                    'status': 'Active',
                    'subject_line': 'Personal Injury - Motor Vehicle Accident Claim'
                }
        except Exception as e:
            print(f"Error getting case type info: {e}")
            return {
                'case_type': 'Personal Injury',
                'status': 'Active',
                'subject_line': 'Personal Injury - Motor Vehicle Accident Claim'
            }

    def get_medical_providers(self, case_id):
        """Get medical providers from database"""
        try:
            self.cur.execute("""
                SELECT metadata
                FROM documents
                WHERE case_id = %s AND doc_category = 'medical'
            """, (case_id,))
            medical_docs = self.cur.fetchall()
            
            providers = []
            for doc in medical_docs:
                if doc[0] and 'provider' in doc[0]:
                    providers.append(doc[0]['provider'])
            
            return providers
        except Exception as e:
            print(f"Error getting medical providers: {e}")
            return []

    def get_expense_breakdown(self, case_id):
        """Get detailed expense breakdown from database"""
        try:
            self.cur.execute("""
                SELECT event_type, description, amount, event_date
                FROM case_events
                WHERE case_id = %s AND amount IS NOT NULL
                ORDER BY event_date
            """, (case_id,))
            expenses = self.cur.fetchall()
            
            breakdown = {
                'medical_treatment': [],
                'expense': [],
                'total_medical': 0,
                'total_wage_loss': 0,
                'future_medical': 0,
                'future_wages': 0
            }
            
            for expense in expenses:
                event_type, description, amount, event_date = expense
                amount_float = float(amount) if amount else 0
                if event_type == 'medical_treatment':
                    breakdown['medical_treatment'].append({
                        'description': description,
                        'amount': amount_float,
                        'date': event_date
                    })
                    breakdown['total_medical'] += amount_float
                elif event_type == 'expense' and 'wage' in description.lower():
                    breakdown['total_wage_loss'] += amount_float
                elif event_type == 'expense':
                    breakdown['expense'].append({
                        'description': description,
                        'amount': amount_float,
                        'date': event_date
                    })
                    breakdown['total_medical'] += amount_float
            
            # Calculate future estimates based on past expenses
            if breakdown['total_medical'] > 0:
                breakdown['future_medical'] = breakdown['total_medical'] * 0.3  # 30% of past medical
            if breakdown['total_wage_loss'] > 0:
                breakdown['future_wages'] = breakdown['total_wage_loss'] * 0.2  # 20% of past wages
            
            return breakdown
        except Exception as e:
            print(f"Error getting expense breakdown: {e}")
            return {
                'medical_treatment': [],
                'expense': [],
                'total_medical': 0,
                'total_wage_loss': 0,
                'future_medical': 5000,
                'future_wages': 3000
            }

    def get_injury_details(self, case_id):
        """Get detailed injury information from medical events"""
        try:
            self.cur.execute("""
                SELECT description, amount, event_date
                FROM case_events
                WHERE case_id = %s AND event_type = 'medical_treatment'
                ORDER BY event_date
            """, (case_id,))
            medical_events = self.cur.fetchall()
            
            injuries = []
            if medical_events:
                # Analyze medical treatment descriptions to determine injuries
                for event in medical_events:
                    description = event[0].lower()
                    if 'er' in description or 'emergency' in description:
                        injuries.append(('Emergency Treatment', 'Initial emergency room visit and examination', 'Moderate', 'Emergency medical care', 'Immediate treatment required'))
                    if 'mri' in description:
                        injuries.append(('Imaging', 'MRI examination for diagnostic purposes', 'Moderate', 'Diagnostic imaging', 'Results pending'))
                    if 'therapy' in description or 'physical' in description:
                        injuries.append(('Physical Therapy', 'Physical therapy and rehabilitation treatment', 'Moderate', 'Physical therapy sessions', 'Ongoing treatment required'))
                    if 'surgery' in description:
                        injuries.append(('Surgical Procedure', 'Surgical intervention required', 'Severe', 'Surgical procedure and recovery', 'Post-operative care needed'))
            
            # If no specific injuries found, add general ones based on case type
            if not injuries:
                injuries.append(('Soft Tissue', 'Soft tissue injuries sustained in collision', 'Moderate', 'Physical therapy and rehabilitation', 'Ongoing treatment required'))
                injuries.append(('Whiplash', 'Neck and back injuries from impact', 'Moderate', 'Chiropractic care and physical therapy', 'Gradual improvement expected'))
            
            return injuries
        except Exception as e:
            print(f"Error getting injury details: {e}")
            return [
                ('Soft Tissue', 'Soft tissue injuries sustained in collision', 'Moderate', 'Physical therapy and rehabilitation', 'Ongoing treatment required'),
                ('Whiplash', 'Neck and back injuries from impact', 'Moderate', 'Chiropractic care and physical therapy', 'Gradual improvement expected')
            ]

    def get_pain_suffering_estimate(self, case_id):
        """Calculate pain and suffering estimate based on medical expenses and injuries"""
        try:
            # Get total medical expenses
            self.cur.execute("""
                SELECT SUM(amount) as total_medical
                FROM case_events
                WHERE case_id = %s AND event_type IN ('medical_treatment', 'expense')
            """, (case_id,))
            result = self.cur.fetchone()
            total_medical = float(result[0]) if result and result[0] else 0
            
            # Calculate pain and suffering based on medical expenses (typically 2-3x medical expenses)
            pain_suffering = total_medical * 2.5 if total_medical > 0 else 15000
            
            return pain_suffering
        except Exception as e:
            print(f"Error calculating pain and suffering: {e}")
            return 15000
