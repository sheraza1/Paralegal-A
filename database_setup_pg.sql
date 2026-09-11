-- Legal Case Management Database Setup (PostgreSQL Compatible)

-- Cases table
CREATE TABLE cases (
    case_id VARCHAR(50) PRIMARY KEY,
    case_type VARCHAR(100) NOT NULL,
    date_filed DATE NOT NULL,
    status VARCHAR(50) NOT NULL,
    attorney_id INT,
    case_summary TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Parties table  
CREATE TABLE parties (
    party_id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) NOT NULL,
    party_type VARCHAR(50) NOT NULL,
    name VARCHAR(200) NOT NULL,
    contact_info JSON,
    insurance_info JSON,
    FOREIGN KEY (case_id) REFERENCES cases(case_id)
);

-- Documents table
CREATE TABLE documents (
    doc_id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    doc_category VARCHAR(100) NOT NULL,
    upload_date DATE NOT NULL,
    document_title VARCHAR(300),
    metadata JSON,
    FOREIGN KEY (case_id) REFERENCES cases(case_id)
);

-- Case_events table
CREATE TABLE case_events (
    event_id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) NOT NULL,
    event_date DATE NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    amount DECIMAL(10,2),
    FOREIGN KEY (case_id) REFERENCES cases(case_id)
);

-- Insert data (start with cases, then parties, then documents and events)
INSERT IGNORE INTO cases (case_id, case_type, date_filed, status, attorney_id, case_summary)
VALUES (
    '2024-PI-001',
    'Personal Injury - Motor Vehicle Accident',
    '2024-03-15',
    'Active - Demand Phase',
    101,
    'Rear-end collision case where plaintiff John Smith was struck by defendant Sarah Johnson while stopped at traffic light. Plaintiff sustained soft tissue injuries and lost wages. Defendant distracted by phone.'
);

-- Sample party insert
INSERT IGNORE INTO parties (case_id, party_type, name, contact_info, insurance_info)
VALUES 
(
    '2024-PI-001',
    'plaintiff',
    'John Smith',
    '{"phone": "(555) 987-6543", "email": "jsmith@email.com"}',
    '{"company": "State Farm", "policy": "SF-789456123"}'
);

-- Parties
INSERT IGNORE INTO parties (case_id, party_type, name, contact_info, insurance_info) VALUES
('2024-PI-001', 'defendant', 'Sarah Johnson',
 '{"phone": "(555) 456-7890", "address": "789 Pine Ave, Springfield"}',
 '{"company": "ABC Insurance", "policy": "ABC-456789012"}'),

('2024-PI-001', 'witness', 'Maria Rodriguez',
 '{"phone": "(555) 321-6547", "address": "234 Elm St"}',
 '{}'),

('2024-PI-001', 'witness', 'Robert Chen',
 '{"phone": "(555) 654-3210", "address": "567 Maple Drive"}',
 '{}');

-- Documents
INSERT IGNORE INTO documents (case_id, file_path, doc_category, upload_date, document_title, metadata) VALUES
('2024-PI-001', '/sample_docs/medical_records_dr_jones.pdf', 'medical', '2024-03-20',
 'Medical Records - Dr. Jones',
 '{"provider": "Dr. Michael Jones", "total_expenses": 8950}'),

('2024-PI-001', '/sample_docs/police_report_incident_789.pdf', 'police_report', '2024-03-16',
 'Police Report - SPD #789',
 '{"fault_determination": "100% defendant"}'),

('2024-PI-001', '/sample_docs/wage_statements_2024.pdf', 'financial', '2024-07-05',
 'Wage Statements - Pacific Construction',
 '{"employer": "Pacific Construction", "total_wage_loss": 4940}'),

('2024-PI-001', '/sample_docs/insurance_correspondence.pdf', 'correspondence', '2024-06-25',
 'Insurance Communications - ABC Insurance',
 '{"settlement_range": "25000-35000", "liability_accepted": true}');

-- Case events
INSERT IGNORE INTO case_events (case_id, event_date, event_type, description, amount) VALUES
('2024-PI-001', '2024-03-15', 'accident', 'Plaintiff rear-ended at red light. Defendant distracted by phone.', NULL),

('2024-PI-001', '2024-03-16', 'medical_treatment', 'Initial ER visit and examination.', 1200.00),
('2024-PI-001', '2024-03-25', 'medical_treatment', 'MRI and 12 sessions of physical therapy.', 1400.00),
('2024-PI-001', '2024-04-15', 'expense', 'Physical therapy bill (12 sessions).', 1440.00),
('2024-PI-001', '2024-05-20', 'expense', 'More therapy sessions.', 1440.00),
('2024-PI-001', '2024-07-05', 'expense', 'Lost wages due to injury.', 4940.00),
('2024-PI-001', '2024-06-25', 'correspondence', 'Insurance counter-offer for $28,500 received.', 28500.00);
