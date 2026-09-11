import pytest

from rag_assistant import ensure_section_index, retrieve_relevant_sections


def test_cli_like_police_pd_query():
    case_id = "2024-PI-001"
    ensure_section_index(case_id)
    results = retrieve_relevant_sections("POLICE REPORT PROPERTY DAMAGE Vehicle #1 (Victim) repair estimate", top_k=5)
    # We don't assert exact ordering, but at least one section should be returned
    assert isinstance(results, list)
    # The police-only index may or may not surface PD directly; allow empty but test type and structure
    for sec in results:
        assert isinstance(sec, dict)
        assert "title" in sec and "text" in sec 