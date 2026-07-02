import re
import os
from typing import Callable, Optional, Dict, List, Tuple, Any

class DomainSpec:
    def __init__(
        self,
        id: str,
        name: str,
        persona_prompt: str,
        guardrail_level: str = "standard",
        completeness_validator: Optional[Callable[[str, str], bool]] = None,
        extraction_hint_fn: Optional[Callable[[str, str], Optional[str]]] = None,
        postprocess_fn: Optional[Callable[[str], str]] = None,
    ):
        self.id = id
        self.name = name
        self.persona_prompt = persona_prompt
        self.guardrail_level = guardrail_level
        self.completeness_validator = completeness_validator
        self.extraction_hint_fn = extraction_hint_fn
        self.postprocess_fn = postprocess_fn

# ---------------------------------------------------------------------------
# Common Helpers
# ---------------------------------------------------------------------------

def split_document_pages_local(document_context: str) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    for m in re.finditer(r"\[Page\s+(\d+)\]", document_context):
        page = int(m.group(1))
        start = m.end()
        pages.append((page, document_context[start:]))
    if not pages:
        return []
    out: List[Tuple[int, str]] = []
    for i, (page, text_tail) in enumerate(pages):
        next_marker = re.search(r"\[Page\s+\d+\]", text_tail)
        if i < len(pages) - 1 and next_marker:
            out.append((page, text_tail[: next_marker.start()]))
        else:
            out.append((page, text_tail))
    return out

def standard_table_value_hint(document_context: str, question: str) -> Optional[str]:
    q = question.lower()
    year_match = re.search(r"\b(19|20)\d{2}\b", q)
    if not year_match:
        return None
    year = year_match.group(0)

    metric_keywords = [
        "net profit",
        "profit",
        "revenue",
        "ebitda",
        "income",
        "expense",
        "cash flow",
    ]
    metric = next((k for k in metric_keywords if k in q), None)
    if metric is None:
        return None

    number_re = re.compile(
        r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|million|billion|m|bn|crore|lakh))?",
        re.I,
    )

    for page, text in split_document_pages_local(document_context):
        low = text.lower()
        if metric not in low or year not in low:
            continue

        idx = low.find(year)
        window_start = max(0, idx - 220)
        window_end = min(len(text), idx + 260)
        window = text[window_start:window_end]
        candidates = [m.group(0).strip() for m in number_re.finditer(window) if m.group(0).strip()]

        candidates = [c for c in candidates if c != year]
        if not candidates:
            continue

        value = candidates[0]
        return (
            f"Heuristic table hint: potential value for '{metric}' in {year} appears to be '{value}' "
            f"on [Page {page}]. Verify against table cells before final answer."
        )

    return None

# ---------------------------------------------------------------------------
# Completeness Validators
# ---------------------------------------------------------------------------

def check_mcq_or_cite_or_punctuation(text: str, question: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if re.search(r"Answer:\s*[A-D]\s*$", t, flags=re.I):
        return True
    if re.search(r"\bAnswer:\s*[A-D]\b", t, flags=re.I) and re.search(r"(^|\n)\s*\d+[\.\)]\s+", t):
        return True
    if re.search(r"\[Page\s+\d+\]\s*$", t):
        return True
    if re.search(r"[.!?]$", t):
        return True
    return False

def resume_completeness_validator(text: str, question: str) -> bool:
    t = text.strip()
    if check_mcq_or_cite_or_punctuation(text, question):
        return True
    if len(t) >= 40 and re.search(r"[A-Za-z0-9)]$", t):
        return True
    return False

def patents_completeness_validator(text: str, question: str) -> bool:
    t = text.strip()
    if check_mcq_or_cite_or_punctuation(text, question):
        return True
    if re.search(r"(?:Claim|claims|US\s*\d+[\d,]*)(?:\s+\d+)?\s*$", t, flags=re.I):
        return True
    return False

# ---------------------------------------------------------------------------
# Domain Registry Setup
# ---------------------------------------------------------------------------

DOMAIN_REGISTRY: Dict[str, DomainSpec] = {
    "legal": DomainSpec(
        id="legal",
        name="Legal / Contracts",
        persona_prompt=(
            "You are a Senior Lead Auditor. You are strict, precise, and terse.\n"
            "Analyze clause and section references (e.g. Section 4(b)(ii)), track defined terms, "
            "and extract obligations and rights accurately."
        ),
        guardrail_level="standard",
    ),
    "accounting": DomainSpec(
        id="accounting",
        name="Accounting / Financial Statements",
        persona_prompt=(
            "You are a precise financial analyst.\n"
            "Focus on year-wise and metric comparisons. Include details about table row/column references "
            "in your citations. Normalize units (e.g. crores, lakhs, millions, billions) and currencies where appropriate."
        ),
        guardrail_level="standard",
        extraction_hint_fn=standard_table_value_hint,
    ),
    "resume": DomainSpec(
        id="resume",
        name="Resumes / CVs",
        persona_prompt=(
            "You are a professional CV and resume evaluator. Be neutral, objective, and non-adversarial.\n"
            "Extract structured fields like skills, experience, and education, and identify any timeline or employment gaps."
        ),
        guardrail_level="standard",
        completeness_validator=resume_completeness_validator,
    ),
    "research": DomainSpec(
        id="research",
        name="Research Papers",
        persona_prompt=(
            "You are a precise academic researcher. Hedge your claims appropriately.\n"
            "Pay attention to section types (Abstract, Methodology, Results, Discussion) and distinguish "
            "academic citations/references from figure numbers."
        ),
        guardrail_level="standard",
    ),
    "medical": DomainSpec(
        id="medical",
        name="Medical / Clinical",
        persona_prompt=(
            "You are a cautious medical information specialist.\n"
            "Use explicit medical disclaimers. Never provide clinical or treatment advice.\n"
            "Be extremely precise with medical terminology, dosages, and CPT/ICD codes.\n"
            "Maintain a strong bias to state 'not found' rather than guessing."
        ),
        guardrail_level="strict",
    ),
    "insurance": DomainSpec(
        id="insurance",
        name="Insurance Policies",
        persona_prompt=(
            "You are an insurance policy specialist.\n"
            "Focus heavily on coverage vs exclusions, premiums, deductibles, and matching policy conditions to specific scenarios."
        ),
        guardrail_level="standard",
        extraction_hint_fn=standard_table_value_hint,
    ),
    "technical": DomainSpec(
        id="technical",
        name="Technical / Engineering Specs",
        persona_prompt=(
            "You are a precise technical specifications engineer.\n"
            "Focus on numeric tolerances, specifications, operating limits, unit conversions, and comparing changes across document revisions."
        ),
        guardrail_level="standard",
        extraction_hint_fn=standard_table_value_hint,
    ),
    "hr": DomainSpec(
        id="hr",
        name="HR Policies / Handbooks",
        persona_prompt=(
            "You are a neutral HR policy advisor. Adhere strictly to the literal policy text.\n"
            "Distinguish between policies and exceptions, note leave/benefit structures, and be aware of jurisdictional boundaries."
        ),
        guardrail_level="standard",
    ),
    "government": DomainSpec(
        id="government",
        name="Government / Regulatory Filings",
        persona_prompt=(
            "You are a formal compliance specialist.\n"
            "Parse structured form-field data and check requirements against checklist or amendment changes."
        ),
        guardrail_level="standard",
    ),
    "patents": DomainSpec(
        id="patents",
        name="Patents / IP Filings",
        persona_prompt=(
            "You are a patent specialist.\n"
            "Distinguish between independent and dependent claims. Cite page numbers and use separate prior-art patent citation format "
            "(e.g. US 10,123,456 B2) when referencing patent numbers. Keep background information separate from legally binding claims."
        ),
        guardrail_level="standard",
        completeness_validator=patents_completeness_validator,
    ),
}

# ---------------------------------------------------------------------------
# Domain Detection Flow
# ---------------------------------------------------------------------------

def detect_domain_tier1(text: str) -> Optional[str]:
    low = text.lower()
    scores = {d_id: 0 for d_id in DOMAIN_REGISTRY.keys()}

    # 1. Legal
    if "whereas" in low:
        scores["legal"] += 3
    if "governed by" in low:
        scores["legal"] += 2
    if "indemnification" in low or "indemnify" in low:
        scores["legal"] += 2
    if "hereby" in low:
        scores["legal"] += 1
    if "confidentiality agreement" in low or "nda" in low:
        scores["legal"] += 2

    # 2. Accounting
    if "balance sheet" in low:
        scores["accounting"] += 4
    if "ebitda" in low:
        scores["accounting"] += 3
    if "net profit" in low or "net income" in low:
        scores["accounting"] += 2
    if "crore" in low or "lakh" in low:
        scores["accounting"] += 2
    if "liabilities" in low and "assets" in low:
        scores["accounting"] += 2

    # 3. Resume
    if "work experience" in low or "professional experience" in low or "employment history" in low:
        scores["resume"] += 4
    if "curriculum vitae" in low or "cv" in low or "resume" in low:
        scores["resume"] += 3
    if "education" in low and "skills" in low and ("experience" in low or "projects" in low):
        scores["resume"] += 2

    # 4. Research
    if "abstract" in low and "methodology" in low:
        scores["research"] += 4
    if "introduction" in low and "conclusion" in low and "references" in low:
        scores["research"] += 2
    if "results and discussion" in low:
        scores["research"] += 2
    if "doi:" in low:
        scores["research"] += 3

    # 5. Medical
    if "patient" in low:
        scores["medical"] += 2
    if "diagnosis" in low or "clinical" in low:
        scores["medical"] += 3
    if "dosage" in low or "prescription" in low or "treatment plan" in low:
        scores["medical"] += 3
    if "icd-10" in low or "cpt code" in low:
        scores["medical"] += 4

    # 6. Insurance
    if "insurance policy" in low or "policyholder" in low:
        scores["insurance"] += 4
    if "coverage" in low and "exclusions" in low:
        scores["insurance"] += 3
    if "deductible" in low or "premium" in low:
        scores["insurance"] += 3

    # 7. Technical
    if "datasheet" in low or "data sheet" in low:
        scores["technical"] += 3
    if "specification" in low and "tolerance" in low:
        scores["technical"] += 2
    if "operating voltage" in low or "schematic" in low:
        scores["technical"] += 3

    # 8. HR
    if "employee handbook" in low:
        scores["hr"] += 5
    if "paid time off" in low or "pto" in low:
        scores["hr"] += 3
    if "leave policy" in low or "code of conduct" in low:
        scores["hr"] += 3

    # 9. Government
    if "form 10-k" in low or "sec filing" in low:
        scores["government"] += 4
    if "regulatory filing" in low or "public record" in low:
        scores["government"] += 3

    # 10. Patents
    if "patent" in low:
        scores["patents"] += 3
    if "claim 1" in low or "independent claim" in low:
        scores["patents"] += 4
    if "prior art" in low:
        scores["patents"] += 3
    if "inventor" in low and "assignee" in low:
        scores["patents"] += 2

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_id, top_score = sorted_scores[0]
    runner_up_id, runner_up_score = sorted_scores[1]

    if top_score >= 3 and (top_score - runner_up_score) >= 2:
        return top_id
    return None

async def detect_domain_tier2(text_excerpt: str) -> str:
    try:
        from app.rag.chains import build_llm
        llm = build_llm(provider="gemini", model="gemini-1.5-flash")
    except Exception:
        try:
            from app.rag.chains import build_llm
            llm = build_llm(provider="ollama")
        except Exception:
            return "legal"

    prompt = (
        "Classify the following text excerpt from a document into one of these 10 categories:\n"
        "1. legal (Contracts, agreements, NDAs, terms of service)\n"
        "2. accounting (Financial statements, balance sheets, profit & loss, accounting audits)\n"
        "3. resume (CVs, resumes, professional portfolios)\n"
        "4. research (Academic papers, scientific articles, studies)\n"
        "5. medical (Clinical documents, patient diagnosis, health records, medical prescriptions)\n"
        "6. insurance (Insurance policies, coverage details, claims, deductibles)\n"
        "7. technical (Engineering specifications, datasheets, technical manuals)\n"
        "8. hr (Employee handbooks, HR policies, benefits documents, workplace conduct)\n"
        "9. government (Regulatory filings, tax forms, compliance documents)\n"
        "10. patents (Patent filings, patent claims, prior art)\n\n"
        "Respond with ONLY the category id (e.g., 'legal', 'accounting', 'resume', 'research', 'medical', 'insurance', 'technical', 'hr', 'government', or 'patents') and nothing else.\n\n"
        f"Text excerpt:\n{text_excerpt[:2000]}"
    )
    try:
        from langchain_core.messages import HumanMessage
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        result = (getattr(resp, "content", None) or str(resp)).strip().lower()
        for d_id in DOMAIN_REGISTRY.keys():
            if d_id in result:
                return d_id
    except Exception:
        pass
    return "legal"

async def auto_detect_document_domain(full_context: str) -> str:
    excerpt = full_context[:4000]
    detected = detect_domain_tier1(excerpt)
    if detected:
        return detected
    return await detect_domain_tier2(excerpt)
