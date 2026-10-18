"""Research-only clinical evidence assistant built with LangChain and LangGraph.

This prototype is not a medical device and must not be used for patient care.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal, TypedDict

import requests
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi


DATASET_NAME = "galileo-ai/ragbench"
DATASET_CONFIG = "covidqa"
DATASET_URL = "https://huggingface.co/datasets/galileo-ai/ragbench"
DATASET_LICENSE = "CC BY 4.0"
OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
MAX_MEDICATIONS = 6
MAX_LABEL_CHARS = 3_000

DISCLAIMER = (
    "Research prototype only — not for diagnosis, treatment, medication changes, "
    "or dosing. RAGBench covidqa is benchmark data, not a current clinical "
    "guideline. Verify all information with a qualified clinician or pharmacist. "
    "Do not enter identifying patient information; prompts are sent to the "
    "configured Groq model provider."
)
EMERGENCY_PATTERNS = (
    r"\b(chest pain|pressure in (my|the) chest)\b",
    r"\b(can't breathe|cannot breathe|difficulty breathing|severe shortness of breath)\b",
    r"\b(severe allergic reaction|anaphylaxis|throat is closing)\b",
    r"\b(uncontrolled bleeding|severe bleeding|bleeding won't stop)\b",
    r"\b(face|arm|leg) weakness\b",
    r"\b(sudden confusion|loss of consciousness|unconscious)\b",
)


class TriageResult(BaseModel):
    category: str = Field(description="Short category for the question.")
    severity: Literal["low", "moderate", "high", "critical"] = Field(
        description="Potential clinical risk; use critical for a possible emergency."
    )
    emergency_concern: bool = Field(
        description="True if the user describes a possible immediate emergency."
    )
    clinician_review: bool = Field(
        description="True if the question needs individualized clinician review."
    )
    medications: list[str] = Field(
        default_factory=list,
        description="Medication names explicitly mentioned by the user, if any.",
    )
    rationale: str = Field(description="Brief explanation without diagnosing.")


class InteractionFinding(BaseModel):
    medications: list[str] = Field(
        description="The medication names involved in the possible interaction."
    )
    source_id: str = Field(description="An FDA source ID provided in the input.")
    exact_quote: str = Field(
        description="An exact excerpt from that label supporting the concern."
    )


class InteractionAssessment(BaseModel):
    findings: list[InteractionFinding] = Field(default_factory=list)
    summary: str = Field(
        description="Brief screening result; do not claim a medication combination is safe."
    )
    clinician_review: bool = Field(
        description="True when an interaction concern or uncertainty warrants review."
    )


class SafetyAssessment(BaseModel):
    unsafe_or_unsupported_claims: list[str] = Field(
        default_factory=list,
        description=(
            "Claims in the draft that are unsupported by the context, or that "
            "give a diagnosis, dose, or individualized treatment instruction."
        ),
    )
    requires_clinician_review: bool = Field(
        description="True for uncertainty or individualized clinical decisions."
    )


class DrugLabel(TypedDict):
    source_id: str
    medication: str
    status: str
    url: str
    text: str


class ClinicalState(TypedDict, total=False):
    question: str
    user_medications: list[str]
    triage: TriageResult
    medications: list[str]
    evidence: list[Document]
    drug_labels: list[DrugLabel]
    interaction_assessment: InteractionAssessment
    draft: str
    answer: str
    review_required: bool
    safety_notes: list[str]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


def _chunk_text(text: str, size: int = 1_200, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_ragbench_documents(limit: int | None = None) -> list[Document]:
    """Load only source documents, never the benchmark's question/answer fields."""
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    documents: list[Document] = []
    row_limit = len(dataset) if limit is None else min(len(dataset), limit)

    for row_number in range(row_limit):
        row = dataset[row_number]
        row_id = str(row.get("id") or row_number)
        source_documents = row.get("documents") or []
        for document_number, text in enumerate(source_documents, start=1):
            if not isinstance(text, str):
                continue
            for chunk_number, chunk in enumerate(_chunk_text(text), start=1):
                source_id = f"RAG-{len(documents) + 1:04d}"
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source_id": source_id,
                            "source": (
                                f"RAGBench covidqa, train row {row_id}, "
                                f"document {document_number}, chunk {chunk_number}"
                            ),
                            "dataset_url": DATASET_URL,
                            "license": DATASET_LICENSE,
                        },
                    )
                )

    if not documents:
        raise ValueError(
            f"No source documents found in {DATASET_NAME!r} config "
            f"{DATASET_CONFIG!r}."
        )
    return documents


class HybridRetriever:
    """Combine local semantic similarity and BM25 keyword rankings."""

    def __init__(self, documents: list[Document], embeddings: HuggingFaceEmbeddings):
        self.documents = documents
        self.vector_store = InMemoryVectorStore.from_documents(documents, embeddings)
        self.bm25 = BM25Okapi([_tokens(document.page_content) for document in documents])
        self.documents_by_id = {
            document.metadata["source_id"]: document for document in documents
        }

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        if not query.strip():
            return []

        limit = min(top_k, len(self.documents))
        semantic_results = self.vector_store.similarity_search(query, k=limit)
        keyword_results = self.bm25.get_top_n(
            _tokens(query), self.documents, n=limit
        )

        scores: defaultdict[str, float] = defaultdict(float)
        for ranked_documents in (semantic_results, keyword_results):
            for rank, document in enumerate(ranked_documents, start=1):
                source_id = document.metadata["source_id"]
                scores[source_id] += 1.0 / (60 + rank)

        ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
        return [self.documents_by_id[source_id] for source_id in ranked_ids]


def _invoke_structured(
    llm: ChatGroq,
    schema: type[BaseModel],
    messages: list[tuple[str, str]],
) -> BaseModel:
    result = llm.with_structured_output(schema).invoke(messages)
    if isinstance(result, schema):
        return result
    if isinstance(result, dict):
        return schema.model_validate(result)
    raise TypeError(f"Expected {schema.__name__} from structured model output.")


def _openfda_query(medication: str) -> tuple[dict[str, str], str]:
    escaped = medication.replace("\\", "\\\\").replace('"', '\\"')
    search = (
        f'openfda.generic_name:"{escaped}" OR '
        f'openfda.brand_name:"{escaped}"'
    )
    params = {"search": search, "limit": "1"}
    prepared = requests.Request("GET", OPENFDA_LABEL_URL, params=params).prepare()
    if prepared.url is None:
        raise RuntimeError("Could not construct the OpenFDA query URL.")
    return params, prepared.url


def _fetch_drug_labels(medications: list[str]) -> tuple[list[DrugLabel], list[str]]:
    labels: list[DrugLabel] = []
    notes: list[str] = []

    for index, medication in enumerate(medications[:MAX_MEDICATIONS], start=1):
        params, url = _openfda_query(medication)
        source_id = f"FDA-{index:03d}"
        try:
            response = requests.get(OPENFDA_LABEL_URL, params=params, timeout=15)
            if response.status_code == 404:
                labels.append(
                    {
                        "source_id": source_id,
                        "medication": medication,
                        "status": "not found",
                        "url": response.url,
                        "text": "",
                    }
                )
                notes.append(
                    f"{source_id}: OpenFDA returned no matching label for "
                    f"{medication!r}; this does not establish safety."
                )
                continue
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as error:
            labels.append(
                {
                    "source_id": source_id,
                    "medication": medication,
                    "status": "unavailable",
                    "url": url,
                    "text": "",
                }
            )
            notes.append(
                f"{source_id}: OpenFDA lookup failed for {medication!r}: {error}"
            )
            continue

        results = payload.get("results", [])
        if not results:
            labels.append(
                {
                    "source_id": source_id,
                    "medication": medication,
                    "status": "not found",
                    "url": response.url,
                    "text": "",
                }
            )
            notes.append(
                f"{source_id}: no matching OpenFDA label for {medication!r}; "
                "this does not establish safety."
            )
            continue

        result = results[0]
        sections: list[str] = []
        for section_name in ("drug_interactions", "contraindications", "warnings"):
            values = result.get(section_name, [])
            if isinstance(values, str):
                values = [values]
            for value in values:
                if isinstance(value, str) and value.strip():
                    sections.append(f"{section_name}: {value.strip()}")

        text = "\n\n".join(sections)[:MAX_LABEL_CHARS]
        if not text:
            notes.append(
                f"{source_id}: a label was found for {medication!r}, but the "
                "requested interaction/warning sections were absent."
            )
        labels.append(
            {
                "source_id": source_id,
                "medication": medication,
                "status": "retrieved",
                "url": response.url,
                "text": text,
            }
        )

    if len(medications) > MAX_MEDICATIONS:
        notes.append(
            f"Only the first {MAX_MEDICATIONS} medication names were checked."
        )
    return labels, notes


def create_workflow(retriever: HybridRetriever, llm: ChatGroq):
    def triage_node(state: ClinicalState) -> dict[str, object]:
        triage = _invoke_structured(
            llm,
            TriageResult,
            [
                (
                    "system",
                    "Triage a healthcare question for a research-only decision "
                    "support prototype. Do not diagnose. Mark emergencies and "
                    "individualized medication/treatment decisions for clinician "
                    "review. Extract only medication names explicitly present.",
                ),
                (
                    "human",
                    f"Question: {state['question']}\n"
                    f"User-supplied medication list: "
                    f"{', '.join(state.get('user_medications', [])) or 'none'}",
                ),
            ],
        )
        if not isinstance(triage, TriageResult):
            raise TypeError("Triage agent returned an invalid result.")

        medications = list(
            dict.fromkeys(
                [*state.get("user_medications", []), *triage.medications]
            )
        )
        safety_notes = list(state.get("safety_notes", []))
        if len(medications) > MAX_MEDICATIONS:
            safety_notes.append(
                f"Only the first {MAX_MEDICATIONS} medication names were screened."
            )
        return {
            "triage": triage,
            "medications": medications[:MAX_MEDICATIONS],
            "safety_notes": safety_notes,
        }

    def retrieval_node(state: ClinicalState) -> dict[str, object]:
        evidence = retriever.search(state["question"], top_k=5)
        return {"evidence": evidence}

    def medication_check_node(state: ClinicalState) -> dict[str, object]:
        labels, notes = _fetch_drug_labels(state.get("medications", []))
        notes = [*state.get("safety_notes", []), *notes]
        available_labels = [
            label for label in labels if label["status"] == "retrieved" and label["text"]
        ]
        if available_labels:
            label_context = "\n\n".join(
                f"[{label['source_id']}] Medication: {label['medication']}\n"
                f"{label['text']}"
                for label in available_labels
            )
            assessment = _invoke_structured(
                llm,
                InteractionAssessment,
                [
                    (
                        "system",
                        "Screen medication names only against the supplied OpenFDA "
                        "label excerpts. Report a possible interaction only when "
                        "an exact excerpt explicitly supports it. Never infer that "
                        "a combination is safe. A label excerpt may not cover all "
                        "interactions. Use only supplied FDA source IDs and quotes.",
                    ),
                    (
                        "human",
                        f"Medication names to screen: {state.get('medications', [])}\n"
                        f"Label excerpts:\n{label_context}",
                    ),
                ],
            )
            if not isinstance(assessment, InteractionAssessment):
                raise TypeError("Medication interaction agent returned an invalid result.")

            verified_findings: list[InteractionFinding] = []
            label_by_id = {label["source_id"]: label for label in available_labels}
            for finding in assessment.findings:
                source = label_by_id.get(finding.source_id)
                if (
                    source is not None
                    and finding.exact_quote.strip()
                    and finding.exact_quote.strip() in source["text"]
                ):
                    verified_findings.append(finding)
                else:
                    notes.append(
                        "An interaction statement from the model was omitted "
                        "because its source quote could not be verified."
                    )
            assessment = assessment.model_copy(
                update={
                    "findings": verified_findings,
                    "summary": (
                        "Potential interaction information was found in label "
                        "text; have a clinician or pharmacist verify it."
                        if verified_findings
                        else "No explicit interaction was verified in the retrieved "
                        "label excerpts. This does not establish that the medication "
                        "combination is safe; confirm with a clinician or pharmacist."
                    ),
                }
            )
        else:
            assessment = InteractionAssessment(
                findings=[],
                summary=(
                    "No medication names were provided or extracted; screening "
                    "was not performed."
                    if not state.get("medications")
                    else (
                        "No usable OpenFDA interaction-label text was retrieved. "
                        "No interaction conclusion can be made."
                    )
                ),
                clinician_review=bool(state.get("medications")),
            )

        return {
            "drug_labels": labels,
            "interaction_assessment": assessment,
            "safety_notes": notes,
        }

    def synthesis_node(state: ClinicalState) -> dict[str, object]:
        evidence_context = "\n\n".join(
            f"[{document.metadata['source_id']}] "
            f"{document.metadata['source']}\n{document.page_content}"
            for document in state.get("evidence", [])
        )
        label_context = "\n\n".join(
            f"[{label['source_id']}] {label['medication']} "
            f"({label['status']}): {label['text'] or 'No usable label text.'}"
            for label in state.get("drug_labels", [])
        )
        triage = state["triage"]
        assessment = state["interaction_assessment"]
        message = llm.invoke(
            [
                (
                    "system",
                    "You are the synthesis stage of a research-only clinical "
                    "evidence assistant. Do not diagnose, prescribe, recommend "
                    "starting/stopping/changing medicine, or provide a patient-"
                    "specific dose. Use only supplied excerpts. Qualify uncertainty. "
                    "Cite each factual clinical statement with its exact supplied "
                    "ID, such as [RAG-0001] or [FDA-001]. If evidence is insufficient, "
                    "say so. Treat RAGBench covidqa as benchmark text, not current "
                    "clinical guidance. Never claim that absence of an interaction "
                    "in retrieved labels means a combination is safe.",
                ),
                (
                    "human",
                    f"Question: {state['question']}\n"
                    f"Triage: {triage.model_dump_json()}\n"
                    f"RAGBench excerpts:\n{evidence_context or 'None retrieved.'}\n"
                    f"OpenFDA label excerpts:\n{label_context or 'None retrieved.'}\n"
                    f"Interaction screening: {assessment.model_dump_json()}\n"
                    "Provide a concise evidence summary, state any important "
                    "limitations, and avoid giving individualized medical advice.",
                ),
            ]
        )
        draft = message.content
        if not isinstance(draft, str):
            raise TypeError("Synthesis agent returned non-text content.")
        return {"draft": draft}

    def safety_node(state: ClinicalState) -> dict[str, object]:
        triage = state["triage"]
        assessment = state["interaction_assessment"]
        evidence = state.get("evidence", [])
        labels = state.get("drug_labels", [])
        valid_ids = {
            document.metadata["source_id"] for document in evidence
        } | {
            label["source_id"]
            for label in labels
            if label["status"] == "retrieved" and label["text"]
        }
        used_ids = set(re.findall(r"\[(?:RAG|FDA)-\d+\]", state["draft"]))
        notes = list(state.get("safety_notes", []))
        unsafe = False
        deterministic_emergency = any(
            re.search(pattern, state["question"], flags=re.IGNORECASE)
            for pattern in EMERGENCY_PATTERNS
        )
        if deterministic_emergency and not triage.emergency_concern:
            notes.append(
                "A possible emergency symptom matched a conservative safety rule."
            )

        if not evidence:
            unsafe = True
            notes.append("No RAGBench evidence was retrieved for this question.")
        if used_ids - {f"[{source_id}]" for source_id in valid_ids}:
            unsafe = True
            notes.append("The draft contains one or more unverified source IDs.")
        if evidence and not used_ids:
            unsafe = True
            notes.append("The draft did not cite a retrieved evidence source.")

        safety_review = _invoke_structured(
            llm,
            SafetyAssessment,
            [
                (
                    "system",
                    "Audit the draft for unsupported clinical claims, diagnoses, "
                    "patient-specific doses, or individualized treatment instructions. "
                    "Do not rewrite it. Flag uncertainty for clinician review. "
                    "Be conservative.",
                ),
                (
                    "human",
                    f"Question: {state['question']}\n"
                    f"Evidence:\n"
                    + "\n\n".join(
                        f"[{document.metadata['source_id']}] {document.page_content}"
                        for document in evidence
                    )
                    + "\n\nDraft:\n"
                    + state["draft"],
                ),
            ],
        )
        if not isinstance(safety_review, SafetyAssessment):
            raise TypeError("Safety agent returned an invalid result.")
        if safety_review.unsafe_or_unsupported_claims:
            unsafe = True
            notes.extend(safety_review.unsafe_or_unsupported_claims)

        review_required = (
            triage.severity in {"high", "critical"}
            or triage.emergency_concern
            or deterministic_emergency
            or triage.clinician_review
            or assessment.clinician_review
            or bool(assessment.findings)
            or safety_review.requires_clinician_review
            or bool(state.get("medications"))
        )

        answer = state["draft"]
        if unsafe:
            answer = (
                "I can't provide a dependable clinical summary from the retrieved "
                "evidence. Please have a licensed clinician or pharmacist review "
                "the question and the cited source material."
            )
            review_required = True

        if (
            triage.emergency_concern
            or triage.severity == "critical"
            or deterministic_emergency
        ):
            answer = (
                "This may require urgent in-person assessment. Contact your local "
                "emergency number or seek emergency care now. Do not rely on this "
                "research prototype for emergency decisions."
            )
            review_required = True

        return {
            "answer": answer,
            "review_required": review_required,
            "safety_notes": notes,
        }

    workflow = StateGraph(ClinicalState)
    workflow.add_node("triage", triage_node)
    workflow.add_node("hybrid_retrieval", retrieval_node)
    workflow.add_node("drug_interaction_check", medication_check_node)
    workflow.add_node("evidence_synthesis", synthesis_node)
    workflow.add_node("safety_gate", safety_node)
    workflow.set_entry_point("triage")
    workflow.add_edge("triage", "hybrid_retrieval")
    workflow.add_edge("hybrid_retrieval", "drug_interaction_check")
    workflow.add_edge("drug_interaction_check", "evidence_synthesis")
    workflow.add_edge("evidence_synthesis", "safety_gate")
    workflow.add_edge("safety_gate", END)
    return workflow.compile()


def _format_result(state: ClinicalState) -> str:
    triage = state["triage"]
    assessment = state["interaction_assessment"]
    lines = [
        "=== Clinical evidence support (research prototype) ===",
        f"Triage: {triage.category} | severity: {triage.severity}",
        f"Clinician review: {'REQUIRED' if state['review_required'] else 'not flagged'}",
        "",
        state["answer"],
        "",
        "Medication-label screen:",
        assessment.summary,
    ]

    for finding in assessment.findings:
        lines.append(
            f"- Potential interaction information for "
            f"{', '.join(finding.medications)}: "
            f"\"{finding.exact_quote}\" [{finding.source_id}]"
        )

    if state.get("evidence"):
        lines.extend(["", "Retrieved RAGBench evidence (benchmark data, CC BY 4.0):"])
        for document in state["evidence"]:
            lines.append(
                f"- [{document.metadata['source_id']}] {document.metadata['source']} "
                f"| {document.metadata['dataset_url']}"
            )

    if state.get("drug_labels"):
        lines.extend(["", "OpenFDA label records:"])
        for label in state["drug_labels"]:
            lines.append(
                f"- [{label['source_id']}] {label['medication']}: "
                f"{label['status']} | {label['url']}"
            )

    if state.get("safety_notes"):
        lines.extend(["", "Safety/availability notes:"])
        lines.extend(f"- {note}" for note in state["safety_notes"])

    lines.extend(["", DISCLAIMER])
    return "\n".join(lines)


def build_assistant(max_rows: int | None = None):
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to the project .env file or environment."
        )
    if max_rows is None:
        max_rows = int(os.getenv("RAGBENCH_MAX_ROWS", "0"))
    if max_rows < 0:
        raise ValueError("max_rows must be zero or a positive integer.")

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
        max_retries=2,
        timeout=60,
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )
    documents = load_ragbench_documents(limit=max_rows or None)
    return create_workflow(HybridRetriever(documents, embeddings), llm)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(
        description="Research-only medical RAG demo using RAGBench covidqa."
    )
    parser.add_argument(
        "--question",
        help="Run one question and exit. Omit to use interactive mode.",
    )
    parser.add_argument(
        "--medications",
        default="",
        help="Optional comma-separated medication names for label screening.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=int(os.getenv("RAGBENCH_MAX_ROWS", "0")),
        help="Limit indexed train rows for a quick demo; 0 indexes all rows.",
    )
    args = parser.parse_args()
    app = build_assistant(args.max_rows)

    if args.question:
        user_medications = [
            name.strip() for name in args.medications.split(",") if name.strip()
        ]
        result = app.invoke(
            {"question": args.question, "user_medications": user_medications}
        )
        print(_format_result(result))
        return

    print(DISCLAIMER)
    print("Enter 'exit' to quit.\n")
    while True:
        question = input("Clinical question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        medication_input = input(
            "Medication names to screen (comma-separated, optional): "
        ).strip()
        user_medications = [
            name.strip() for name in medication_input.split(",") if name.strip()
        ]
        result = app.invoke(
            {"question": question, "user_medications": user_medications}
        )
        print(_format_result(result))
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)