"""Offline tests for evaluation matching; does not call Ollama."""
from evaluation import evaluate_retrieval_metrics, evaluate_citation_accuracy

chunks = [
    {
        "file": "jjact2015.pdf",
        "act_title": "Juvenile Justice (Care and Protection of Children) Act, 2015",
        "section_hint": "Section 2(13)",
        "text": "2. Definitions. (13) child in conflict with law means any person...",
    },
    {
        "file": "jjact2015.pdf",
        "act_title": "Juvenile Justice (Care and Protection of Children) Act, 2015",
        "section_hint": "Section 12",
        "text": "12. Bail. Every child alleged to have committed an offence shall be released on bail...",
    },
]

m = evaluate_retrieval_metrics(
    ["Section 2(13)"], chunks, top_k=5,
    target_act="Juvenile Justice (Care and Protection of Children) Act, 2015",
)
assert m["hit_1"] == 1.0 and m["hit_3"] == 1.0 and m["mrr"] == 1.0, m
assert evaluate_citation_accuracy("Under Section 2(13), ...", ["Section 2(13)"]) == 1.0
print("evaluation tests passed")
