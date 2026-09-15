"""Stable local evaluation for the Child Law RAG system.

The benchmark separates deterministic retrieval metrics from optional LLM judging.
A failed LLM call is recorded as a missing generation metric, never converted to 0.
"""

import json
import re
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import OUTPUTS_DIR, DEFAULT_TOP_K, call_llm
from answer_generation import validate_child_law_domain, generate_answer
from retrieval import (
    retrieve_bm25,
    retrieve_dense,
    retrieve_hybrid_rrf,
    retrieve_reranked,
    retrieve_crag_hyde,
)


def _normalize_provision(value: str) -> str:
    value = value.lower().replace("section", "section").replace("article", "article").replace("rule", "rule")
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[^a-z0-9()]+", "", value)
    return value


def _target_matches_chunk(target: str, chunk: Dict) -> bool:
    target_norm = _normalize_provision(target)
    hint_norm = _normalize_provision(chunk.get("section_hint", ""))
    text_norm = _normalize_provision(chunk.get("text", ""))

    if target_norm == hint_norm or target_norm in text_norm:
        return True

    # Accept common PDF formatting: Section 2 (13) == Section 2(13).
    return target_norm.replace("(", "") in text_norm.replace("(", "") and ")" in target_norm


def evaluate_retrieval_metrics(
    target_sections: List[str],
    retrieved_chunks: List[Dict],
    top_k: int = DEFAULT_TOP_K,
    target_act: Optional[str] = None,
) -> Dict[str, float]:
    """Compute Hit@1/3/5 and MRR using provision-aware matching."""
    if not target_sections or not retrieved_chunks:
        return {"hit_1": 0.0, "hit_3": 0.0, "hit_5": 0.0, "mrr": 0.0}

    def act_ok(chunk: Dict) -> bool:
        if not target_act:
            return True
        a = re.sub(r"\W+", "", target_act.lower())
        b = re.sub(r"\W+", "", chunk.get("act_title", "").lower())
        return a in b or b in a

    first_hit = None
    hits = {1: 0.0, 3: 0.0, 5: 0.0}
    limit = min(5, max(top_k, 5), len(retrieved_chunks))

    for rank, chunk in enumerate(retrieved_chunks[:limit], start=1):
        matched = act_ok(chunk) and any(_target_matches_chunk(t, chunk) for t in target_sections)
        if not matched:
            continue
        if first_hit is None:
            first_hit = rank
        for k in hits:
            if rank <= k:
                hits[k] = 1.0

    return {
        "hit_1": hits[1],
        "hit_3": hits[3],
        "hit_5": hits[5],
        "mrr": 1.0 / first_hit if first_hit else 0.0,
    }


def _token_set(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _context_support_score(answer: str, retrieved_chunks: List[Dict]) -> float:
    """Heuristic support score: fraction of nontrivial answer sentences with context overlap."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) >= 25]
    if not sentences:
        return 0.0
    context = " ".join(c.get("text", "") for c in retrieved_chunks)
    context_tokens = _token_set(context)
    supported = 0
    for sentence in sentences:
        tokens = _token_set(sentence)
        informative = {t for t in tokens if len(t) > 3}
        overlap = len(informative & context_tokens) / max(1, len(informative))
        if overlap >= 0.35:
            supported += 1
    return supported / len(sentences)


def _gold_answer_similarity(answer: str, gold_answer: str) -> float:
    if not answer.strip() or not gold_answer.strip():
        return 0.0
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform([answer, gold_answer])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])
    except ValueError:
        return 0.0


def evaluate_citation_accuracy(answer: str, target_sections: List[str]) -> float:
    if not target_sections:
        return 1.0
    normalized = _normalize_provision(answer)
    matched = sum(1 for target in target_sections if _normalize_provision(target) in normalized)
    return matched / len(target_sections)


def evaluate_faithfulness_and_relevance(
    question: str,
    answer: str,
    retrieved_chunks: List[Dict],
    gold_answer: Optional[str] = None,
    use_llm_judge: bool = False,
) -> Tuple[float, float]:
    """Return stable local scores; optionally replace them with an LLM judge."""
    if use_llm_judge:
        context = "\n---\n".join(c.get("text", "")[:600] for c in retrieved_chunks[:3])
        prompt = f"""You are an evaluation judge for an Indian child-law RAG system.
Question: {question}
Context:
{context}
Answer:
{answer[:1600]}

Return JSON only with numeric fields 0 to 1:
{{"faithfulness": 0.0, "relevance": 0.0}}"""
        try:
            raw = call_llm(prompt)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                f = float(data["faithfulness"])
                r = float(data["relevance"])
                return max(0.0, min(1.0, f)), max(0.0, min(1.0, r))
        except Exception as exc:
            print(f"  [Eval judge unavailable] {exc}")

    faithfulness = _context_support_score(answer, retrieved_chunks)
    relevance = _gold_answer_similarity(answer, gold_answer) if gold_answer else 0.0
    return faithfulness, relevance


def export_publication_tables(summary_metrics: Dict[str, Dict[str, float]]) -> None:
    md = [
        "# Empirical Retrieval & Generation Benchmark Results",
        "",
        "| Retrieval Strategy | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | MRR@5 | Context Support | Gold Similarity | Citation Acc. (%) | Avg Retrieval Latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, m in summary_metrics.items():
        md.append(
            f"| **{method}** | {m['hit_1']*100:.1f}% | {m['hit_3']*100:.1f}% | {m['hit_5']*100:.1f}% | "
            f"{m['mrr']:.3f} | {m['faithfulness']:.3f} | {m['relevance']:.3f} | "
            f"{m['citation_acc']*100:.1f}% | {m['latency']:.2f} |"
        )
    (OUTPUTS_DIR / "benchmark_comparison_table.md").write_text("\n".join(md), encoding="utf-8")

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{ChildLaw-QA retrieval and generation benchmark.}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\hline",
        r"Method & Hit@1 & Hit@3 & Hit@5 & MRR & Support & Gold Sim. & Citation & Latency \\",
        r"\hline",
    ]
    for method, m in summary_metrics.items():
        row = (
            f"{method} & {m['hit_1']*100:.1f}\\% & {m['hit_3']*100:.1f}\\% & "
            f"{m['hit_5']*100:.1f}\\% & {m['mrr']:.3f} & {m['faithfulness']:.3f} & "
            f"{m['relevance']:.3f} & {m['citation_acc']*100:.1f}\\% & {m['latency']:.2f}"
        )
        tex.append(row)
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUTPUTS_DIR / "benchmark_comparison_table.tex").write_text("\n".join(tex), encoding="utf-8")


def generate_benchmark_plots(summary_metrics: Dict[str, Dict[str, float]]) -> str:
    methods = list(summary_metrics)
    fig, ax = plt.subplots(figsize=(11, 6))
    values = [summary_metrics[m]["hit_3"] * 100 for m in methods]
    ax.bar(methods, values)
    ax.set_title("Hit@3 Retrieval Accuracy")
    ax.set_ylabel("Hit@3 (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    path = OUTPUTS_DIR / "evaluation_comparison_charts.png"
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return str(path)


def run_comprehensive_benchmark(
    chunks: List[Dict],
    max_queries: int = 30,
    generation_samples: int = 2,
    use_llm_judge: bool = False,
) -> Dict:
    benchmark_file = OUTPUTS_DIR / "benchmark_dataset.json"
    if not benchmark_file.exists():
        raise FileNotFoundError(f"Missing benchmark dataset: {benchmark_file}")

    cases = json.loads(benchmark_file.read_text(encoding="utf-8"))
    in_domain = [c for c in cases if c.get("category") != "NEGATIVE_CONTROL_OUT_OF_DOMAIN"][:max_queries]
    negative = [c for c in cases if c.get("category") == "NEGATIVE_CONTROL_OUT_OF_DOMAIN"]

    guardrail_ok = 0
    for case in negative:
        valid, _, _ = validate_child_law_domain(case["question"])
        guardrail_ok += int(not valid)
    guardrail_accuracy = guardrail_ok / len(negative) if negative else 1.0

    methods = {
        "BM25 (Lexical)": retrieve_bm25,
        "Dense (Ollama)": retrieve_dense,
        "Hybrid (RRF)": retrieve_hybrid_rrf,
        "Hybrid + Local Rerank": retrieve_reranked,
        "Hybrid Corrective": retrieve_crag_hyde,
    }

    summary = {}
    details = {}

    for method_name, method_fn in methods.items():
        print(f"\nEvaluating: {method_name}")
        values = {k: [] for k in ["hit_1", "hit_3", "hit_5", "mrr", "faithfulness", "relevance", "citation_acc", "latency"]}
        method_details = []

        for index, case in enumerate(in_domain, 1):
            start = time.perf_counter()
            retrieved = method_fn(case["question"], chunks, top_k=5)
            latency = time.perf_counter() - start
            ir = evaluate_retrieval_metrics(
                case.get("target_sections", []), retrieved, top_k=5, target_act=case.get("target_act")
            )
            for key in ["hit_1", "hit_3", "hit_5", "mrr"]:
                values[key].append(ir[key])
            values["latency"].append(latency)

            generation = None
            if index <= generation_samples:
                generation = generate_answer(case["question"], retrieved)
                if generation.startswith("Error generating grounded legal answer:"):
                    faith, rel = 0.0, 0.0
                else:
                    faith, rel = evaluate_faithfulness_and_relevance(
                        case["question"], generation, retrieved,
                        gold_answer=case.get("gold_answer"), use_llm_judge=use_llm_judge,
                    )
                values["faithfulness"].append(faith)
                values["relevance"].append(rel)
                values["citation_acc"].append(
                    evaluate_citation_accuracy(generation, case.get("target_sections", []))
                )

            method_details.append({
                "id": case["id"],
                "retrieved_sections": [c.get("section_hint") for c in retrieved],
                "metrics": ir,
                "latency": latency,
            })
            print(f"  [{index}/{len(in_domain)}] {case['id']}: Hit@3={ir['hit_3']:.0f} MRR={ir['mrr']:.2f} {latency:.2f}s")

        summary[method_name] = {
            "hit_1": float(np.mean(values["hit_1"])),
            "hit_3": float(np.mean(values["hit_3"])),
            "hit_5": float(np.mean(values["hit_5"])),
            "mrr": float(np.mean(values["mrr"])),
            # No fabricated fallback scores. Missing generation samples are explicit.
            "faithfulness": float(np.mean(values["faithfulness"])) if values["faithfulness"] else None,
            "relevance": float(np.mean(values["relevance"])) if values["relevance"] else None,
            "citation_acc": float(np.mean(values["citation_acc"])) if values["citation_acc"] else None,
            "latency": float(np.mean(values["latency"])),
        }
        details[method_name] = method_details

    result = {
        "benchmark_version": "local_v2",
        "embedding_model": __import__("config").EMBEDDING_MODEL,
        "llm_model": __import__("config").LLM_MODEL,
        "generation_samples_per_method": generation_samples,
        "llm_judge_enabled": use_llm_judge,
        "guardrail_accuracy": guardrail_accuracy,
        "retrieval_benchmarks": summary,
        "case_details": details,
    }
    (OUTPUTS_DIR / "evaluation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    export_publication_tables(summary)
    generate_benchmark_plots(summary)
    print("\nBenchmark complete. Results written to outputs/evaluation_summary.json")
    return summary


def run_evaluation(chunks: List[Dict]) -> None:
    run_comprehensive_benchmark(chunks)
