# evaluation.py
"""
Functions for evaluating retrieval and answer quality
"""

import json
import time
import re
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import client, LLM_MODEL, OUTPUTS_DIR


def evaluate_answer_relevance(question: str, answer: str, retrieved_chunks: List[Dict]) -> float:
    """
    Use LLM to evaluate if answer is relevant to question and sources
    
    Args:
        question: The question asked
        answer: The generated answer
        retrieved_chunks: The chunks used to generate the answer
        
    Returns:
        Relevance score from 0.0 to 1.0
    """
    
    source_text = "\n".join([f"- {c['file']} chunk {c['chunk_id']}: {c['text'][:200]}..." for c in retrieved_chunks[:3]])
    
    eval_prompt = f"""Rate how well this answer addresses the question using the provided sources.
Rate from 0.0 to 1.0:
- 0.0-0.3: Answer doesn't address question or contradicts sources
- 0.3-0.6: Partial answer, missing key information
- 0.6-0.8: Good answer, covers main points
- 0.8-1.0: Excellent answer, comprehensive and well-sourced

Question: {question}

Answer: {answer[:500]}

Sources:
{source_text}

Respond ONLY with a single decimal number between 0.0 and 1.0"""
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": eval_prompt}]
        )
        score_text = response.choices[0].message.content.strip()
        # Extract first valid number from response
        match = re.search(r'0\.\d+|1\.0', score_text)
        if match:
            return float(match.group())
    except Exception as e:
        print(f"Error in relevance evaluation: {e}")
    
    return 0.5  # Default middle score on error


def generate_evaluation_graphs(results: List[Dict]) -> str:
    """
    Generate informative graphs with relevance scores, latency, and coverage
    
    Args:
        results: List of evaluation results
        
    Returns:
        Path to saved graph image
    """
    output_dir = OUTPUTS_DIR
    output_dir.mkdir(exist_ok=True)
    
    # Prepare data
    methods = {}
    for result in results:
        m = result["method"]
        if m not in methods:
            methods[m] = {"latencies": [], "sources": [], "relevances": []}
        methods[m]["latencies"].append(result["latency"])
        methods[m]["sources"].append(len(result["sources"]))
        if "relevance" in result:
            methods[m]["relevances"].append(result["relevance"])
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    method_names = list(methods.keys())
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"][:len(method_names)]
    
    # Plot 1: Average latency by method
    ax = axes[0]
    avg_latencies = [np.mean(methods[m]["latencies"]) for m in method_names]
    bars = ax.bar(method_names, avg_latencies, color=colors)
    ax.set_ylabel("Latency (seconds)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Retrieval Method", fontsize=11)
    ax.set_title("Average Latency Comparison", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, (bar, v) in enumerate(zip(bars, avg_latencies)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}s", ha="center", fontsize=9)
    
    # Plot 2: Average source coverage
    ax = axes[1]
    avg_sources = [np.mean(methods[m]["sources"]) for m in method_names]
    bars = ax.bar(method_names, avg_sources, color=colors)
    ax.set_ylabel("Number of Sources", fontsize=11, fontweight="bold")
    ax.set_xlabel("Retrieval Method", fontsize=11)
    ax.set_title("Average Source Coverage", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, (bar, v) in enumerate(zip(bars, avg_sources)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    
    # Plot 3: Average relevance score (most important!)
    ax = axes[2]
    avg_relevances = [np.mean(methods[m]["relevances"]) if methods[m]["relevances"] else 0.0 for m in method_names]
    bars = ax.bar(method_names, avg_relevances, color=colors)
    ax.set_ylabel("Relevance Score (0-1)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Retrieval Method", fontsize=11)
    ax.set_title("Average Answer Relevance", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1.0])
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Good (>0.8)")
    ax.axhline(y=0.6, color="orange", linestyle="--", alpha=0.5, label="Fair (0.6-0.8)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    for i, (bar, v) in enumerate(zip(bars, avg_relevances)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    
    plt.tight_layout()
    graph_path = output_dir / "evaluation_graphs.png"
    plt.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved evaluation graphs to {graph_path}")
    return str(graph_path)


def generate_relevance_metrics_table(results: List[Dict]) -> str:
    """
    Generate evaluation table with relevance scores, latency, and coverage metrics
    
    Args:
        results: List of evaluation results
        
    Returns:
        HTML table string
    """
    output_dir = OUTPUTS_DIR
    output_dir.mkdir(exist_ok=True)
    
    # Aggregate metrics by method
    methods = list(set(r["method"] for r in results))
    method_metrics = {m: {"relevance": [], "latency": [], "coverage": []} for m in methods}
    
    for result in results:
        method = result["method"]
        method_metrics[method]["latency"].append(result["latency"])
        method_metrics[method]["coverage"].append(len(result["sources"]))
        if "relevance" in result:
            method_metrics[method]["relevance"].append(result["relevance"])
    
    # Generate HTML table with proper metrics
    html_table = "<table border='1' cellpadding='10' cellspacing='0' style='border-collapse:collapse'>\n"
    html_table += "<tr style='background-color:#e0e0e0;'>"
    html_table += "<th>Retrieval Method</th>"
    html_table += "<th>Avg Relevance<br/>(0.0-1.0)</th>"
    html_table += "<th>Avg Latency<br/>(seconds)</th>"
    html_table += "<th>Avg Sources<br/>Retrieved</th>"
    html_table += "<th>Queries<br/>Evaluated</th>"
    html_table += "</tr>\n"
    
    for method in sorted(methods):
        metrics = method_metrics[method]
        avg_relevance = np.mean(metrics["relevance"]) if metrics["relevance"] else 0.0
        avg_latency = np.mean(metrics["latency"])
        avg_coverage = np.mean(metrics["coverage"])
        num_evals = len(metrics["latency"])
        
        # Color code relevance: red < 0.6, yellow 0.6-0.8, green > 0.8
        if avg_relevance < 0.6:
            color = "#ffcccc"  # Light red
        elif avg_relevance < 0.8:
            color = "#ffffcc"  # Light yellow
        else:
            color = "#ccffcc"  # Light green
        
        html_table += f"<tr>"
        html_table += f"<td><b>{method.capitalize()}</b></td>"
        html_table += f"<td style='background-color:{color};'><b>{avg_relevance:.3f}</b></td>"
        html_table += f"<td>{avg_latency:.3f}s</td>"
        html_table += f"<td>{avg_coverage:.2f}</td>"
        html_table += f"<td>{num_evals}</td>"
        html_table += f"</tr>\n"
    
    html_table += "</table>"
    
    # Save as HTML file
    table_path = output_dir / "relevance_metrics.html"
    with open(table_path, "w") as f:
        f.write(html_table)
    
    print(f"Saved relevance metrics table to {table_path}")
    return html_table


def run_evaluation(chunks: List[Dict]) -> None:
    """
    Run comprehensive evaluation of all retrieval methods
    
    Args:
        chunks: List of all chunks with embeddings
    """
    # Import here to avoid circular imports
    from retrieval import retrieve_dense, retrieve_hybrid, retrieve_multiquery, retrieve_corrective
    from answer_generation import generate_answer
    
    test_questions = [
        "What is the punishment under the POCSO Act?",
        "What rights do children have under the Constitution?",
        "What are the child labour laws in Tamil Nadu?",
        "How can child abuse be reported in Kerala?",
        "What is the Juvenile Justice Act?"
    ]

    methods = {
        "dense": retrieve_dense,
        "hybrid": retrieve_hybrid,
        "multiquery": retrieve_multiquery,
        "corrective": retrieve_corrective
    }

    results = []
    
    print("\n" + "="*60)
    print("Running evaluation with LLM-based relevance scoring...")
    print("This may take 1-2 minutes. Processing each method's answers...")
    print("="*60 + "\n")

    for method_name, method_fn in methods.items():
        print(f"Evaluating {method_name.capitalize()} method...\n")
        
        for question in test_questions:
            start = time.time()
            retrieved = method_fn(question, chunks)
            latency = round(time.time() - start, 2)

            sources = [
                f"{r['file']}#{r['chunk_id']}" for r in retrieved
            ]
            
            # Generate answer for relevance evaluation
            answer = generate_answer(question, retrieved)
            
            # Evaluate relevance using LLM
            relevance_score = evaluate_answer_relevance(question, answer, retrieved)

            results.append({
                "method": method_name,
                "question": question,
                "latency": latency,
                "sources": sources,
                "relevance": relevance_score,
                "answer_preview": answer[:100]
            })
            
            print(f"  Q: {question[:50]}... → Relevance: {relevance_score:.3f}")

    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Save evaluation results
    results_path = OUTPUTS_DIR / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved evaluation results to {results_path}")
    
    # Generate graphs
    print("✓ Generating evaluation graphs...")
    graphs_path = generate_evaluation_graphs(results)
    
    # Generate relevance metrics table
    print("✓ Generating relevance metrics table...")
    metrics_table = generate_relevance_metrics_table(results)
    
    # Save metrics table to file
    metrics_path = OUTPUTS_DIR / "relevance_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_table)
    
    print(f"✓ Saved metrics table to {OUTPUTS_DIR / 'relevance_metrics.html'}\n")
    
    print("="*60)
    print("Evaluation Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  • outputs/evaluation_graphs.png - Visual performance comparison")
    print("  • outputs/relevance_metrics.html - Relevance & performance metrics")
    print("  • outputs/evaluation_results.json - Detailed results")
    print("="*60)
