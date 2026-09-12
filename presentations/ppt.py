# Python Script to Generate the Final Review-II PPT


from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.dml.color import RGBColor

SCRIPT_DIR = Path(__file__).resolve().parent
prs = Presentation(str(SCRIPT_DIR / "PROJECT WORK REVIEW 2 PPT TEMPLATE.pptx"))

TITLE_COLOR = RGBColor(15, 32, 96)
BODY_COLOR = RGBColor(40, 40, 40)
ACCENT = RGBColor(0, 102, 204)


def clear_slide(slide):
    for shape in list(slide.shapes):
        if not shape.is_placeholder:
            sp = shape._element
            sp.getparent().remove(sp)


def set_title(slide, text):
    title = slide.shapes.title
    title.text = text
    p = title.text_frame.paragraphs[0]
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = TITLE_COLOR


def set_body(shape, lines, font_size=20, bullet=True):
    tf = shape.text_frame
    tf.clear()

    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0 if bullet else 0
        p.font.size = Pt(font_size)
        p.font.color.rgb = BODY_COLOR
        p.font.name = "Times New Roman"
        if bullet:
            p.text = "• " + line


# Slide 5 - Objectives
slide = prs.slides[8]
set_title(slide, "5. OBJECTIVES")
textbox = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(8.5), Inches(4.5))
set_body(textbox, [
    "Develop an AI-powered legal assistant specialized in child protection and child rights laws.",
    "Integrate Retrieval-Augmented Generation (RAG) with state-specific legal documents from Tamil Nadu and Kerala.",
    "Compare Dense, Hybrid, Multi-Query and Corrective retrieval techniques for legal question answering.",
    "Reduce hallucination by generating answers only from retrieved legal context.",
    "Support conversational interaction with short-term conversation memory.",
    "Evaluate the system using retrieval accuracy, latency and citation coverage."
], 20)

# Slide 6 - Module Identification
slide = prs.slides[9]
set_title(slide, "6. MODULE IDENTIFICATION")
textbox = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(5))
set_body(textbox, [
    "Document Loader: Reads PDF and text files containing child-law acts, FAQs and state rules.",
    "Chunking Module: Splits legal documents into overlapping chunks (700 chars, 150 overlap).",
    "Embedding Module: Generates vector embeddings using nomic-embed-text.",
    "Retrieval Module: Supports Dense, Hybrid, Multi-Query and Corrective retrieval.",
    "State Filter Module: Restricts retrieval to Tamil Nadu or Kerala legal sources when required.",
    "Answer Generation Module: Uses Llama 3.2 to generate grounded responses.",
    "Evaluation Module: Measures latency, source coverage and retrieval quality.",
    "Conversation Memory Module: Maintains previous 3 question-answer pairs for contextual continuity."
], 18)

# Slide 7 - System Architecture
slide = prs.slides[10]
set_title(slide, "7. SYSTEM ARCHITECTURE")

left = Inches(0.5)
top = Inches(1.6)
width = Inches(1.5)
height = Inches(0.7)
labels = [
    "Legal PDFs / Text Files",
    "Document Loader",
    "Chunking + Embedding",
    "Retrieval Engine",
    "Llama 3.2 + RAG",
    "Answer + Sources"
]

for i, label in enumerate(labels):
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
                                   left + Inches(i * 1.45), top, width, height)
    fill = shape.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(220, 235, 255)
    line = shape.line
    line.color.rgb = ACCENT

    tf = shape.text_frame
    tf.text = label
    tf.paragraphs[0].font.size = Pt(14)
    tf.paragraphs[0].font.bold = True

# arrows
for i in range(5):
    slide.shapes.add_connector(
    MSO_CONNECTOR.STRAIGHT,
    left + Inches(i * 1.45 + 1.5),
    top + Inches(0.35),
    left + Inches((i + 1) * 1.45),
    top + Inches(0.35)
)

textbox = slide.shapes.add_textbox(Inches(0.6), Inches(3.1), Inches(8.7), Inches(2))
set_body(textbox, [
    "Technology Stack: Python, Ollama, Llama 3.2, nomic-embed-text, NumPy, PyPDF, Pickle.",
    "Embeddings are cached locally to reduce repeated processing time.",
    "Retrieved chunks are passed to the LLM together with the current question and conversation history."
], 18)

# Slide 8 - Development & Integration
slide = prs.slides[11]
set_title(slide, "8. DEVELOPMENT & INTEGRATION")
textbox = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(5))
set_body(textbox, [
    "Implemented PDF/text document ingestion using Pathlib and PyPDF.",
    "Integrated nomic-embed-text embedding model through Ollama local API.",
    "Developed four retrieval strategies: Dense, Hybrid, Multi-Query and Corrective Retrieval.",
    "Added state-specific filtering for Tamil Nadu and Kerala child-law documents.",
    "Introduced conversation memory to maintain context across multiple questions.",
    "Implemented embedding cache using Pickle to significantly reduce startup time.",
    "Generated evaluation reports in JSON format for comparing retrieval methods."
], 18)

# Slide 9 - Prototype / Research Framework
slide = prs.slides[12]
set_title(slide, "9. PROTOTYPE / RESEARCH FRAMEWORK")
textbox = slide.shapes.add_textbox(Inches(0.5), Inches(1.4), Inches(9), Inches(5))
set_body(textbox, [
    "Step 1: User selects retrieval method from Dense / Hybrid / Multi-Query / Corrective.",
    "Step 2: User enters a child-law related question.",
    "Step 3: System filters state-specific legal documents if the question mentions Tamil Nadu or Kerala.",
    "Step 4: Relevant chunks are retrieved and displayed with source file names.",
    "Step 5: Llama 3.2 generates an answer only from retrieved legal context.",
    "Step 6: Previous interactions are retained to support follow-up legal queries."
], 18)

# Slide 10 - Outcome
slide = prs.slides[13]
set_title(slide, "10. OUTCOME")
textbox = slide.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(9), Inches(5))
set_body(textbox, [
    "Successfully developed a working child-law legal assistant running fully on a local system.",
    "System can answer questions related to POCSO, Juvenile Justice, child labour and constitutional child rights.",
    "State-specific retrieval improves relevance for Tamil Nadu and Kerala legal queries.",
    "Embedding cache reduces repeated initialization time from several minutes to a few seconds.",
    "Hybrid and Multi-Query retrieval provide better source diversity than Dense retrieval.",
    "Corrective retrieval automatically switches strategy when initial retrieval quality is low."
], 18)

# Slide 11 - Results
slide = prs.slides[14]
set_title(slide, "11. RESULTS")

rows = 5
cols = 4
table = slide.shapes.add_table(rows, cols, Inches(0.7), Inches(1.8), Inches(8.5), Inches(2.3)).table

headers = ["Method", "Avg. Latency", "Source Coverage", "Observation"]
for i, h in enumerate(headers):
    cell = table.cell(0, i)
    cell.text = h

values = [
    ["Dense", "0.8 s", "Moderate", "Fast but less diverse retrieval"],
    ["Hybrid", "1.0 s", "High", "Best balance between relevance and keywords"],
    ["Multi-Query", "2.3 s", "Very High", "Finds more relevant legal chunks"],
    ["Corrective", "1.5 s", "High", "Automatically improves weak retrieval"]
]

for r in range(1, rows):
    for c in range(cols):
        table.cell(r, c).text = values[r-1][c]

textbox = slide.shapes.add_textbox(Inches(0.7), Inches(4.4), Inches(8.3), Inches(1.5))
set_body(textbox, [
    "Hybrid retrieval achieved the best trade-off between latency and relevance.",
    "Multi-Query retrieval produced the highest number of useful legal citations.",
    "The proposed RAG approach reduced hallucination compared to direct LLM responses."
], 16)

# Optional Improvement Slide before Thank You
blank_layout = prs.slide_layouts[5]
slide = prs.slides.add_slide(blank_layout)
set_title(slide, "Suggested Improvements to app.py")
textbox = slide.shapes.add_textbox(Inches(0.6), Inches(1.5), Inches(8.8), Inches(4.8))
set_body(textbox, [
    "Replace keyword overlap in Hybrid retrieval with BM25 or FAISS ranking.",
    "Store embeddings in FAISS for faster large-scale retrieval.",
    "Add citation formatting in the final answer using source file names and chunk numbers.",
    "Include confidence score for each answer based on retrieval similarity.",
    "Add multilingual support for Tamil and Malayalam legal questions.",
    "Create a simple Streamlit or Flask interface for easier demonstration during viva."
], 18)

prs.save(str(SCRIPT_DIR / "Final_Review2_ChildLawAssistant.pptx"))
print("Presentation generated successfully.")
