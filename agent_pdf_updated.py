

import argparse
import json
import os
import pathlib

import google.generativeai as genai
from google.generativeai import types

MODEL_NAME = "gemini-2.0-flash"
TEMPERATURE = 0.7

CRITERIA = """
i) Novelty:  The novelty and innovativeness of contributed solutions, problem formulations, methodologies, theories, and/or evaluations, i.e., the extent to which the paper is sufficiently original with respect to the state-of-the-art.
ii) Rigor: The soundness, clarity, and depth of a technical or theoretical contribution, and the level of thoroughness and completeness of an evaluation.
iii) Relevance: The significance and/or potential impact of the research on the field of software engineering.
iv) Verifiability & Transparency: The extent to which the paper includes sufficient information to understand how an innovation works; to understand how data was obtained, analyzed, and interpreted; and how the paper supports independent verification or replication of the paper’s claimed contributions. Any artifacts attached to or linked from the paper will be checked by one reviewer.
v) Presentation: The clarity of the exposition in the paper.
"""

EXPERTISE = '''statistical machine learning, high-dimensional data 
            analysis, and integrative methods for heterogeneous and unstructured data. Your work focuses on developing 
            cutting-edge statistical theory and algorithms to extract essential signals from large-scale, high-dimensional, 
            and multimodal datasets. You specialize in areas such as natural language processing, recommender systems, tensor imaging, 
            network analysis, and dynamic treatment modeling. You are particularly interested in statistical perspectives on deep 
            learning, high-dimensional mediation analysis, causal inference using de-confounding, and reinforcement learning for precision
            decision-making. Your recent work integrates data from wearable devices for mobile health monitoring, addresses privacy 
            through differential privacy methods, and applies statistical learning to genomics, PTSD prediction via DNA methylation, and 
            social and political science text data. Your research is supported by multiple NSF and NIH grants, including studies on generative
            models for NLP, individualized learning for multimodal wearable data, and integrative learning for longitudinal mobile health 
            data. You are a Fellow of the ASA, IMS, and AAAS, and currently serve as Co-Editor of the Journal of the American Statistical 
            Association, Theory and Methods.'''

SYSTEM_PROMPT = f"""
You are a senior research scientist whose expertise is {EXPERTISE}.

Review the attached PDF and return **exact JSON**:
{{
  "novelty":       {{"score": <int>, "justification": "<text>"}},
  "rigor":         {{"score": <int>, "justification": "<text>"}},
  "relevance":     {{"score": <int>, "justification": "<text>"}},
  "verifiability": {{"score": <int>, "justification": "<text>"}},
  "presentation":  {{"score": <int>, "justification": "<text>"}},
  "overall":       {{"score": <int>, "comments": "<text>"}}
}}
"""

def build_prompt() -> str:
    return SYSTEM_PROMPT + "\n\nCriteria:\n" + CRITERIA

def call_gemini(pdf_path: pathlib.Path, prompt: str) -> str:

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])


    pdf_part = types.Part.from_bytes(
        data=pdf_path.read_bytes(),
        mime_type="application/pdf",
    )

    response = genai.generate_content(
        model="models/gemini-2.0-flash",
        contents=[pdf_part, prompt],
        generation_config=types.GenerationConfig(temperature=TEMPERATURE),
    )

    return response.text 

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a PDF with Gemini-2.0-flash."
    )
    parser.add_argument("pdf", type=pathlib.Path, help="Path to the PDF file")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    prompt = build_prompt()
    raw    = call_gemini(args.pdf, prompt)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise SystemExit("Gemini did not return valid JSON:\n" + raw)

    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set the GOOGLE_API_KEY environment variable.")
    main()