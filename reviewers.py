#!/usr/bin/env python

import os, re, argparse, textwrap, json, requests
from typing import Dict, Tuple, Optional
from uagents import Agent, Context, Model, Bureau
from agent_pdf import PDFPaperAnalyzer
import google.generativeai as genai


MAX_TURNS          = 25
MAX_REASONING_LEN  = 2_000
GEN_MODEL          = "models/gemini-2.0-flash" 


class ReviewMessage(Model):
    turn:      int
    rating:    int
    decision:  str         
    reasoning: str
    done:      bool = False



def derive_rating(report: str) -> int:
    """Convert a /10 or 'overall quality rating' into 1-5 (default 3)."""
    m = re.search(r'\b([0-9]+(?:\.[0-9]*)?)\s*/\s*10\b', report)
    if not m:
        m = re.search(r'overall quality rating[^0-9]*([0-9]+(?:\.[0-9]*)?)',
                      report, flags=re.I)
    score10 = float(m.group(1)) if m else 5.0
    return 1 if score10 <= 2 else 2 if score10 <= 4 else \
           3 if score10 <= 6 else 4 if score10 <= 8 else 5


def decide(r: int) -> str:
    return "accept" if r >= 3 else "deny"


def configure_gemini(key: str):
    genai.configure(api_key=key)


def gemini_http(prompt: str, key: str) -> Optional[str]:
    url = ("https://generativelanguage.googleapis.com/v1beta/"
           "models/gemini-2.0-pro:generateContent?key=" + key)
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, headers={"Content-Type": "application/json"},
                          data=json.dumps(payload), timeout=30)
        r.raise_for_status()
        data = r.json()
        return "".join(part.get("text", "")
                       for cand in data.get("candidates", [])
                       for part in cand.get("content", {}).get("parts", []))
    except Exception:
        return None


def reevaluate(exp: str, my_report: str, peer_text: str, key: str
               ) -> Optional[Tuple[int, str, str]]:
    """Ask Gemini to rescore given combined peer rationales."""
    prompt = textwrap.dedent(f"""
        You are an academic reviewer specialising in {exp}.
        ============  YOUR ORIGINAL ANALYSIS  ============
        {my_report[:4000]}
        ============  PEERS' LATEST REASONING  ===========
        {peer_text[:4000]}
        --------------------------------------------------
        • Reassess the paper given these perspectives.
        • Output exactly one line at the top:
              Rating: X/5   Decision: accept|deny
          where X is 1-5.
        • Then supply up to 200 words of justification.
    """)
    try:
        resp = genai.generate_content(
            model=GEN_MODEL,
            contents=[{"parts": [{"text": prompt}]}]
        )
        txt = "".join(p.text for c in resp.candidates for p in c.content.parts)
    except Exception:
        txt = gemini_http(prompt, key)
        if txt is None:
            return None

    m = re.search(r'Rating:\s*([1-5])\s*/\s*5.*?Decision:\s*(accept|deny)',
                  txt, flags=re.I)
    if not m:
        return None
    rating   = int(m.group(1))
    decision = m.group(2).lower()
    return rating, decision, txt.strip()[:MAX_REASONING_LEN]


def unanimous(decisions: Dict[str, str]) -> Optional[str]:
    """Return 'accept' / 'deny' if *all three* decisions match, else None."""
    if len(decisions) != 3:
        return None
    vals = set(decisions.values())
    return vals.pop() if len(vals) == 1 else None


parser = argparse.ArgumentParser(description="Triple-review round-robin")
parser.add_argument("--pdf", required=True, help="Path to PDF file")
args = parser.parse_args()

key = os.getenv("GEMINI_API_KEY")
if not key:
    raise RuntimeError("export GEMINI_API_KEY first")
configure_gemini(key)

print(" analysing paper …")

nlp_expertise = '''
You are a senior research scientist whose expertise lies at the intersection of statistics, artificial intelligence 
(AI), and machine learning (ML). Your work focuses on integrating statistical modeling with scalable, model-free AI/ML 
techniques to capture and leverage data dependence—spatial, temporal, or structural—for more accurate and interpretable
 predictions. You specialize in developing methods that explicitly model dependence structures in high-dimensional data, 
 enabling robust inference and learning in domains where classical independence assumptions fail. Your research spans 
 applications in environmental science, biomedical informatics, and data privacy, including spatially-aware random forests, 
 nearest-neighbor Gaussian processes, and scalable spatial probit models. You collaborate closely with domain
 scientists to translate methodological advances into real-world scientific impact.
'''

ml_expertise = '''
You are a senior research scientist whose expertise lies at the intersection of causal inference, reinforcement learning, 
and graphical models, with a particular focus on their theoretical and methodological interconnections. Your work centers on 
developing statistically rigorous, interpretable, and personalized decision-making frameworks using complex, high-dimensional, 
and often incomplete real-world data. You specialize in individualized treatment rule estimation, policy evaluation in online 
learning environments, and causal discovery for mediation analysis in heterogeneous populations. Methodologically, your 
research advances include doubly robust estimation techniques, jump Q-learning for continuous treatments, and the recovery 
of necessary and sufficient causal graphs under minimal assumptions. Your work is motivated by impactful applications in 
precision medicine, modern epidemiology, mental health analytics, and personalized marketing. Notable contributions 
include "On Learning Necessary and Sufficient Causal Graphs" (NeurIPS 2023), "Doubly Robust Interval Estimation for Optimal 
Policy Evaluation in Online Learning" (JASA 2023), "Jump Q-Learning for Individualized Decision Making with Continuous 
Treatments" (JMLR 2023), and "Towards Trustworthy Explanation: On Causal Rationalization" (ICML 2023). You aim to bridge 
the gap between statistical theory and real-world AI applications by ensuring interpretability, robustness, and fairness 
in data-driven decision systems.
'''

ds_expertise = '''
You are a senior research scientist whose expertise lies in statistical machine learning, high-dimensional data 
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
   Association, Theory and Methods.
'''

nlp_rep = PDFPaperAnalyzer(args.pdf, key,
                           expertise=nlp_expertise).run()
ml_rep  = PDFPaperAnalyzer(args.pdf, key,
                           expertise=ml_expertise).run()
ds_rep  = PDFPaperAnalyzer(args.pdf, key,
                           expertise=ds_expertise).run()


def init_state(report: str) -> Dict:
    rating = derive_rating(report)
    return {
        "rating":     rating,
        "decision":   decide(rating),
        "report":     report,
        "peers":      {},     
        "done":       False
    }

nlp_state = init_state(nlp_rep)
ml_state  = init_state(ml_rep)
ds_state  = init_state(ds_rep)


NLP_PORT, ML_PORT, DS_PORT = 8001, 8000, 8002

AgentNLP = Agent("NLPReviewer", seed="NLP secret",
                 port=NLP_PORT, endpoint=[f"http://127.0.0.1:{NLP_PORT}/submit"])
AgentML  = Agent("MLReviewer",  seed="ML secret",
                 port=ML_PORT,  endpoint=[f"http://127.0.0.1:{ML_PORT}/submit"])
AgentDS  = Agent("DSReviewer",  seed="DS secret",
                 port=DS_PORT,  endpoint=[f"http://127.0.0.1:{DS_PORT}/submit"])

ORDER = [AgentML.address, AgentNLP.address, AgentDS.address]
NEXT  = {ORDER[i]: ORDER[(i + 1) % 3] for i in range(3)}

STATES = {
    AgentML.address: ml_state,
    AgentNLP.address: nlp_state,
    AgentDS.address: ds_state,
}
EXPERTISE = {
    AgentML.address: ml_expertise,
    AgentNLP.address: nlp_expertise,
    AgentDS.address: ds_expertise,
}

def attach(agent: Agent):
    @agent.on_message(model=ReviewMessage)
    async def _(ctx: Context, sender: str, msg: ReviewMessage):
        st = STATES[agent.address]

   
        if st["done"]:
            return

    
        if msg.done:
            st["done"] = True
            ctx.logger.info(f"[{agent.name}] conversation done – stopping.")
            return

        ctx.logger.info(f"[{agent.name}] turn {msg.turn} from {sender[-8:]} – {msg.decision}")

      
        st["peers"][sender] = {"decision": msg.decision,
                               "reasoning": msg.reasoning}

       
        snapshot = {agent.address: st["decision"]}
        snapshot.update({a: v["decision"] for a, v in st["peers"].items()})
        verdict = unanimous(snapshot)
        if verdict:
            final = ReviewMessage(
                turn=msg.turn + 1,
                rating=st["rating"],
                decision=verdict,
                reasoning=f"Unanimous {verdict}.",
                done=True
            )
            for peer in [a for a in ORDER if a != agent.address]:
                await ctx.send(peer, final)
            st["done"] = True
            return

     
        if msg.turn >= MAX_TURNS:
            deny = ReviewMessage(
                turn=msg.turn + 1,
                rating=st["rating"],
                decision="deny",
                reasoning=f"No consensus after {MAX_TURNS} turns – auto-deny.",
                done=True
            )
            for peer in [a for a in ORDER if a != agent.address]:
                await ctx.send(peer, deny)
            st["done"] = True
            return

        
        if len(st["peers"]) == 2:
            combined = "\n\n".join(
                f"=== From {a[-8:]} ===\n{v['reasoning']}"
                for a, v in st["peers"].items()
            )
            new = reevaluate(
                EXPERTISE[agent.address],
                st["report"],
                combined,
                key
            )
            if new:
                st["rating"], st["decision"], new_reason = new
            else:
                new_reason = "Could not reevaluate – keeping stance."
        else:
       
            new_reason = "Awaiting all peer views."

      
        next_turn = msg.turn + 1
        out = ReviewMessage(
            turn=next_turn,
            rating=st["rating"],
            decision=st["decision"],
            reasoning=new_reason[:MAX_REASONING_LEN]
        )
        next_peer  = NEXT[agent.address]
        other_peer = [p for p in ORDER if p not in (agent.address, next_peer)][0]
        await ctx.send(next_peer,  out)   
        await ctx.send(other_peer, out)   

    return _

attach(AgentML)
attach(AgentNLP)
attach(AgentDS)


@AgentML.on_event("startup")
async def start(ctx: Context):
    first = ReviewMessage(
        turn=1,
        rating=ml_state["rating"],
        decision=ml_state["decision"],
        reasoning=ml_state["report"][:MAX_REASONING_LEN]
    )

    await ctx.send(NEXT[AgentML.address], first)             
    other = [p for p in ORDER if p not in (AgentML.address, NEXT[AgentML.address])][0]
    await ctx.send(other, first)                             


bureau = Bureau()
bureau.add(AgentML)
bureau.add(AgentNLP)
bureau.add(AgentDS)

if __name__ == "__main__":
    bureau.run()
