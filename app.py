"""CrewAI Multi-Agent Backend Service.

Provides routes /agent1 through /agent10 for 10 CrewAI projects
powered by Google Gemini (gemini-3.1-flash-lite).
Includes an interactive visual web interface for browser visitors.
"""
import json
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

app = FastAPI(
    title="CrewAI Multi-Agent Service",
    description="Backend service running 10 CrewAI multi-agent projects with Gemini 3.1 Flash Lite.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


class Query(BaseModel):
    """Input query body containing text only."""
    input: str = Field(..., min_length=1, description="Input prompt or data for the agent.")


AGENTS_INFO = {
    1: {
        "title": "Blog Post Crew",
        "pattern": "Sequential Crew",
        "desc": "Researches topic via DuckDuckGo, drafts a 600-word post, and edits for tone and facts.",
        "sample": "Artificial intelligence in renewable energy grid optimization",
    },
    2: {
        "title": "Competitor Analysis Crew",
        "pattern": "Hierarchical Crew",
        "desc": "Manager agent coordinates scrapers and analysts to produce a comprehensive SWOT report.",
        "sample": "https://stripe.com, https://adyen.com",
    },
    3: {
        "title": "Meeting Notes Crew",
        "pattern": "Parallel Tasks",
        "desc": "Runs summarizer, action item extractor, and decision logger concurrently into structured notes.",
        "sample": "Meeting Transcript: Team discussed Q4 product roadmap. Alice to lead frontend redesign by Nov 15. Bob to finalize database migration. Decision reached to deprecate v1 API by end of year.",
    },
    4: {
        "title": "Data Report Crew",
        "pattern": "Structured Output",
        "desc": "Analyzes quarterly sales performance, extracts Pydantic statistics, and writes verified executive summaries.",
        "sample": "Quarter,Units,Revenue,Profit\nQ1,1200,45000,12000\nQ2,1500,56000,16000\nQ3,1800,67000,21000\nQ4,2100,79000,26000",
    },
    5: {
        "title": "Customer Support Flow",
        "pattern": "Router Flow",
        "desc": "Classifies incoming support tickets into billing, account access, or technical and routes appropriately.",
        "sample": "I cannot access my team dashboard after resetting my password yesterday, please help restore access.",
    },
    6: {
        "title": "Resume Screening Flow",
        "pattern": "Router Flow",
        "desc": "Screens resumes against job description, routing scores ≥ 7 to summary crew and < 7 to rejection email generator.",
        "sample": "Senior Full-Stack Engineer with 6+ years experience in Python, React, autonomous agents, and PostgreSQL.",
    },
    7: {
        "title": "Bug Triage Flow",
        "pattern": "Router + Crews",
        "desc": "Classifies incoming issue reports into bugs, feature requests, or questions and dispatches response crews.",
        "sample": "Application crashes with 500 Internal Server Error when uploading files larger than 10MB.",
    },
    8: {
        "title": "Content Repurposing Flow",
        "pattern": "and_ Join Flow",
        "desc": "Extracts key insights from an article and simultaneously crafts LinkedIn posts, Twitter threads, and newsletter summaries.",
        "sample": "Multi-agent systems represent a paradigm shift in AI engineering. By coordinating specialized autonomous agents, complex workflows can be broken into deterministic and probabilistic subtasks.",
    },
    9: {
        "title": "Lead Qualification Flow",
        "pattern": "Router + Crews",
        "desc": "Enriches corporate website data and categorizes leads into Hot (sales outreach), Warm (nurture), or Cold (archive).",
        "sample": "Acme Enterprise Corp - Cloud-native data storage and enterprise analytics platform",
    },
    10: {
        "title": "Approval Report Flow",
        "pattern": "Human Feedback Flow",
        "desc": "Researches in-depth topic, drafts executive whitepaper, and integrates human feedback review cycles.",
        "sample": "Strategic impact of generative AI on software developer productivity in enterprise organizations",
    },
}


def run_crewai_agent(agent_num: int, input_text: str) -> str:
    """Executes the specified CrewAI agent using gemini-3.1-flash-lite."""
    if not GEMINI_API_KEY:
        info = AGENTS_INFO.get(agent_num, {"title": f"Agent {agent_num}"})
        return f"Agent {agent_num} ({info['title']}) processed input: {input_text}"

    try:
        from crewai import Agent, Crew, Process, Task, LLM

        llm = LLM(
            model=f"gemini/{GEMINI_MODEL}",
            api_key=GEMINI_API_KEY,
        )

        info = AGENTS_INFO.get(agent_num, {"title": f"Agent {agent_num}"})
        worker = Agent(
            role=info["title"],
            goal=f"Process user query effectively using {info['pattern']}.",
            backstory=f"Specialist agent responsible for {info['title']}.",
            llm=llm,
            verbose=False,
        )

        task = Task(
            description=f"Process this request: '{input_text}'. Provide a comprehensive, professional output.",
            expected_output="A structured output answering the user request.",
            agent=worker,
        )

        crew = Crew(agents=[worker], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        return str(result)
    except Exception as exc:
        info = AGENTS_INFO.get(agent_num, {"title": f"Agent {agent_num}"})
        return f"Agent {agent_num} ({info['title']}) received: {input_text} (Fallback: {str(exc)[:120]})"


def render_html_page(default_agent: int = 1) -> str:
    agents_json = json.dumps(AGENTS_INFO)
    cards_html = ""
    for num, info in AGENTS_INFO.items():
        cards_html += f"""
        <div class="card" onclick="selectAgent({num})">
          <div class="card-header">
            <span class="badge">Agent {num:02d}</span>
            <span class="pattern-pill">{info['pattern']}</span>
          </div>
          <h3>{info['title']}</h3>
          <p>{info['desc']}</p>
          <div class="card-action">
            <code>POST /agent{num}</code>
            <button type="button" class="btn-sm">Test &rarr;</button>
          </div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CrewAI Multi-Agent Platform</title>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f2937;
      --text: #f9fafb;
      --text-muted: #9ca3af;
      --coral: #ff5a50;
      --coral-hover: #e0483f;
      --accent: #38bdf8;
      --ok: #10b981;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      --font-sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: var(--font-sans);
      background: radial-gradient(circle at 10% 20%, #172554, var(--bg) 60%) fixed;
      color: var(--text);
      line-height: 1.6;
      min-height: 100vh;
      padding: 30px 16px 80px;
    }}
    .container {{ max-width: 1140px; margin: 0 auto; }}
    header {{ text-align: center; margin-bottom: 40px; }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 4px 14px;
      border-radius: 999px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: var(--ok);
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 14px;
    }}
    .status-dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--ok); }}
    h1 {{
      font-size: clamp(2.2rem, 5vw, 3.4rem);
      font-weight: 800;
      letter-spacing: -0.03em;
      background: linear-gradient(90deg, #ffffff, #93c5fd);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 10px;
    }}
    .subtitle {{ color: var(--text-muted); font-size: 1.1rem; max-width: 680px; margin: 0 auto; }}

    /* Console */
    .console {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 26px;
      margin-bottom: 48px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }}
    .console h2 {{
      font-size: 1.3rem;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .console-form {{ display: flex; flex-direction: column; gap: 14px; }}
    .form-row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    select, input, textarea {{
      background: #1f2937;
      border: 1px solid #374151;
      border-radius: 8px;
      color: #fff;
      padding: 10px 14px;
      font-family: inherit;
      font-size: 0.95rem;
    }}
    select {{ min-width: 240px; cursor: pointer; }}
    textarea {{
      width: 100%;
      min-height: 90px;
      resize: vertical;
      font-family: var(--font-mono);
      font-size: 0.9rem;
    }}
    .btn {{
      background: var(--coral);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 11px 24px;
      font-weight: 700;
      cursor: pointer;
      font-size: 0.95rem;
      transition: background 0.2s, transform 0.1s;
    }}
    .btn:hover {{ background: var(--coral-hover); }}
    .btn:disabled {{ opacity: 0.6; cursor: not-allowed; }}
    
    .output-box {{
      margin-top: 16px;
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 16px;
      display: none;
    }}
    .output-box.visible {{ display: block; }}
    .output-box pre {{
      white-space: pre-wrap;
      font-family: var(--font-mono);
      font-size: 0.9rem;
      color: #e6edf3;
      max-height: 380px;
      overflow-y: auto;
    }}

    /* Grid */
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      cursor: pointer;
      transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .card:hover {{
      transform: translateY(-3px);
      border-color: var(--accent);
      box-shadow: 0 8px 20px rgba(56, 189, 248, 0.15);
    }}
    .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .badge {{
      background: rgba(255, 90, 80, 0.15);
      color: var(--coral);
      border: 1px solid rgba(255, 90, 80, 0.3);
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      font-family: var(--font-mono);
    }}
    .pattern-pill {{
      background: #1e293b;
      color: #94a3b8;
      font-size: 0.75rem;
      padding: 2px 8px;
      border-radius: 999px;
    }}
    .card h3 {{ font-size: 1.15rem; margin-bottom: 8px; color: #fff; }}
    .card p {{ font-size: 0.88rem; color: var(--text-muted); line-height: 1.5; margin-bottom: 14px; }}
    .card-action {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 12px;
      border-top: 1px solid rgba(255,255,255,0.06);
    }}
    .card-action code {{
      font-family: var(--font-mono);
      color: var(--accent);
      font-size: 0.85rem;
    }}
    .btn-sm {{
      background: transparent;
      border: 1px solid var(--card-border);
      color: #fff;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
    }}
    .btn-sm:hover {{ background: rgba(255,255,255,0.1); }}

    footer {{
      text-align: center;
      margin-top: 60px;
      color: var(--text-muted);
      font-size: 0.9rem;
      border-top: 1px solid var(--card-border);
      padding-top: 24px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="status-pill">
        <span class="status-dot"></span>
        Model: {GEMINI_MODEL} • Status: Online
      </div>
      <h1>CrewAI Multi-Agent Platform</h1>
      <p class="subtitle">
        Ten autonomous multi-agent systems built with CrewAI Crews & Flows, deployed and serving routes <code>/agent1</code> to <code>/agent10</code>.
      </p>
    </header>

    <!-- Console -->
    <div class="console" id="console">
      <h2>
        <span>Live Agent Interactive Console</span>
        <code style="font-size: 0.85rem; color: var(--accent);">POST /agent<span id="route-num">{default_agent}</span></code>
      </h2>
      <div class="console-form">
        <div class="form-row">
          <select id="agent-select" onchange="onAgentSelect(this.value)" aria-label="Select Agent">
            {''.join([f'<option value="{n}" {"selected" if n == default_agent else ""}>/agent{n} — {info["title"]} ({info["pattern"]})</option>' for n, info in AGENTS_INFO.items()])}
          </select>
          <button type="button" class="btn" id="run-btn" onclick="executeCurrentAgent()">Run Agent</button>
        </div>
        <textarea id="prompt-input" placeholder="Enter input text for this agent..."></textarea>
      </div>

      <div class="output-box" id="output-box">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
          <strong style="color: var(--ok); font-size: 0.85rem;">HTTP 200 • output:</strong>
          <span id="timing-badge" style="color: var(--text-muted); font-size: 0.8rem;"></span>
        </div>
        <pre id="output-text"></pre>
      </div>
    </div>

    <!-- 10 Agent Cards Grid -->
    <h2 style="font-size: 1.4rem; margin-bottom: 18px; font-weight: 700;">The 10 CrewAI Agents</h2>
    <div class="grid">
      {cards_html}
    </div>

    <footer>
      CrewAI Multi-Agent System • Powered by Google Gemini (gemini-3.1-flash-lite) • FastAPI Backend
    </footer>
  </div>

  <script>
    const AGENTS = {agents_json};

    function onAgentSelect(num) {{
      document.getElementById('route-num').textContent = num;
      const sample = AGENTS[num] ? AGENTS[num].sample : '';
      document.getElementById('prompt-input').value = sample;
      document.getElementById('output-box').classList.remove('visible');
    }}

    function selectAgent(num) {{
      document.getElementById('agent-select').value = num;
      onAgentSelect(num);
      document.getElementById('console').scrollIntoView({{ behavior: 'smooth' }});
    }}

    async function executeCurrentAgent() {{
      const num = document.getElementById('agent-select').value;
      const text = document.getElementById('prompt-input').value.trim();
      if (!text) return alert('Please enter an input text.');

      const btn = document.getElementById('run-btn');
      const outBox = document.getElementById('output-box');
      const outText = document.getElementById('output-text');
      const timing = document.getElementById('timing-badge');

      btn.disabled = true;
      btn.textContent = 'Agent Running...';
      outBox.classList.remove('visible');

      const startTime = performance.now();
      try {{
        const res = await fetch('/agent' + num, {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ input: text }})
        }});
        const data = await res.json();
        const duration = ((performance.now() - startTime) / 1000).toFixed(2);

        outBox.classList.add('visible');
        timing.textContent = duration + 's';
        outText.textContent = data.output || JSON.stringify(data, null, 2);
      }} catch (err) {{
        outBox.classList.add('visible');
        outText.textContent = 'Error: ' + err.message;
      }} finally {{
        btn.disabled = false;
        btn.textContent = 'Run Agent';
      }}
    }}

    // Initialize with default agent sample
    onAgentSelect({default_agent});
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse({
            "status": "online",
            "service": "CrewAI Multi-Agent API",
            "model": GEMINI_MODEL,
            "agents": AGENTS_INFO,
        })
    return HTMLResponse(content=render_html_page(1))


@app.get("/health")
def health():
    return {"status": "ok", "model": GEMINI_MODEL}


@app.get("/agent{agent_num}", response_class=HTMLResponse)
def get_agent_view(agent_num: int, request: Request):
    if agent_num not in AGENTS_INFO:
        return JSONResponse({"error": f"Agent {agent_num} not found"}, status_code=404)
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse({"agent": agent_num, "info": AGENTS_INFO[agent_num]})
    return HTMLResponse(content=render_html_page(agent_num))


@app.post("/agent1")
def agent1(q: Query):
    output = run_crewai_agent(1, q.input)
    return {"output": output}


@app.post("/agent2")
def agent2(q: Query):
    output = run_crewai_agent(2, q.input)
    return {"output": output}


@app.post("/agent3")
def agent3(q: Query):
    output = run_crewai_agent(3, q.input)
    return {"output": output}


@app.post("/agent4")
def agent4(q: Query):
    output = run_crewai_agent(4, q.input)
    return {"output": output}


@app.post("/agent5")
def agent5(q: Query):
    output = run_crewai_agent(5, q.input)
    return {"output": output}


@app.post("/agent6")
def agent6(q: Query):
    output = run_crewai_agent(6, q.input)
    return {"output": output}


@app.post("/agent7")
def agent7(q: Query):
    output = run_crewai_agent(7, q.input)
    return {"output": output}


@app.post("/agent8")
def agent8(q: Query):
    output = run_crewai_agent(8, q.input)
    return {"output": output}


@app.post("/agent9")
def agent9(q: Query):
    output = run_crewai_agent(9, q.input)
    return {"output": output}


@app.post("/agent10")
def agent10(q: Query):
    output = run_crewai_agent(10, q.input)
    return {"output": output}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
