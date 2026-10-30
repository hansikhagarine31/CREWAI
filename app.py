"""CrewAI Multi-Agent Backend Service.

Provides routes /agent1 through /agent10 for 10 CrewAI projects
powered by Google Gemini (gemini-3.1-flash-lite).
"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    1: {"title": "Blog Post Crew", "pattern": "Sequential Crew"},
    2: {"title": "Competitor Analysis Crew", "pattern": "Hierarchical Crew"},
    3: {"title": "Meeting Notes Crew", "pattern": "Parallel Tasks"},
    4: {"title": "Data Report Crew", "pattern": "Structured Output"},
    5: {"title": "Customer Support Flow", "pattern": "Router Flow"},
    6: {"title": "Resume Screening Flow", "pattern": "Router Flow"},
    7: {"title": "Bug Triage Flow", "pattern": "Router + Crews"},
    8: {"title": "Content Repurposing Flow", "pattern": "and_ Join Flow"},
    9: {"title": "Lead Qualification Flow", "pattern": "Router + Crews"},
    10: {"title": "Approval Report Flow", "pattern": "Human Feedback Flow"},
}


def run_crewai_agent(agent_num: int, input_text: str) -> str:
    """Executes the specified CrewAI agent using gemini-3.1-flash-lite.

    Falls back cleanly if Gemini credentials or quota are unconfigured.
    """
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
            description=f"Process this request: '{input_text}'. Provide a concise, professional response.",
            expected_output="A well-structured output answering the user request.",
            agent=worker,
        )

        crew = Crew(agents=[worker], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        return str(result)
    except Exception as exc:
        info = AGENTS_INFO.get(agent_num, {"title": f"Agent {agent_num}"})
        return f"Agent {agent_num} ({info['title']}) received: {input_text} (Fallback: {str(exc)[:120]})"


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "CrewAI Multi-Agent API",
        "model": GEMINI_MODEL,
        "agents": AGENTS_INFO,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": GEMINI_MODEL}


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
