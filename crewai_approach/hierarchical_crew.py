#!/usr/bin/env python
"""
Hierarchical CrewAI workflow for PR‑Generator
--------------------------------------------
Adds two low‑level agents that split & summarise large diffs so that no
single prompt can blow past the model context window, then lets a
top‑level PR Orchestrator steer pattern analysis, grouping, validation
and final PR drafting.

Usage (CLI helper):
    python -m crewai_approach.hierarchical_crew \
        /absolute/path/to/repo \
        --max-files 60 \
        --manager-llm gpt-4o
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, List

from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew, before_kickoff

from shared.utils.logging_utils import get_logger

from dotenv import load_dotenv

# --- Load env variables ----------------------------------------------------
# This makes OPENAI_API_KEY (and any other secrets) available to LiteLLM / CrewAI
load_dotenv()  # by default looks for a `.env` file in the current working dir or parent dirs


# ------------------------------------------------------------------------
# IMPORTS — adapt these if your local paths differ
# ------------------------------------------------------------------------
from .tools.repo_analyzer_tool import RepoAnalyzerTool
from .tools.batch_splitter_tool import BatchSplitterTool   # we’ll reuse as chunk router
from .tools.pattern_analyzer_tool import PatternAnalyzerTool
from .tools.grouping_strategy_selector_tool import GroupingStrategySelector
from .tools.group_merging_tool import GroupMergingTool
from .tools.group_validator_tool import GroupValidatorTool
from .tools.group_refiner_tool import GroupRefinerTool
from .tools.file_grouper_tool import FileGrouperTool       # existing grouper
# ------------------------------------------------------------------------

logger = get_logger(__name__)


# ------------------------------------------------------------------------
# 1 · AGENT DEFINITIONS
# ------------------------------------------------------------------------

# --- Worker 1: Chunk Router ------------------------------------------------
chunk_router_agent = Agent(
    role="Chunk Router",
    goal=(
        "Split a potentially giant git diff into ≤N‑line chunks and dispatch "
        "each chunk to the Diff Summariser.  Emit a list of chunk strings."
    ),
    backstory=(
        "You understand git syntax and simple heuristics for chunking: "
        "prefer whole‑file boundaries; if a file is >400 lines, split it on "
        "function or class definitions."
    ),
    allow_delegation=True,
    verbose=False,
)

# --- Worker 2: Diff Summariser --------------------------------------------
summariser_agent = Agent(
    role="Diff Summariser",
    goal=(
        "Convert one code‑diff chunk into a terse 8‑line Markdown summary "
        "that captures intent ('rename function', 'bug‑fix in calc.py'), "
        "plus the file list touched."
    ),
    backstory=(
        "You are a senior developer skilled at spotting what each patch does "
        "without seeing the whole repo.  Keep output *short*—under 1200 tokens."
    ),
    allow_delegation=False,
    verbose=False,
)

def _make_repo_tool(tool_cls, repo_path: Path):
    """Factory to ensure each BaseRepoTool subclass gets a repo_path."""
    return tool_cls(repo_path=str(repo_path))


# --- Analytic & PR agents --------------------------------------------------
# (We’ll pass a dummy Path now; the actual repo_path gets injected later
#  by HierarchicalPRCrew.__init__, which re‑creates the agents.)
def _create_analytic_agents(repo_path: Path):
    pattern_analyzer = Agent(
        role="Pattern Analyzer",
        goal="Detect common themes and recurring motifs across diff summaries.",
        backstory=(
            "A senior engineer specialised in large‑scale code base analytics. "
            "Quickly recognises refactor patterns, duplicated logic, and cross‑file dependencies."
        ),
        verbose=False,
        tools=[_make_repo_tool(PatternAnalyzerTool, repo_path)],
    )

    strategy_selector = Agent(
        role="Grouping Strategist",
        goal="Pick the best high‑level grouping strategy for PR creation.",
        backstory=(
            "Has deep knowledge of multiple grouping heuristics and chooses the one "
            "that minimises reviewer effort and preserves logical cohesion."
        ),
        verbose=False,
        tools=[_make_repo_tool(GroupingStrategySelector, repo_path)],
    )

    pr_writer = Agent(
        role="PR Writer",
        goal="Draft polished pull‑request suggestions for each file cluster.",
        backstory=(
            "Skilled technical writer who turns diff clusters into clear, reviewer‑friendly "
            "PR descriptions with accurate titles and concise reasoning."
        ),
        verbose=False,
        tools=[_make_repo_tool(FileGrouperTool, repo_path)],
    )

    validator = Agent(
        role="PR Validator",
        goal="Validate and refine PR groups so they are coherent, complete, and easy to review.",
        backstory=(
            "Quality‑control specialist ensuring every PR bundle is logically sound, "
            "contains all necessary files, and meets contribution guidelines."
        ),
        verbose=False,
        tools=[
            _make_repo_tool(GroupValidatorTool, repo_path),
            _make_repo_tool(GroupRefinerTool, repo_path),
            _make_repo_tool(GroupMergingTool, repo_path),
        ],
    )
    return pattern_analyzer, strategy_selector, pr_writer, validator


# --- Manager --------------------------------------------------------------
manager_agent = Agent(
    role="PR Orchestrator",
    goal=(
        "Oversee chunking → summarisation → pattern analysis → PR drafting.  "
        "Ensure each agent receives *only* what it needs so token budgets "
        "remain safe; reject outputs over 8 k tokens."
    ),
    backstory=(
        "You are an engineering‑manager‑level architect.  Delegate tasks, "
        "gate‑keep quality, and stitch the final answer."
    ),
    allow_delegation=True,
    verbose=True,
)


# ------------------------------------------------------------------------
# 2 · TASK DEFINITIONS
# ------------------------------------------------------------------------
@task
def chunk_repo(task_input: Dict) -> Task:
    """Call BatchSplitterTool to chop the raw diff into chunks."""
    repo_path: str = task_input["repo_path"]
    max_lines = task_input.get("max_diff_size", 500)
    return BatchSplitterTool(repo_path=repo_path).split_into_chunks(max_lines)


@task
def summarise_chunks(chunk_list: List[str]) -> Task:
    """Loop over chunks and ask summariser_agent to summarise each."""
    summaries = []
    for chunk in chunk_list:
        summaries.append(
            summariser_agent.call(chunk)  # CrewAI Agent's direct call
        )
    return summaries



# --- Task functions with explicit agent dependencies ----------------------
@task
def analyse_patterns(pattern_analyzer_agent: Agent, summaries: List[Dict]) -> Task:
    """Detect recurring themes in the summaries."""
    return pattern_analyzer_agent.call(summaries)

@task
def select_strategy(strategy_selector_agent: Agent, pattern_report: Dict) -> Task:
    """Choose a grouping strategy in light of pattern analysis."""
    return strategy_selector_agent.call(pattern_report)

@task
def draft_prs(pr_writer_agent: Agent, data: Dict) -> Task:
    """
    Build concrete PR suggestions using the chosen strategy and
    original summaries.
    """
    return pr_writer_agent.call(data)

@task
def validate_and_refine(validator_agent: Agent, pr_bundle: Dict) -> Task:
    """Run validator & refiner pass to polish the bundle."""
    return validator_agent.call(pr_bundle)


# ------------------------------------------------------------------------
# 3 · HIERARCHICAL CREW WRAPPER
# ------------------------------------------------------------------------
@CrewBase
class HierarchicalPRCrew:
    """Manager + worker hierarchy for PR generation."""
    # Disable YAML auto‑loading at the class level so CrewBase skips mapping
    agents_config: Dict = {}
    tasks_config: Dict = {}

    def __init__(
        self,
        repo_path: str,
        max_files: Optional[int] = 50,
        max_diff_size: int = 400,
        manager_llm_name: str = "gpt-4o",
        verbose: bool | int = False,
    ):
        self.repo_path = Path(repo_path).resolve()
        if not (self.repo_path / ".git").is_dir():
            raise ValueError(f"Not a git repo: {self.repo_path}")
        self.max_files = max_files
        self.max_diff_size = max_diff_size
        self.manager_llm_name = manager_llm_name
        self.verbose = verbose

        # Disable YAML auto‑loading by providing empty configs
        self.agents_config = {}
        self.tasks_config = {}

        # Re‑create analytic agents with the correct repo path
        (self.pattern_analyzer_agent,
         self.strategy_selector_agent,
         self.pr_writer_agent,
         self.validator_agent) = _create_analytic_agents(self.repo_path)

    # ---------- CrewBase plumbing ---------------------------------------
    @before_kickoff
    def make_inputs(self, inputs: Optional[Dict] = None) -> Dict:
        """Inject runtime arguments so tools can pick them up."""
        return {
            "repo_path": str(self.repo_path),
            "max_diff_size": self.max_diff_size,
            "max_files": self.max_files,
        }

    @crew
    def crew(self) -> Crew:
        """Assemble the actual crew in hierarchical mode."""
        return Crew(
            agents=[
                chunk_router_agent,
                summariser_agent,
                self.pattern_analyzer_agent,
                self.strategy_selector_agent,
                self.pr_writer_agent,
                self.validator_agent,
            ],
            tasks=[
                Task(
                    fn=chunk_repo,
                    description="Split repository diff into chunks",
                    expected_output="List[str]"
                ),
                Task(
                    fn=summarise_chunks,
                    description="Summarise each diff chunk",
                    expected_output="List[Dict]"
                ),
                Task(
                    fn=lambda summaries: analyse_patterns(self.pattern_analyzer_agent, summaries),
                    description="Detect recurring patterns in the summaries",
                    expected_output="Dict"
                ),
                Task(
                    fn=lambda report: select_strategy(self.strategy_selector_agent, report),
                    description="Select appropriate grouping strategy",
                    expected_output="Dict"
                ),
                Task(
                    fn=lambda data: draft_prs(self.pr_writer_agent, data),
                    description="Draft pull‑request suggestions",
                    expected_output="Dict"
                ),
                Task(
                    fn=lambda bundle: validate_and_refine(self.validator_agent, bundle),
                    description="Validate and refine pull‑request bundles",
                    expected_output="Dict"
                ),
            ],
            process=Process.hierarchical,   # ← THE MAGIC LINE
            manager_agent=manager_agent,    # explicit, per docs
            manager_llm=self.manager_llm_name,
            verbose=self.verbose,
        )


# ------------------------------------------------------------------------
# 4 · CLI ENTRY‑POINT
# ------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hierarchical PR‑Generator powered by CrewAI."
    )
    parser.add_argument(
        "repo_path",
        type=str,
        help="Path to the target git repository",
    )
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--max-diff-size", type=int, default=400)
    parser.add_argument(
        "--manager-llm",
        default="gpt-4o",
        help="LLM name for the manager agent",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    crew_instance = HierarchicalPRCrew(
        repo_path=args.repo_path,
        max_files=args.max_files,
        max_diff_size=args.max_diff_size,
        manager_llm_name=args.manager_llm,
        verbose=args.verbose,
    )

    logger.info("🔰 Kicking off hierarchical crew …")
    result = crew_instance.crew().kickoff()

    logger.info("✅ Finished; see JSON suggestions above.")
    print(result)  # simple console dump; callbacks can still write to disk.
    return 0


if __name__ == "__main__":
    sys.exit(main())