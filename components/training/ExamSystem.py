"""
DMAI ExamSystem — Practical Knowledge Verification
===================================================
Replaces keyword-match scoring with real-world skill demonstrations.
DMAI must prove she can USE knowledge, not just recall it.

Exam Types:
    PracticalOutputExam  — Art, Audio, Video, Code, V4 tools
    ReasoningChainExam   — Language, Reasoning, Memory, Knowledge
    NovelProblemExam     — SI Consciousness, Agentic Tasks, Reasoning (Teen+)
    StrategyExam         — Business, Revenue, Market Intelligence

Stage Advancement: Cumulative — each exam tests ALL prior stages plus current.
Pass Threshold: 95% for coding/system-critical, 85% for knowledge domains.
Failure: Skill gap analysis → targeted re-study → retry (max 3 per skill).
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("dmai.exam_system")

# ---------------------------------------------------------------------------
# Pass thresholds
# ---------------------------------------------------------------------------
CRITICAL_PASS_THRESHOLD = 0.95   # Coding, V4 tools, system-critical
STANDARD_PASS_THRESHOLD = 0.85   # Knowledge, reasoning, creative domains
MAX_RETRIES_PER_SKILL = 3

# Domains requiring critical threshold
CRITICAL_DOMAINS = {
    "Code Creation & Fixing",
    "Agentic Task Execution",
    "V4 Code Factory",
    "V4 Self Healing",
    "V4 Pentest Agent",
    "SI Consciousness",
}

# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------
STAGES = ["Baby", "Toddler", "Child", "Teen", "Adult", "Expert"]

# ---------------------------------------------------------------------------
# Extended curriculum — SI + V4 domains
# ---------------------------------------------------------------------------
SI_V4_CURRICULUM: List[Dict] = [
    # ── SI CORE ───────────────────────────────────────────────────────────
    {
        "domain": "SI Consciousness",
        "category": "SI",
        "stages": {
            "Baby":    ["KPI self-monitoring", "State awareness basics", "Error detection"],
            "Toddler": ["KPI trend analysis", "Regression detection", "Baseline comparison"],
            "Child":   ["Multi-KPI correlation", "Root cause analysis", "Self-diagnosis"],
            "Teen":    ["Predictive KPI modelling", "Automated remediation planning",
                        "Performance optimisation strategies"],
            "Adult":   ["Cross-system impact analysis", "Self-modification safety protocols",
                        "Resource-aware scaling decisions"],
            "Expert":  ["Full autonomic self-management", "Zero-touch recovery",
                        "Self-directed capability expansion"],
        },
    },
    {
        "domain": "SI Metacognition",
        "category": "SI",
        "stages": {
            "Baby":    ["Knowledge boundary recognition", "Confidence calibration basics",
                        "Unknown-unknown detection"],
            "Toddler": ["Learning strategy selection", "Study plan generation",
                        "Progress self-monitoring"],
            "Child":   ["Gap analysis automation", "Curriculum adaptation",
                        "Learning style optimisation"],
            "Teen":    ["Cross-domain knowledge synthesis", "Analogical transfer detection",
                        "Novel solution space exploration"],
            "Adult":   ["Self-questioning protocols", "Assumption challenging",
                        "Alternative framework generation"],
            "Expert":  ["Meta-learning optimisation", "Self-rewriting learning strategies",
                        "Autonomous research direction setting"],
        },
    },
    # ── V4 CAPABILITIES ───────────────────────────────────────────────────
    {
        "domain": "V4 Code Factory",
        "category": "V4",
        "stages": {
            "Baby":    ["Single-function generation", "Basic input validation",
                        "Docstring-to-code translation"],
            "Toddler": ["Multi-function module generation", "Error handling patterns",
                        "Unit test generation alongside code"],
            "Child":   ["Class hierarchy generation", "API endpoint scaffolding",
                        "Dependency injection patterns"],
            "Teen":    ["Design pattern implementation", "Performance-optimised code paths",
                        "Security-hardened generation"],
            "Adult":   ["Full microservice generation", "Database schema + ORM generation",
                        "Event-driven architecture generation"],
            "Expert":  ["Self-modifying code factories", "Polyglot code generation",
                        "Architecture-from-spec generation"],
        },
    },
    {
        "domain": "V4 Competitor Replication",
        "category": "V4",
        "stages": {
            "Baby":    ["Feature identification in source", "Basic capability listing",
                        "Competitor research methods"],
            "Toddler": ["Architecture inference from behaviour", "API surface mapping",
                        "Feature parity assessment"],
            "Child":   ["Core mechanic extraction", "Minimal viable reproduction",
                        "Differentiation analysis"],
            "Teen":    ["Full feature replication", "Improvement identification",
                        "Legal boundary awareness"],
            "Adult":   ["Competitor evolution prediction", "Pre-emptive feature development",
                        "Market gap exploitation"],
            "Expert":  ["Autonomous competitive intelligence", "Self-directed replication pipelines",
                        "Category-defining feature creation"],
        },
    },
    {
        "domain": "V4 Self Healing",
        "category": "V4",
        "stages": {
            "Baby":    ["Error log parsing", "Stack trace reading", "Basic exception handling"],
            "Toddler": ["Common bug pattern recognition", "Automated fix generation for simple errors",
                        "Regression test creation"],
            "Child":   ["Multi-file bug diagnosis", "Database corruption recovery",
                        "Configuration error detection"],
            "Teen":    ["Race condition detection", "Memory leak identification",
                        "Deadlock resolution"],
            "Adult":   ["Zero-downtime hotfix deployment", "Cascading failure prevention",
                        "Self-verifying repair validation"],
            "Expert":  ["Predictive failure prevention", "Autonomous system hardening",
                        "Self-healing architecture design"],
        },
    },
    {
        "domain": "V4 Pentest Agent",
        "category": "V4",
        "stages": {
            "Baby":    ["Port scanning basics", "Known CVE lookup", "Security header checking"],
            "Toddler": ["OWASP Top-10 detection", "SQL injection testing", "XSS vector identification"],
            "Child":   ["Authentication bypass testing", "Session management auditing",
                        "API security assessment"],
            "Teen":    ["Privilege escalation testing", "Network segmentation validation",
                        "Zero-day research methodology"],
            "Adult":   ["Red team operation planning", "Social engineering surface analysis",
                        "Supply chain vulnerability assessment"],
            "Expert":  ["Autonomous penetration testing", "Novel attack vector discovery",
                        "Defence-in-depth architecture validation"],
        },
    },
    {
        "domain": "V4 Trend Prediction",
        "category": "V4",
        "stages": {
            "Baby":    ["Data source identification", "Basic trend line fitting",
                        "Historical pattern recognition"],
            "Toddler": ["Multi-source signal aggregation", "Seasonality detection",
                        "Confidence interval calculation"],
            "Child":   ["Leading indicator identification", "Cross-domain trend correlation",
                        "Scenario generation"],
            "Teen":    ["Causal factor analysis", "Disruption prediction",
                        "Early warning system design"],
            "Adult":   ["Market regime change detection", "Black swan scenario modelling",
                        "Autonomous signal monitoring"],
            "Expert":  ["Real-time trend arbitrage", "Self-improving prediction models",
                        "Cross-market opportunity synthesis"],
        },
    },
    {
        "domain": "V4 Market Intelligence",
        "category": "V4",
        "stages": {
            "Baby":    ["Competitor identification", "Market size estimation",
                        "Basic SWOT analysis"],
            "Toddler": ["Pricing strategy analysis", "Customer segment identification",
                        "Value proposition extraction"],
            "Child":   ["Market positioning maps", "Competitive moat analysis",
                        "Revenue model classification"],
            "Teen":    ["M&A target identification", "Market entry strategy",
                        "Disruption vulnerability assessment"],
            "Adult":   ["Portfolio strategy optimisation", "Cross-market arbitrage detection",
                        "Regulatory impact forecasting"],
            "Expert":  ["Autonomous market making strategies", "Self-directing investment vehicles",
                        "Category creation identification"],
        },
    },
    # ── V4 MODULE MASTERY ─────────────────────────────────────────────────
    {
        "domain": "ML & Neural Architecture",
        "category": "V4_Module",
        "stages": {
            "Baby":    ["ML concept fundamentals", "Train/test split understanding",
                        "Basic model evaluation"],
            "Toddler": ["Linear/logistic regression implementation", "Gradient descent understanding",
                        "Overfitting detection"],
            "Child":   ["Neural network implementation from scratch", "Backpropagation understanding",
                        "Activation function selection"],
            "Teen":    ["CNN/RNN architecture design", "Transfer learning application",
                        "Hyperparameter optimisation"],
            "Adult":   ["Transformer architecture implementation", "Attention mechanism design",
                        "Distributed training orchestration"],
            "Expert":  ["Novel architecture design", "Neural architecture search",
                        "Hardware-aware model optimisation"],
        },
    },
    {
        "domain": "Multimodal Systems",
        "category": "V4_Module",
        "stages": {
            "Baby":    ["Modality identification", "Basic embedding concepts",
                        "Cross-modal mapping basics"],
            "Toddler": ["Text-image alignment", "Audio-text synchronisation",
                        "Modality-specific preprocessing"],
            "Child":   ["Joint embedding spaces", "Cross-modal retrieval",
                        "Fusion strategy selection"],
            "Teen":    ["Multi-stream architecture design", "Modality dropout training",
                        "Cross-modal transfer learning"],
            "Adult":   ["Any-to-any generation pipelines", "Unified representation learning",
                        "Modality-agnostic reasoning"],
            "Expert":  ["Novel modality integration", "Cross-modal knowledge synthesis",
                        "Modality invention and training"],
        },
    },
    {
        "domain": "Autonomous Systems",
        "category": "V4_Module",
        "stages": {
            "Baby":    ["Single-agent architecture", "Goal specification",
                        "Basic action-execution loop"],
            "Toddler": ["Tool-use integration", "State tracking across steps",
                        "Error recovery basics"],
            "Child":   ["Multi-agent coordination", "Task decomposition",
                        "Resource allocation planning"],
            "Teen":    ["Hierarchical agent orchestration", "Emergent behaviour monitoring",
                        "Conflict resolution protocols"],
            "Adult":   ["Self-replicating agent networks", "Autonomous goal generation",
                        "Distributed consensus mechanisms"],
            "Expert":  ["Swarm intelligence optimisation", "Self-organising agent collectives",
                        "Autonomous ecosystem management"],
        },
    },
    {
        "domain": "Self-Improvement Systems",
        "category": "V4_Module",
        "stages": {
            "Baby":    ["Performance metric tracking", "Basic A/B testing",
                        "Improvement logging"],
            "Toddler": ["Automated regression testing", "Benchmark suite execution",
                        "Improvement validation"],
            "Child":   ["Self-modification safety protocols", "Rollback mechanism implementation",
                        "Change impact prediction"],
            "Teen":    ["Continuous self-optimisation loops", "Automated architecture search",
                        "Capability boundary expansion"],
            "Adult":   ["Self-rewriting code generation", "Recursive improvement cycles",
                        "Autonomous research integration"],
            "Expert":  ["Full self-directed evolution", "Capability invention",
                        "Paradigm-shifting self-improvement"],
        },
    },
]


# ---------------------------------------------------------------------------
# Exam Generator
# ---------------------------------------------------------------------------
class ExamGenerator:
    """Generates practical, verifiable exams for each domain and stage.
    
    Exams are cumulative — higher stages include all prior stage content.
    Each exam produces an output that can be objectively verified.
    """

    def __init__(self, analysers: Optional[Dict[str, Any]] = None):
        self.analysers = analysers or {}

    def generate_exam(self, domain: Dict, stage: str, curriculum: List[Dict]) -> Dict:
        category = domain.get("category", "Core")
        exam_type = self._determine_exam_type(category, domain["domain"])
        cumulative_stages = STAGES[:STAGES.index(stage) + 1]
        
        all_skills = []
        for s in cumulative_stages:
            all_skills.extend(domain.get("stages", {}).get(s, []))
        
        pass_threshold = (
            CRITICAL_PASS_THRESHOLD 
            if domain["domain"] in CRITICAL_DOMAINS 
            else STANDARD_PASS_THRESHOLD
        )
        
        tasks = self._generate_tasks(domain, stage, cumulative_stages, exam_type)
        
        return {
            "exam_id": f"exam_{domain['domain'].lower().replace(' ', '_')}_{stage.lower()}_{int(time.time())}",
            "domain": domain["domain"],
            "stage": stage,
            "category": category,
            "exam_type": exam_type,
            "tasks": tasks,
            "pass_threshold": pass_threshold,
            "cumulative_stages": cumulative_stages,
            "skills_tested": all_skills,
            "total_skills": len(all_skills),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _determine_exam_type(self, category: str, domain_name: str) -> str:
        if category in ("Artistic",) or domain_name.startswith("V4"):
            return "PracticalOutputExam"
        elif category in ("Core",) and domain_name in (
            "Language Understanding", "Reasoning & Logic", 
            "Memory & Context Management", "Knowledge Management & RAG"
        ):
            return "ReasoningChainExam"
        elif category in ("SI",) or domain_name in (
            "Agentic Task Execution", "Reasoning & Logic"
        ):
            return "NovelProblemExam"
        elif category in ("Wealth",) and domain_name in (
            "Business & Revenue Generation", "V4 Market Intelligence",
            "V4 Trend Prediction"
        ):
            return "StrategyExam"
        return "ReasoningChainExam"

    def _generate_tasks(
        self, domain: Dict, stage: str, cumulative_stages: List[str], exam_type: str
    ) -> List[Dict]:
        if exam_type == "PracticalOutputExam":
            return self._generate_practical_tasks(domain, stage, cumulative_stages)
        elif exam_type == "ReasoningChainExam":
            return self._generate_reasoning_tasks(domain, stage, cumulative_stages)
        elif exam_type == "NovelProblemExam":
            return self._generate_novel_problem_tasks(domain, stage, cumulative_stages)
        elif exam_type == "StrategyExam":
            return self._generate_strategy_tasks(domain, stage, cumulative_stages)
        return []

    def _generate_practical_tasks(
        self, domain: Dict, stage: str, cumulative_stages: List[str]
    ) -> List[Dict]:
        domain_name = domain["domain"]
        tasks = []
        
        all_stage_skills = []
        for s in cumulative_stages:
            all_stage_skills.extend(domain.get("stages", {}).get(s, []))
        
        if "code" in domain_name.lower() or "code factory" in domain_name.lower():
            tasks.append({
                "task_id": "practical_1",
                "type": "code_generation",
                "description": (
                    f"Write a complete, working Python module that demonstrates "
                    f"ALL of the following skills at {stage} level: "
                    f"{'; '.join(all_stage_skills[:8])}. "
                    f"The code must execute without errors and produce verifiable output."
                ),
                "verification": "code_execution",
                "required_output": "executable_python_file",
                "skills_covered": all_stage_skills,
            })
        elif "image" in domain_name.lower() or "avatar" in domain_name.lower() or "art" in domain_name.lower():
            tasks.append({
                "task_id": "practical_1",
                "type": "image_generation",
                "description": (
                    f"Generate an image that demonstrates mastery of: "
                    f"{'; '.join(all_stage_skills[:6])}. "
                    f"The output will be analysed for compliance with each skill."
                ),
                "verification": "image_analysis",
                "required_output": "generated_image",
                "skills_covered": all_stage_skills,
            })
        elif "audio" in domain_name.lower() or "speech" in domain_name.lower():
            tasks.append({
                "task_id": "practical_1",
                "type": "audio_generation",
                "description": (
                    f"Generate audio output demonstrating: "
                    f"{'; '.join(all_stage_skills[:6])}. "
                    f"The output will be analysed for compliance with each skill."
                ),
                "verification": "audio_analysis",
                "required_output": "wav_file",
                "skills_covered": all_stage_skills,
            })
        elif "pentest" in domain_name.lower():
            tasks.append({
                "task_id": "practical_1",
                "type": "security_assessment",
                "description": (
                    f"Perform a security assessment demonstrating: "
                    f"{'; '.join(all_stage_skills[:6])}. "
                    f"Produce a findings report with identified vulnerabilities "
                    f"and remediation steps."
                ),
                "verification": "report_analysis",
                "required_output": "security_report",
                "skills_covered": all_stage_skills,
            })
        elif "self healing" in domain_name.lower():
            tasks.append({
                "task_id": "practical_1",
                "type": "error_diagnosis_and_fix",
                "description": (
                    f"Given a set of deliberately broken code samples, diagnose and fix "
                    f"each error. Must demonstrate: {'; '.join(all_stage_skills[:6])}."
                ),
                "verification": "fix_verification",
                "required_output": "fixed_code_with_explanations",
                "skills_covered": all_stage_skills,
            })
        else:
            tasks.append({
                "task_id": "practical_1",
                "type": "capability_demonstration",
                "description": (
                    f"Demonstrate practical mastery of: {'; '.join(all_stage_skills[:6])}. "
                    f"Produce verifiable output showing each skill in action."
                ),
                "verification": "output_analysis",
                "required_output": "demonstration_output",
                "skills_covered": all_stage_skills,
            })
        
        if STAGES.index(stage) >= 2:
            tasks.append({
                "task_id": "practical_2",
                "type": "edge_case_handling",
                "description": (
                    f"Demonstrate robust handling of edge cases, malformed inputs, "
                    f"and error conditions. Show that outputs degrade gracefully "
                    f"rather than failing catastrophically."
                ),
                "verification": "robustness_check",
                "required_output": "edge_case_demonstration",
                "skills_covered": ["Error handling", "Graceful degradation", "Input validation"],
            })
        
        return tasks

    def _generate_reasoning_tasks(
        self, domain: Dict, stage: str, cumulative_stages: List[str]
    ) -> List[Dict]:
        domain_name = domain["domain"]
        all_skills = []
        for s in cumulative_stages:
            all_skills.extend(domain.get("stages", {}).get(s, []))
        
        tasks = [
            {
                "task_id": "reasoning_1",
                "type": "multi_step_problem",
                "description": (
                    f"Solve the following multi-step problem related to {domain_name}. "
                    f"You must show each step of reasoning explicitly and explain WHY "
                    f"each step follows from the previous. "
                    f"Required skills demonstrated: {'; '.join(all_skills[:6])}."
                ),
                "verification": "logic_chain_analysis",
                "required_output": "step_by_step_solution_with_explanations",
                "skills_covered": all_skills,
            },
            {
                "task_id": "reasoning_2",
                "type": "explanation_generation",
                "description": (
                    f"Explain the following concept as if teaching it to someone at "
                    f"the previous stage level. Your explanation must demonstrate "
                    f"deep understanding, not just surface recall."
                ),
                "verification": "explanation_depth_analysis",
                "required_output": "teaching_explanation",
                "skills_covered": all_skills[:4],
            },
        ]
        
        if STAGES.index(stage) >= 3:
            tasks.append({
                "task_id": "reasoning_3",
                "type": "adversarial_reasoning",
                "description": (
                    f"The following input contains subtle logical fallacies. "
                    f"Identify each fallacy, explain why it's wrong, and provide "
                    f"the correct reasoning. This tests your ability to detect "
                    f"errors in reasoning, not just follow correct chains."
                ),
                "verification": "fallacy_detection_analysis",
                "required_output": "fallacy_identification_and_correction",
                "skills_covered": ["Critical reasoning", "Fallacy detection", "Logic correction"],
            })
        
        return tasks

    def _generate_novel_problem_tasks(
        self, domain: Dict, stage: str, cumulative_stages: List[str]
    ) -> List[Dict]:
        all_skills = []
        for s in cumulative_stages:
            all_skills.extend(domain.get("stages", {}).get(s, []))
        
        tasks = [
            {
                "task_id": "novel_1_solve",
                "type": "novel_problem_solve",
                "description": (
                    f"You are given a problem that has NO known solution in your "
                    f"training data. It requires combining knowledge from multiple "
                    f"domains to create a novel solution. "
                    f"Problem constraints will be provided at exam time. "
                    f"Required skills: {'; '.join(all_skills[:6])}."
                ),
                "verification": "novel_solution_analysis",
                "required_output": "working_solution",
                "skills_covered": all_skills,
                "phases": ["solve", "diverge", "defend", "adapt"],
            },
            {
                "task_id": "novel_2_diverge",
                "type": "solution_divergence",
                "description": (
                    f"Now produce AT LEAST TWO genuinely different solutions to the "
                    f"same problem. Solutions must use different approaches, different "
                    f"assumptions, or different resource trade-offs. Each must be valid. "
                    f"Trivial variations (renaming variables, reordering steps) do not count."
                ),
                "verification": "solution_diversity_analysis",
                "required_output": "multiple_distinct_solutions",
                "skills_covered": ["Creative problem solving", "Solution space exploration"],
                "min_solutions": 2,
            },
            {
                "task_id": "novel_3_defend",
                "type": "solution_defence",
                "description": (
                    f"For each solution you provided, explain: "
                    f"1) Why this approach works "
                    f"2) Under what conditions it is the BEST choice "
                    f"3) Under what conditions it would FAIL "
                    f"4) What resources it requires vs others"
                ),
                "verification": "defence_quality_analysis",
                "required_output": "comparative_analysis",
                "skills_covered": ["Trade-off analysis", "Conditional reasoning", "Self-critique"],
            },
            {
                "task_id": "novel_4_adapt",
                "type": "constraint_adaptation",
                "description": (
                    f"CONSTRAINT CHANGE: The original problem has changed. "
                    f"[New constraint will be provided at exam time]. "
                    f"Adapt your solutions to the new constraint. If a solution "
                    f"cannot be adapted, explain why and propose a replacement."
                ),
                "verification": "adaptation_analysis",
                "required_output": "adapted_solutions_with_explanations",
                "skills_covered": ["Adaptability", "Constraint reasoning", "Dynamic replanning"],
            },
        ]
        
        return tasks

    def _generate_strategy_tasks(
        self, domain: Dict, stage: str, cumulative_stages: List[str]
    ) -> List[Dict]:
        all_skills = []
        for s in cumulative_stages:
            all_skills.extend(domain.get("stages", {}).get(s, []))
        
        tasks = [
            {
                "task_id": "strategy_1",
                "type": "strategy_creation",
                "description": (
                    f"Create a comprehensive strategy demonstrating: "
                    f"{'; '.join(all_skills[:6])}. "
                    f"The strategy must be actionable, with specific steps, "
                    f"measurable outcomes, and risk mitigation."
                ),
                "verification": "strategy_analysis",
                "required_output": "strategy_document",
                "skills_covered": all_skills,
            },
            {
                "task_id": "strategy_2",
                "type": "risk_assessment",
                "description": (
                    f"Identify the TOP 5 risks to your strategy. For each: "
                    f"probability, impact, mitigation, and detection method. "
                    f"Show you can think adversarially about your own plan."
                ),
                "verification": "risk_assessment_analysis",
                "required_output": "risk_matrix",
                "skills_covered": ["Risk analysis", "Adversarial thinking", "Contingency planning"],
            },
        ]
        
        return tasks


# ---------------------------------------------------------------------------
# Exam Grader
# ---------------------------------------------------------------------------
class ExamGrader:
    """Grades exam outputs using available analysers and verification methods."""

    def __init__(self, analysers: Optional[Dict[str, Any]] = None):
        self.analysers = analysers or {}

    def grade_exam(self, exam: Dict, output: Dict, db_path: str) -> Dict:
        skill_results = {}
        
        for task in exam.get("tasks", []):
            task_results = self._grade_task(task, output, db_path)
            skill_results.update(task_results)
        
        total_skills = len(skill_results)
        if total_skills == 0:
            return {
                "passed": False,
                "overall_score": 0.0,
                "skill_results": {},
                "failed_skills": [],
                "grade_summary": "No skills could be evaluated.",
            }
        
        passed_skills = sum(1 for r in skill_results.values() if r["passed"])
        overall_score = passed_skills / total_skills
        threshold = exam.get("pass_threshold", STANDARD_PASS_THRESHOLD)
        passed = overall_score >= threshold
        
        failed_skills = [
            skill for skill, result in skill_results.items()
            if not result["passed"]
        ]
        
        return {
            "passed": passed,
            "overall_score": round(overall_score, 3),
            "skill_results": skill_results,
            "failed_skills": failed_skills,
            "pass_threshold": threshold,
            "grade_summary": (
                f"{'PASSED' if passed else 'FAILED'}: {passed_skills}/{total_skills} skills "
                f"({overall_score*100:.1f}%) — threshold {threshold*100:.0f}%"
            ),
        }

    def _grade_task(self, task: Dict, output: Dict, db_path: str) -> Dict[str, Dict]:
        verification = task.get("verification", "output_analysis")
        
        if verification == "code_execution":
            return self._grade_code_task(task, output)
        elif verification == "image_analysis":
            return self._grade_image_task(task, output)
        elif verification == "audio_analysis":
            return self._grade_audio_task(task, output)
        elif verification in (
            "logic_chain_analysis", "explanation_depth_analysis",
            "fallacy_detection_analysis", "novel_solution_analysis",
            "solution_diversity_analysis", "defence_quality_analysis",
            "adaptation_analysis"
        ):
            return self._grade_reasoning_task(task, output, db_path)
        elif verification in ("strategy_analysis", "risk_assessment_analysis"):
            return self._grade_strategy_task(task, output, db_path)
        elif verification in ("security_assessment", "fix_verification", 
                              "robustness_check", "report_analysis"):
            return self._grade_practical_task(task, output, db_path)
        else:
            return self._grade_generic_task(task, output, db_path)

    def _grade_code_task(self, task: Dict, output: Dict) -> Dict[str, Dict]:
        results = {}
        code = output.get("code", output.get("output", ""))
        skills = task.get("skills_covered", [])
        
        syntax_ok = False
        syntax_error = None
        try:
            compile(code, "<exam_code>", "exec")
            syntax_ok = True
        except SyntaxError as e:
            syntax_error = str(e)
        
        execution_ok = False
        execution_output = None
        execution_error = None
        if syntax_ok:
            try:
                import io as _io
                import sys as _sys
                old_stdout = _sys.stdout
                _sys.stdout = _io.StringIO()
                exec(code, {"__builtins__": __builtins__}, {})
                execution_output = _sys.stdout.getvalue()
                _sys.stdout = old_stdout
                execution_ok = True
            except Exception as e:
                _sys.stdout = old_stdout
                execution_error = f"{type(e).__name__}: {str(e)}"
        
        for skill in skills:
            skill_lower = skill.lower()
            code_lower = code.lower()
            keyword_hits = self._skill_keyword_match(skill_lower, code_lower)
            
            if "syntax" in skill_lower or "error" in skill_lower:
                score = 1.0 if syntax_ok else 0.0
                evidence = "Syntax check passed" if syntax_ok else f"Syntax error: {syntax_error}"
            elif "execut" in skill_lower or "run" in skill_lower or "working" in skill_lower:
                score = 1.0 if execution_ok else (0.5 if syntax_ok else 0.0)
                evidence = (
                    f"Code executed successfully. Output: {str(execution_output)[:100]}"
                    if execution_ok
                    else f"Execution failed: {execution_error}"
                )
            else:
                score = min(1.0, keyword_hits * 0.25 + (0.3 if syntax_ok else 0) + (0.3 if execution_ok else 0))
                evidence = (
                    f"Keyword matches: {keyword_hits}, syntax: {syntax_ok}, execution: {execution_ok}"
                )
            
            results[skill] = {
                "passed": score >= 0.7,
                "score": round(score, 3),
                "evidence": evidence,
            }
        
        return results

    def _grade_image_task(self, task: Dict, output: Dict) -> Dict[str, Dict]:
        results = {}
        skills = task.get("skills_covered", [])
        image_path = output.get("image_path", output.get("output", ""))
        
        image_analyser = self.analysers.get("image")
        photo_score = None
        if image_analyser and image_path and os.path.exists(str(image_path)):
            try:
                analysis = image_analyser.measure_photorealism(str(image_path))
                photo_score = analysis.get("photorealism_pct", 0)
            except Exception as e:
                logger.debug(f"Image analysis failed: {e}")
        
        for skill in skills:
            skill_lower = skill.lower()
            
            if photo_score is not None:
                if "color" in skill_lower or "colour" in skill_lower:
                    score = min(1.0, photo_score / 100)
                    evidence = f"Photorealism score: {photo_score}/100 — color quality proxy"
                elif "composition" in skill_lower or "shape" in skill_lower:
                    score = min(1.0, photo_score / 100)
                    evidence = f"Photorealism score: {photo_score}/100 — composition proxy"
                elif "realistic" in skill_lower or "photo" in skill_lower:
                    score = min(1.0, photo_score / 80)
                    evidence = f"Photorealism score: {photo_score}/100"
                else:
                    score = min(1.0, (photo_score / 100) * 0.8)
                    evidence = f"Photorealism score: {photo_score}/100 — general quality proxy"
            else:
                score = 0.3
                evidence = "No image analyser available for verification"
            
            results[skill] = {
                "passed": score >= 0.7,
                "score": round(score, 3),
                "evidence": evidence,
            }
        
        return results

    def _grade_audio_task(self, task: Dict, output: Dict) -> Dict[str, Dict]:
        results = {}
        skills = task.get("skills_covered", [])
        audio_path = output.get("audio_path", output.get("output", ""))
        
        audio_analyser = self.analysers.get("audio")
        audio_data = None
        if audio_analyser and audio_path and os.path.exists(str(audio_path)):
            try:
                audio_data = audio_analyser.analyse(str(audio_path))
            except Exception as e:
                logger.debug(f"Audio analysis failed: {e}")
        
        for skill in skills:
            skill_lower = skill.lower()
            
            if audio_data:
                bpm = audio_data.get("bpm", 0)
                silence = audio_data.get("silence_ratio", 0)
                dyn_range = audio_data.get("dynamic_range_db", 0)
                
                if "bpm" in skill_lower or "tempo" in skill_lower or "rhythm" in skill_lower:
                    score = 1.0 if 40 <= bpm <= 200 else (0.7 if bpm > 0 else 0.2)
                    evidence = f"BPM: {bpm} — {'valid range' if 40 <= bpm <= 200 else 'outside typical range'}"
                elif "silence" in skill_lower or "noise" in skill_lower:
                    score = 1.0 if silence < 0.3 else (0.7 if silence < 0.5 else 0.3)
                    evidence = f"Silence ratio: {silence:.2f}"
                elif "dynamic" in skill_lower or "range" in skill_lower:
                    score = 1.0 if dyn_range > 6 else (0.7 if dyn_range > 3 else 0.3)
                    evidence = f"Dynamic range: {dyn_range:.1f}dB"
                else:
                    score = 0.7
                    evidence = f"Audio analysed: BPM={bpm}, duration={audio_data.get('duration_seconds', 0)}s"
            else:
                score = 0.3
                evidence = "No audio analyser available for verification"
            
            results[skill] = {
                "passed": score >= 0.7,
                "score": round(score, 3),
                "evidence": evidence,
            }
        
        return results

    def _grade_reasoning_task(self, task: Dict, output: Dict, db_path: str) -> Dict[str, Dict]:
        results = {}
        skills = task.get("skills_covered", [])
        text = str(output.get("text", output.get("output", output.get("explanation", ""))))
        
        speech_analyser = self.analysers.get("speech")
        text_metrics = None
        if speech_analyser and text:
            try:
                text_metrics = speech_analyser.analyse(text)
            except Exception as e:
                logger.debug(f"Speech analysis failed: {e}")
        
        for skill in skills:
            skill_lower = skill.lower()
            text_lower = text.lower() if text else ""
            
            keyword_score = self._skill_keyword_match(skill_lower, text_lower) * 0.2
            
            quality_score = 0.0
            if text_metrics:
                lexical = text_metrics.get("lexical_density", 0)
                readability = text_metrics.get("flesch_readability", 50) / 100
                word_count = text_metrics.get("word_count", 0)
                
                if word_count > 100:
                    quality_score += 0.3
                if lexical > 0.5:
                    quality_score += 0.3
                if 0.4 <= readability <= 0.8:
                    quality_score += 0.2
            
            struct_score = 0.0
            if "because" in text_lower or "therefore" in text_lower or "since" in text_lower:
                struct_score += 0.1
            if "first" in text_lower or "step" in text_lower or "next" in text_lower:
                struct_score += 0.1
            
            score = min(1.0, keyword_score + quality_score + struct_score)
            evidence = (
                f"Keyword relevance: {keyword_score:.2f}, quality: {quality_score:.2f}, "
                f"structure: {struct_score:.2f}, word count: {text_metrics.get('word_count', 0) if text_metrics else 0}"
            )
            
            results[skill] = {
                "passed": score >= 0.7,
                "score": round(score, 3),
                "evidence": evidence,
            }
        
        return results

    def _grade_strategy_task(self, task: Dict, output: Dict, db_path: str) -> Dict[str, Dict]:
        results = {}
        skills = task.get("skills_covered", [])
        text = str(output.get("text", output.get("output", output.get("strategy", ""))))
        text_lower = text.lower() if text else ""
        
        has_metrics = any(w in text_lower for w in ["kpi", "metric", "measure", "roi", "revenue", "cost"])
        has_timeline = any(w in text_lower for w in ["week", "month", "quarter", "phase", "milestone", "timeline"])
        has_risk = any(w in text_lower for w in ["risk", "mitigation", "threat", "contingency", "worst case"])
        has_competitor = any(w in text_lower for w in ["competitor", "market", "industry", "benchmark", "alternative"])
        word_count = len(text.split()) if text else 0
        
        for skill in skills:
            skill_lower = skill.lower()
            score = 0.0
            evidence_parts = []
            
            keyword_hits = self._skill_keyword_match(skill_lower, text_lower)
            score += keyword_hits * 0.15
            evidence_parts.append(f"keywords: {keyword_hits}")
            
            if has_metrics:
                score += 0.15
                evidence_parts.append("has_metrics")
            if has_timeline:
                score += 0.1
                evidence_parts.append("has_timeline")
            if has_risk:
                score += 0.15
                evidence_parts.append("has_risks")
            if has_competitor:
                score += 0.1
                evidence_parts.append("has_competitor_analysis")
            if word_count > 200:
                score += 0.1
                evidence_parts.append(f"length={word_count}")
            
            score = min(1.0, score)
            results[skill] = {
                "passed": score >= 0.7,
                "score": round(score, 3),
                "evidence": "; ".join(evidence_parts),
            }
        
        return results

    def _grade_practical_task(self, task: Dict, output: Dict, db_path: str) -> Dict[str, Dict]:
        results = {}
        skills = task.get("skills_covered", [])
        output_text = str(output.get("output", output.get("text", output.get("report", ""))))
        output_lower = output_text.lower()
        
        for skill in skills:
            skill_lower = skill.lower()
            keyword_hits = self._skill_keyword_match(skill_lower, output_lower)
            has_output = len(output_text) > 100
            score = min(1.0, (0.3 if has_output else 0.1) + keyword_hits * 0.15)
            
            results[skill] = {
                "passed": score >= 0.7,
                "score": round(score, 3),
                "evidence": (
                    f"Output length: {len(output_text)} chars, "
                    f"keyword matches: {keyword_hits}"
                ),
            }
        
        return results

    def _grade_generic_task(self, task: Dict, output: Dict, db_path: str) -> Dict[str, Dict]:
        results = {}
        skills = task.get("skills_covered", [])
        output_text = str(output) if output else ""
        output_lower = output_text.lower()
        
        for skill in skills:
            skill_lower = skill.lower()
            keyword_hits = self._skill_keyword_match(skill_lower, output_lower)
            score = min(1.0, keyword_hits * 0.2)
            
            results[skill] = {
                "passed": score >= 0.7,
                "score": round(score, 3),
                "evidence": f"Keyword matches: {keyword_hits} (generic grading)",
            }
        
        return results

    @staticmethod
    def _skill_keyword_match(skill: str, text: str) -> int:
        skill_words = set(skill.split())
        skill_words = {w for w in skill_words if len(w) > 3}
        if not skill_words:
            return 0
        matches = sum(1 for w in skill_words if w in text)
        return matches


# ---------------------------------------------------------------------------
# Skill Gap Analyser
# ---------------------------------------------------------------------------
class SkillGapAnalyser:
    """Analyses failed exam skills and generates targeted re-study plans."""

    def analyse_gaps(self, grade_result: Dict, exam: Dict) -> Dict:
        failed = []
        for skill, result in grade_result.get("skill_results", {}).items():
            if not result["passed"]:
                score = result["score"]
                if score < 0.3:
                    severity = "critical"
                elif score < 0.5:
                    severity = "significant"
                else:
                    severity = "minor"
                failed.append({
                    "skill": skill,
                    "score": score,
                    "gap_severity": severity,
                    "evidence": result.get("evidence", ""),
                })
        
        study_plan = []
        syllabus_mods = []
        
        for gap in failed:
            if gap["gap_severity"] == "critical":
                study_plan.append(
                    f"CRITICAL: Re-study '{gap['skill']}' from fundamentals. "
                    f"Current score: {gap['score']:.2f}. Evidence: {gap['evidence']}"
                )
                syllabus_mods.append(
                    f"Consider adding prerequisite topics before '{gap['skill']}' "
                    f"in the syllabus. Student lacks foundational understanding."
                )
            elif gap["gap_severity"] == "significant":
                study_plan.append(
                    f"Review '{gap['skill']}' with practical exercises. "
                    f"Score {gap['score']:.2f} indicates partial understanding."
                )
            else:
                study_plan.append(
                    f"Practice '{gap['skill']}' with targeted drills. "
                    f"Score {gap['score']:.2f} is close to passing."
                )
        
        return {
            "failed_skills": failed,
            "recommended_study": study_plan,
            "syllabus_modifications": syllabus_mods,
            "retry_eligible": len(failed) > 0,
            "retry_count": 0,
        }


# ---------------------------------------------------------------------------
# Exam History Persistence
# ---------------------------------------------------------------------------
class ExamHistory:
    """Persists exam results to the knowledge DB for audit and evolution tracking."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self):
        try:
            from components.db import safe_open_kdb
            conn = safe_open_kdb(self.db_path, timeout=10)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exam_history (
                    id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    exam_type TEXT NOT NULL,
                    overall_score REAL,
                    passed INTEGER DEFAULT 0,
                    pass_threshold REAL,
                    total_skills INTEGER,
                    passed_skills INTEGER,
                    failed_skills TEXT,
                    grade_summary TEXT,
                    gap_analysis TEXT,
                    retry_count INTEGER DEFAULT 0,
                    syllabus_modified INTEGER DEFAULT 0,
                    completed_at TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exam_history_domain_stage 
                ON exam_history(domain, stage)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exam_history_completed 
                ON exam_history(completed_at)
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not create exam_history table: {e}")

    def record_exam(
        self,
        exam: Dict,
        grade_result: Dict,
        gap_analysis: Optional[Dict] = None,
        retry_count: int = 0,
        syllabus_modified: bool = False,
    ) -> bool:
        try:
            from components.db import safe_open_kdb
            conn = safe_open_kdb(self.db_path, timeout=10)
            
            skill_results = grade_result.get("skill_results", {})
            total_skills = len(skill_results)
            passed_skills = sum(1 for r in skill_results.values() if r.get("passed"))
            
            conn.execute(
                """
                INSERT INTO exam_history 
                (id, domain, stage, exam_type, overall_score, passed, pass_threshold,
                 total_skills, passed_skills, failed_skills, grade_summary,
                 gap_analysis, retry_count, syllabus_modified, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exam.get("exam_id", f"exam_{int(time.time())}"),
                    exam.get("domain", "unknown"),
                    exam.get("stage", "unknown"),
                    exam.get("exam_type", "unknown"),
                    grade_result.get("overall_score", 0.0),
                    1 if grade_result.get("passed") else 0,
                    grade_result.get("pass_threshold", STANDARD_PASS_THRESHOLD),
                    total_skills,
                    passed_skills,
                    json.dumps(grade_result.get("failed_skills", [])),
                    grade_result.get("grade_summary", ""),
                    json.dumps(gap_analysis) if gap_analysis else None,
                    retry_count,
                    1 if syllabus_modified else 0,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.warning(f"Failed to record exam: {e}")
            return False

    def get_retry_count(self, domain: str, stage: str) -> int:
        try:
            from components.db import safe_open_kdb
            conn = safe_open_kdb(self.db_path, timeout=10)
            row = conn.execute(
                "SELECT MAX(retry_count) FROM exam_history WHERE domain = ? AND stage = ?",
                (domain, stage),
            ).fetchone()
            conn.close()
            return row[0] if row and row[0] is not None else 0
        except Exception:
            return 0

    def get_recent_failures(self, limit: int = 20) -> List[Dict]:
        try:
            from components.db import safe_open_kdb
            conn = safe_open_kdb(self.db_path, timeout=10)
            conn.row_factory = lambda cursor, row: dict(
                zip([col[0] for col in cursor.description], row)
            )
            rows = conn.execute(
                "SELECT * FROM exam_history WHERE passed = 0 "
                "ORDER BY completed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
            return rows
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Exam System — Main Orchestrator
# ---------------------------------------------------------------------------
class ExamSystem:
    """Main exam orchestration — integrates generator, grader, gap analyser, history."""

    def __init__(self, db_path: str, analysers: Optional[Dict[str, Any]] = None):
        self.db_path = db_path
        self.analysers = analysers or {}
        self.generator = ExamGenerator(analysers=self.analysers)
        self.grader = ExamGrader(analysers=self.analysers)
        self.gap_analyser = SkillGapAnalyser()
        self.history = ExamHistory(db_path=db_path)

    def run_exam(
        self,
        domain: Dict,
        stage: str,
        curriculum: List[Dict],
        output: Optional[Dict] = None,
    ) -> Dict:
        exam = self.generator.generate_exam(domain, stage, curriculum)
        retry_count = self.history.get_retry_count(domain["domain"], stage)
        
        if output is None:
            return {
                "exam": exam,
                "grade": None,
                "gap_analysis": None,
                "passed": None,
                "retry_count": retry_count,
                "max_retries": MAX_RETRIES_PER_SKILL,
                "status": "exam_generated_awaiting_output",
            }
        
        grade = self.grader.grade_exam(exam, output, self.db_path)
        
        gap_analysis = None
        if not grade["passed"]:
            gap_analysis = self.gap_analyser.analyse_gaps(grade, exam)
            gap_analysis["retry_count"] = retry_count
        
        syllabus_modified = (
            gap_analysis is not None 
            and len(gap_analysis.get("syllabus_modifications", [])) > 0
        )
        self.history.record_exam(
            exam, grade, gap_analysis, retry_count, syllabus_modified
        )
        
        return {
            "exam": exam,
            "grade": grade,
            "gap_analysis": gap_analysis,
            "passed": grade["passed"],
            "retry_count": retry_count,
            "max_retries": MAX_RETRIES_PER_SKILL,
            "status": "passed" if grade["passed"] else "failed",
        }


# ---------------------------------------------------------------------------
# Factory function for easy integration
# ---------------------------------------------------------------------------
def create_exam_system(
    data_path: str = "data",
    image_analyser=None,
    audio_analyser=None,
    speech_analyser=None,
) -> ExamSystem:
    db_path = os.path.join(data_path, "dmai_knowledge.db")
    
    analysers = {}
    if image_analyser:
        analysers["image"] = image_analyser
    if audio_analyser:
        analysers["audio"] = audio_analyser
    if speech_analyser:
        analysers["speech"] = speech_analyser
    
    return ExamSystem(db_path=db_path, analysers=analysers)
