"""
LLM-based benchmark analysis - no hardcoded conclusions.
Everything is decided by the LLM based on actual results.
"""
import os
import json
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List


class BenchmarkAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def analyze_results(self, results: Dict[str, List], tools_tested: List[str], questions: List[str]) -> Dict[str, Any]:
        """Generate comprehensive analysis using LLM - no hardcoded recommendations."""
        
        # Prepare benchmark data
        analysis_data = self._prepare_data(results, tools_tested, questions)
        
        # Get LLM analysis
        prompt = self._create_analysis_prompt(analysis_data)
        
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        
        # Parse the response
        analysis_text = response.content
        
        # Extract structured sections
        return {
            "raw_analysis": analysis_text,
            "benchmark_data": analysis_data
        }
    
    def _prepare_data(self, results: Dict[str, List], tools_tested: List[str], questions: List[str]) -> Dict:
        """Prepare data for LLM analysis."""
        
        analysis_data = {
            "tools_tested": tools_tested,
            "questions_tested": questions,
            "num_questions": len(questions),
            "results_by_tool": {}
        }
        
        for tool_name, runs in results.items():
            if tool_name in tools_tested and runs and len(runs) > 0:
                # Filter out error runs
                valid_runs = [r for r in runs if "error" not in r]
                
                if valid_runs:
                    analysis_data["results_by_tool"][tool_name] = {
                        "total_runs": len(valid_runs),
                        "avg_latency_ms": sum(r["latency_ms"] for r in valid_runs) / len(valid_runs),
                        "avg_total_time_ms": sum(r["total_time"] for r in valid_runs) / len(valid_runs),
                        "min_latency_ms": min(r["latency_ms"] for r in valid_runs),
                        "max_latency_ms": max(r["latency_ms"] for r in valid_runs),
                        "all_tools_used": list(set([tool for r in valid_runs for tool in r.get("tools_used", [])])),
                        "metadata": valid_runs[0].get("metadata", {}),
                        "sample_outputs": [r["output"][:200] + "..." if len(r["output"]) > 200 else r["output"] for r in valid_runs[:2]]
                    }
        
        return analysis_data
    
    def _create_analysis_prompt(self, data: Dict) -> str:
        """Create the prompt for LLM analysis."""
        
        return f"""You are an expert in LLM observability tools. Analyze the following benchmark results and provide a comprehensive, unbiased analysis.

BENCHMARK DATA:
{json.dumps(data, indent=2)}

Your task is to analyze these results and provide:

1. **Performance Analysis**
   - Compare the latency and total time across tools
   - Identify which tool performed best and why
   - Explain any significant performance differences
   - Note if performance is similar across tools

2. **Feature & Integration Analysis**
   - Based on the metadata provided, analyze the setup complexity
   - Evaluate the tracing mechanisms (automatic vs callback vs manual)
   - Assess the integration approach for each tool

3. **Overall Recommendation**
   - Based ONLY on the actual benchmark results and metadata
   - Provide ONE clear recommendation for which tool to use
   - Explain your reasoning based on the data
   - DO NOT provide "if-then" conditional recommendations
   - DO NOT say "choose X if you need Y, choose Z if you need W"
   - Instead, make a definitive choice and explain why

4. **Key Insights**
   - What did the benchmark reveal about these tools?
   - Are there any surprising findings?
   - What are the practical implications?

IMPORTANT INSTRUCTIONS:
- Base your analysis ONLY on the actual data provided
- Do NOT make assumptions about features not evident in the results
- Provide ONE clear winner/recommendation, not multiple options
- Be specific and data-driven
- Keep your analysis concise but thorough
- Format your response in clean markdown

Provide your complete analysis now:"""
    
    def generate_comparison_matrix(self, results: Dict[str, List], tools_tested: List[str]) -> str:
        """Generate feature comparison using LLM."""
        
        # Prepare metadata from results
        metadata = {}
        for tool_name, runs in results.items():
            if tool_name in tools_tested and runs and "error" not in runs[0]:
                metadata[tool_name] = runs[0].get("metadata", {})
        
        prompt = f"""Based on the following metadata from benchmark runs, create a feature comparison table.

METADATA:
{json.dumps(metadata, indent=2)}

Create a markdown table comparing these tools across relevant features.
Include features like: Integration Type, Setup Complexity, Tracing Method, and any other relevant aspects you can infer from the metadata.

Return ONLY the markdown table, nothing else."""

        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return response.content
    
    def generate_summary_metrics(self, results: Dict[str, List], tools_tested: List[str]) -> List[Dict[str, str]]:
        """Generate summary metric cards using LLM."""
        
        analysis_data = self._prepare_data(results, tools_tested, [])
        
        prompt = f"""Based on the benchmark results below, identify the top 3 most important insights and present them as metric cards.

RESULTS:
{json.dumps(analysis_data["results_by_tool"], indent=2)}

For each insight, provide:
- label: A short metric label (e.g., "Fastest Tool", "Best Integration")
- value: The tool name or key finding
- description: Brief explanation (one sentence)

Return your response as a JSON array of objects with these three fields.
Example format:
[
  {{"label": "Fastest Tool", "value": "LangSmith", "description": "Lowest average latency at 250ms"}},
  {{"label": "Best Integration", "value": "Langfuse", "description": "Most flexible callback-based system"}}
]

Return ONLY the JSON array, no other text."""

        response = self.llm.invoke([{"role": "user", "content": prompt}])
        
        try:
            # Extract JSON from response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        except:
            # Fallback if parsing fails
            return [
                {"label": "Analysis Complete", "value": str(len(analysis_data["results_by_tool"])), "description": "Tools benchmarked successfully"}
            ]