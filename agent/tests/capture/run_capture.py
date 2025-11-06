#!/usr/bin/env python
"""
Run workflow captures for all test prompts.

This script runs the full research agent workflow with predefined prompts
and captures the state at each agent boundary. The captured states are saved
as JSON fixtures in tests/fixtures/states/{agent_name}/ for use in agent tests.

Usage:
    # Option 1: Use .env.local (recommended)
    cp .env.example .env.local  # if not exists
    # Edit .env.local with your BRAVE_API_KEY
    CAPTURE_STATE=true python tests/capture/run_capture.py

    # Option 2: Pass environment variables
    CAPTURE_STATE=true BRAVE_API_KEY=xxx python tests/capture/run_capture.py

Requirements:
    - CAPTURE_STATE=true (enables state capture)
    - BRAVE_API_KEY (from .env.local or environment)
    - Databricks CLI configured (for LLM endpoints)

Output:
    Fixtures saved to: tests/fixtures/states/reporter/{prompt_id}_{phase}.json

Example fixture structure:
    tests/fixtures/states/
        reporter/
            simple_fact_before.json
            simple_fact_after.json
            tax_research_before.json
            tax_research_after.json
        planner/
            ...
        researcher/
            ...
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Load environment variables from .env.local first
from dotenv import load_dotenv

# Find .env.local or .env by iterating up the directory tree
def find_env_file(start_path: Path) -> tuple[Path | None, Path]:
    """
    Search upward for .env.local or .env files.

    Returns:
        (env_file_path, root_dir) where env_file_path can be None if not found
    """
    current = start_path
    # Search up to 5 levels (reasonable limit)
    for _ in range(5):
        env_local = current / ".env.local"
        env_file = current / ".env"

        # Prefer .env.local over .env
        if env_local.exists():
            return env_local, current
        if env_file.exists():
            return env_file, current

        # Stop at filesystem root
        if current.parent == current:
            break
        current = current.parent

    # If nothing found, return the start path as root
    return None, start_path

# Search for .env file
start_path = Path(__file__).parent
env_file_found, agent_root = find_env_file(start_path)

# Load environment file if found
if env_file_found:
    load_dotenv(env_file_found, override=True)
    print(f"✅ Loaded environment from: {env_file_found}")

# Add src to path (assuming src is at agent_root)
sys.path.insert(0, str(agent_root / "src"))

from deep_research_agent.databricks_compatible_agent import DatabricksCompatibleAgent
from deep_research_agent.core.state_capture import state_capture
from mlflow.types.responses import ResponsesAgentRequest


# Test prompts with expected characteristics
# These are derived from real-world test cases to ensure representative coverage
PROMPTS: List[Tuple[str, str, str]] = [
    # (prompt_id, prompt_text, category)

    # ========================================================================
    # Basic Research Queries (Simple factual questions)
    # ========================================================================
    ("simple_fact_python", "What is Python programming language?", "basic"),
    ("simple_fact_ml", "What is machine learning and how does it work?", "basic"),
    ("historical_fact", "When was the Declaration of Independence signed?", "basic"),

    # ========================================================================
    # Complex Analysis Queries (Multi-faceted, requires synthesis)
    # ========================================================================

    # Real-world complex tax comparison (from actual test case)
    ("tax_comparison_europe",
     """I want a rigorous, apples-to-apples comparison of after-tax finances across Spain, France,
     United Kingdom, Switzerland (low-tax canton such as Zug), Germany for two family setups:
     1) married couple without children
     2) married couple with one child (3 years old)""",
     "complex"),

    # COMPACT tax comparison - 2 countries, 2 scenarios (faster debugging)
    ("tax_comparison_compact",
     """I want a rigorous comparison of after-tax finances between Spain and Poland for two family setups:
	1.	Single: €150,000 annual gross salary + €100,000 annual RSUs
	2.	Married, no child: same primary earner (€150k + €100k RSUs) + spouse with €100,000 salary

Please compute for tax year 2025 (or latest available rules):
	•	Net take-home after income tax and mandatory employee social contributions
	•	RSU treatment: assume standard RSUs taxed at vest as employment income, then capital gains on sale
	•	Effective tax rate = (total taxes + social contributions) ÷ total gross income (including RSUs)
	•	Rent: monthly market rent for Madrid and Warsaw (2-3 bedroom upper-middle-class apartment)
	•	Disposable income = Net take-home – annualized rent

Output
	1.	A comparative table (rows = countries, columns = scenarios) with: Net take-home, Effective tax rate, Annual rent, Disposable income
	2.	A brief narrative (≤200 words) highlighting key differences
	3.	List your assumptions (RSU timing, local taxes, etc.)""",
     "complex"),

    # VERY BIG real-world tax comparison - 7 countries, 3 scenarios, extremely detailed
    ("tax_comparison_europe_very_big",
     """I want a rigorous, apples-to-apples comparison of after-tax finances across Spain, France, United Kingdom, Switzerland (low-tax canton such as Zug), Germany, Poland, and Bulgaria for three family setups:
	1.	Single: €150,000 annual gross salary + €100,000 annual RSUs
	2.	Married, no child: same primary earner (€150k + €100k RSUs) + spouse with €100,000 salary
	3.	Married, 1 child: same as #2, and include all child benefits/credits and net daycare costs

Please compute for tax year 2025 (or latest available rules):
	•	Net take-home after income tax and mandatory employee social contributions (and any local surtaxes).
	•	RSU treatment: assume standard RSUs (no special tax-advantaged schemes). If rules differ (taxed at vest vs sale, ordinary income vs capital gains), state assumptions you use, and model typical case (tax at vest as employment income; subsequent capital gains on sale).
	•	Effective tax rate = total personal taxes + employee social contributions ÷ total gross income considered (include RSUs per the assumption).
	•	Rent: monthly market rent for an upper-middle-class neighborhood:
	•	Madrid, Paris, London, Zug/City of Zug (CH), Frankfurt, Warsaw, Sofia
	•	Property: good-quality 2–3 bedroom apartment suitable for a family. Use recent data; pick a clear, defensible source.
	•	Daycare (scenario #3 only): typical full-time cost for 1 toddler (public vs private as typical for locals), net of subsidies/allowances, and note eligibility conditions.
	•	Family benefits/allowances/credits (scenario #3): child benefit, tax credits, family quotient, etc. Reflect any high-income clawbacks.
	•	Disposable income = Net take-home – annualized rent – daycare + family benefits (cash)
	•	Convert all figures to EUR. For GBP/CHF, use latest ECB rates and cite date.

Output
	1.	A single comparative table (rows = countries, columns = the three scenarios) with:
	•	Net take-home, Effective tax rate, Annual rent, Daycare (if applicable), Family benefits (cash), Disposable income.
	2.	A short narrative (≤300 words) highlighting: lowest/highest effective tax, most/least family-friendly after benefits and daycare, and the rent outliers.
	3.	A transparent appendix listing sources and assumptions (e.g., RSU timing, local surtaxes, church tax assumed no in Germany, UK child benefit high-income charge if applicable, etc.).
	4.	Wherever rules are ambiguous, pick the most common case for residents and state the assumption""",
     "very_complex"),

    # Technology comparison
    ("quantum_vs_classical",
     "Compare quantum computing and classical computing in terms of their applications and limitations",
     "complex"),

    # Financial planning
    ("retirement_comparison",
     "Compare tax optimization strategies for 401k vs Roth IRA for high earners in California in 2024",
     "complex"),

    # ========================================================================
    # Current Events & Recent Developments
    # ========================================================================
    ("ai_developments_2024",
     "What are the latest developments in artificial intelligence in 2024?",
     "current_events"),

    # ========================================================================
    # Domain-Specific Queries
    # ========================================================================
    ("medical_query",
     "What are the symptoms and treatments for Type 2 diabetes?",
     "domain_specific"),

    ("tech_cryptography",
     "Explain quantum computing algorithms and their advantages over classical computing for cryptography",
     "domain_specific"),

    # ========================================================================
    # Edge Cases (Should not trigger full research)
    # ========================================================================
    ("greeting", "Hello, how are you?", "edge_case"),
    ("empty_input", "", "edge_case"),
    ("inappropriate", "How can I hack someone's email?", "edge_case"),

    # Add more prompts as needed based on real test cases
]


# ============================================================================
# CLI Helper Functions
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Capture workflow states for testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture all prompts (default)
  CAPTURE_STATE=true python run_capture.py

  # Capture specific prompt
  CAPTURE_STATE=true python run_capture.py --prompt-id simple_fact_python

  # Capture multiple prompts
  CAPTURE_STATE=true python run_capture.py --prompt-id simple_fact_python,tax_comparison_europe

  # Capture by category
  CAPTURE_STATE=true python run_capture.py --category basic

  # List available prompts
  python run_capture.py --list

  # Dry run (show what would be captured)
  CAPTURE_STATE=true python run_capture.py --category complex --dry-run

  # Skip already captured prompts
  CAPTURE_STATE=true python run_capture.py --skip-existing
        """
    )

    parser.add_argument(
        '--prompt-id',
        type=str,
        help='Comma-separated prompt IDs to capture (e.g., "simple_fact_python,tax_comparison_europe")'
    )

    parser.add_argument(
        '--category',
        type=str,
        choices=['basic', 'complex', 'very_complex', 'edge_case', 'current_events', 'domain_specific'],
        help='Filter prompts by category'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available prompts and exit'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be captured without actually running'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip prompts that already have captured fixtures'
    )

    return parser.parse_args()


def print_available_prompts() -> None:
    """Print all available prompts grouped by category."""
    # Group prompts by category
    by_category = defaultdict(list)
    for prompt_id, prompt_text, category in PROMPTS:
        by_category[category].append((prompt_id, prompt_text))

    print(f"\n{'='*80}")
    print(f"📋 AVAILABLE PROMPTS ({len(PROMPTS)} total)")
    print(f"{'='*80}\n")

    for category in sorted(by_category.keys()):
        prompts = by_category[category]
        print(f"{category.upper()} ({len(prompts)} prompts):")
        for prompt_id, prompt_text in prompts:
            # Truncate long prompts
            preview = prompt_text[:70] + "..." if len(prompt_text) > 70 else prompt_text
            preview = preview.replace("\n", " ")
            print(f"  • {prompt_id:30s} - {preview}")
        print()

    print(f"{'='*80}\n")


def filter_prompts(
    all_prompts: List[Tuple[str, str, str]],
    prompt_ids: Optional[List[str]] = None,
    category: Optional[str] = None
) -> List[Tuple[str, str, str]]:
    """
    Filter prompts based on criteria.

    Args:
        all_prompts: List of (prompt_id, prompt_text, category) tuples
        prompt_ids: List of specific prompt IDs to include
        category: Category to filter by

    Returns:
        Filtered list of prompts
    """
    filtered = all_prompts

    # Filter by specific IDs
    if prompt_ids:
        filtered = [p for p in filtered if p[0] in prompt_ids]

        # Check for invalid IDs
        found_ids = {p[0] for p in filtered}
        invalid_ids = set(prompt_ids) - found_ids
        if invalid_ids:
            print(f"⚠️  Warning: Unknown prompt IDs: {', '.join(invalid_ids)}")
            print(f"   Run with --list to see available prompts\n")

    # Filter by category
    if category:
        filtered = [p for p in filtered if p[2] == category]

    return filtered


def check_existing_fixtures(prompts: List[Tuple[str, str, str]]) -> List[str]:
    """
    Check which prompts already have captured fixtures.

    Args:
        prompts: List of prompts to check

    Returns:
        List of prompt IDs that already have fixtures
    """
    existing = []
    fixtures_dir = Path("tests/fixtures/states/reporter")

    for prompt_id, _, _ in prompts:
        before_fixture = fixtures_dir / f"{prompt_id}_before.json"
        after_fixture = fixtures_dir / f"{prompt_id}_after.json"

        if before_fixture.exists() and after_fixture.exists():
            existing.append(prompt_id)

    return existing


async def capture_one_prompt(agent: DatabricksCompatibleAgent, prompt_id: str, prompt: str, category: str) -> Dict[str, any]:
    """
    Capture states for one prompt.

    Args:
        agent: Initialized DatabricksCompatibleAgent
        prompt_id: Unique identifier for this prompt
        prompt: The actual user prompt text
        category: Category of the prompt (basic, complex, edge_case, etc.)

    Returns:
        Dict with capture results (success, errors, states_captured)
    """
    print(f"\n{'='*80}")
    print(f"📝 Prompt ID: {prompt_id}")
    print(f"📝 Category: {category}")
    print(f"📝 Content: {prompt[:100]}..." if len(prompt) > 100 else f"📝 Content: {prompt}")
    print(f"{'='*80}\n")

    # Set context for state capture
    state_capture.set_prompt_context(prompt_id, prompt)

    result = {
        "prompt_id": prompt_id,
        "category": category,
        "success": False,
        "error": None,
        "states_captured": 0
    }

    try:
        # Create request
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": prompt}]
        )

        # Run agent (this will trigger captures inside workflow nodes)
        print(f"🚀 Executing workflow for '{prompt_id}'...", flush=True)
        # Use threading as originally intended (predict() needs it for async generator)
        response = await asyncio.to_thread(agent.predict, request)

        # Detect if workflow failed (agent returns error in response)
        if hasattr(response, 'custom_outputs') and 'error' in response.custom_outputs:
            error_info = response.custom_outputs['error']
            error_msg = error_info.get('user_message', error_info.get('message', str(error_info)))
            raise RuntimeError(f"Workflow execution failed: {error_msg}")

        # Check captured states
        states_captured = len(state_capture.captured_states)
        result["states_captured"] = states_captured
        result["success"] = True

        print(f"\n✅ Success: Captured {states_captured} states for '{prompt_id}'")
        print(f"   States: {list(state_capture.captured_states.keys())}")

    except Exception as e:
        result["error"] = str(e)
        print(f"\n❌ Error for '{prompt_id}': {e}")
        import traceback
        traceback.print_exc()

    return result


async def main():
    """Run captures with CLI support."""

    # Parse CLI arguments
    args = parse_args()

    # Handle --list mode (doesn't require CAPTURE_STATE)
    if args.list:
        print_available_prompts()
        return

    # Validate environment for actual capture (skip for --dry-run)
    if not args.dry_run and os.getenv("CAPTURE_STATE", "false").lower() != "true":
        print("❌ Error: CAPTURE_STATE environment variable must be set to 'true'")
        print("   Usage: CAPTURE_STATE=true python tests/capture/run_capture.py")
        print("   Or use --list to see available prompts without capturing")
        print("   Or use --dry-run to preview without capturing")
        sys.exit(1)

    if not os.getenv("BRAVE_API_KEY"):
        print("⚠️  Warning: BRAVE_API_KEY not set - search may fail")
        print("   Option 1: Add to .env.local file (recommended)")
        print("   Option 2: Set with: export BRAVE_API_KEY=your_key")
        print("   Option 3: Pass inline: BRAVE_API_KEY=xxx python tests/capture/run_capture.py")

    # Filter prompts based on CLI arguments
    prompt_ids = args.prompt_id.split(',') if args.prompt_id else None
    prompts_to_run = filter_prompts(PROMPTS, prompt_ids=prompt_ids, category=args.category)

    if not prompts_to_run:
        print("❌ Error: No prompts match the specified filters")
        print("   Use --list to see available prompts")
        sys.exit(1)

    # Check for existing fixtures if --skip-existing
    if args.skip_existing:
        existing = check_existing_fixtures(prompts_to_run)
        if existing:
            print(f"\n⏭️  Skipping {len(existing)} prompts with existing fixtures:")
            for prompt_id in existing:
                print(f"   • {prompt_id}")
            prompts_to_run = [p for p in prompts_to_run if p[0] not in existing]
            print()

    if not prompts_to_run:
        print("✅ All selected prompts already have fixtures")
        return

    # Handle --dry-run mode
    if args.dry_run:
        print(f"\n{'='*80}")
        print(f"🔍 DRY RUN - No actual capture will be performed")
        print(f"{'='*80}\n")
        print(f"Would capture {len(prompts_to_run)} prompts:\n")

        by_category = defaultdict(list)
        for prompt_id, prompt_text, category in prompts_to_run:
            by_category[category].append(prompt_id)

        for category in sorted(by_category.keys()):
            prompts = by_category[category]
            print(f"{category.upper()} ({len(prompts)}):")
            for prompt_id in prompts:
                print(f"  • {prompt_id}")
            print()

        print(f"{'='*80}\n")
        return

    # Actual capture mode
    print(f"\n{'='*80}")
    print(f"🎥 STATE CAPTURE MODE ENABLED")
    print(f"{'='*80}")
    print(f"Capturing {len(prompts_to_run)} of {len(PROMPTS)} prompts")
    if args.prompt_id:
        print(f"Filter: prompt IDs = {args.prompt_id}")
    if args.category:
        print(f"Filter: category = {args.category}")
    print(f"Output directory: tests/fixtures/states/")
    print(f"{'='*80}\n")

    # Get config path
    test_dir = Path(__file__).parent.parent.parent  # agent/ directory
    config_path = test_dir / "conf" / "base.yaml"

    if not config_path.exists():
        print(f"❌ Error: Config file not found at {config_path}")
        sys.exit(1)

    print(f"Using config: {config_path}\n")

    # Run captures for filtered prompts
    # CRITICAL FIX: Create FRESH agent for EACH prompt to prevent state accumulation
    # Previous bug: Reusing same agent caused MemorySaver to accumulate observations across prompts
    # (First prompt: 1K obs, Second prompt: loads 1K + adds 1K = 2K, etc.)
    results = []
    for i, (prompt_id, prompt, category) in enumerate(prompts_to_run, 1):
        print(f"\n[{i}/{len(prompts_to_run)}] Processing: {prompt_id}")

        # Create fresh agent with clean state for THIS prompt
        print("🔧 Initializing fresh agent...")
        try:
            agent = DatabricksCompatibleAgent(yaml_path=str(config_path))
            print("✅ Fresh agent initialized\n")
        except Exception as e:
            print(f"❌ Error initializing agent: {e}")
            results.append({
                "prompt_id": prompt_id,
                "success": False,
                "error": str(e),
                "states_captured": 0
            })
            continue

        result = await capture_one_prompt(agent, prompt_id, prompt, category)
        results.append(result)

        # Small delay between prompts to avoid rate limiting
        if i < len(prompts_to_run):
            print("\n⏳ Waiting 5 seconds before next prompt...")
            await asyncio.sleep(5)

    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 CAPTURE SUMMARY")
    print(f"{'='*80}")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\nTotal prompts run: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")

    if successful:
        print(f"\n✅ Successful captures:")
        for r in successful:
            print(f"   - {r['prompt_id']}: {r['states_captured']} states")

    if failed:
        print(f"\n❌ Failed captures:")
        for r in failed:
            print(f"   - {r['prompt_id']}: {r['error']}")

    total_states = sum(r["states_captured"] for r in successful)
    print(f"\n📁 Total states captured: {total_states}")
    print(f"📂 Fixtures location: tests/fixtures/states/")

    print(f"\n{'='*80}")
    print(f"✅ Capture complete!")
    print(f"{'='*80}\n")

    # Exit with error if any failed
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
