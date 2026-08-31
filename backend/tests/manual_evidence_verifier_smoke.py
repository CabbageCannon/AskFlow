import asyncio

from dotenv import load_dotenv

import open_deep_research.deep_researcher as dr


load_dotenv()


CASES = {
    "sufficient": {
        "research_brief": (
            "Compare OrchidFlow and RiverAgent in terms of "
            "architecture and deployment."
        ),
        "notes": [
            (
                "OrchidFlow uses an explicit graph-based workflow architecture "
                "according to its official documentation [1]."
            ),
            (
                "RiverAgent uses a conversation-oriented agent architecture "
                "according to its official documentation [2]."
            ),
            (
                "Both projects provide official deployment instructions "
                "for container-based environments [1][2]."
            ),
        ],
        "raw_notes": [
            (
                "--- SOURCE 1: OrchidFlow Official Documentation ---\n"
                "URL: https://docs.orchidflow.dev/architecture\n"
                "Source type: official project documentation.\n"
                "OrchidFlow defines workflows as explicit directed graphs "
                "and documents container deployment."
            ),
            (
                "--- SOURCE 2: RiverAgent Official Documentation ---\n"
                "URL: https://docs.riveragent.dev/deployment\n"
                "Source type: official project documentation.\n"
                "RiverAgent describes conversation-oriented agent coordination "
                "and documents container deployment."
            ),
        ],
    },

    "insufficient": {
        "research_brief": (
            "Compare OrchidFlow and RiverAgent in terms of "
            "architecture, deployment, and performance."
        ),
        "notes": [
            (
                "OrchidFlow uses a graph-based architecture and appears "
                "to perform better than RiverAgent."
            ),
        ],
        "raw_notes": [
            (
                "--- SOURCE 1: Personal Developer Blog ---\n"
                "URL: https://example-dev-blog.dev/orchidflow-review\n"
                "Source type: personal blog.\n"
                "The author says OrchidFlow felt faster in one personal project."
            ),
        ],
    },

    "conflict": {
        "research_brief": (
            "Determine the typical throughput of OrchidFlow "
            "under the standard benchmark configuration."
        ),
        "notes": [
            (
                "Benchmark Lab A reports OrchidFlow throughput "
                "of 100 requests per second [1]."
            ),
            (
                "Benchmark Lab B reports OrchidFlow throughput "
                "of 60 requests per second [2]."
            ),
        ],
        "raw_notes": [
            (
                "--- SOURCE 1: Benchmark Lab A ---\n"
                "URL: https://benchmark-a.dev/orchidflow\n"
                "Source type: independent benchmark organization.\n"
                "OrchidFlow version: 2.0.\n"
                "Hardware: 8 CPU cores, 16 GB RAM.\n"
                "Concurrency: 20.\n"
                "Measured throughput: 100 requests per second."
            ),
            (
                "--- SOURCE 2: Benchmark Lab B ---\n"
                "URL: https://benchmark-b.dev/orchidflow\n"
                "Source type: independent benchmark organization.\n"
                "OrchidFlow version: 2.0.\n"
                "Hardware: 8 CPU cores, 16 GB RAM.\n"
                "Concurrency: 20.\n"
                "Measured throughput: 60 requests per second."
            ),
        ],
    },
}


async def run_case(name, state):
    print("\n" + "=" * 80)
    print(f"CASE: {name}")
    print("=" * 80)

    result = await dr.evidence_verifier(
        state,
        {},
    )

    verification = result["verification_result"]

    print(
        verification.model_dump_json(
            indent=2,
        )
    )

    return verification


async def main():
    results = {}

    for name, state in CASES.items():
        results[name] = await run_case(
            name,
            state,
        )

    print("\n" + "=" * 80)
    print("SMOKE CHECK")
    print("=" * 80)

    sufficient = results["sufficient"]
    insufficient = results["insufficient"]
    conflict = results["conflict"]

    print(
        "sufficient:",
        sufficient.evidence_sufficient,
        sufficient.coverage_score,
        sufficient.credibility_score,
    )

    print(
        "insufficient:",
        insufficient.evidence_sufficient,
        insufficient.coverage_score,
        insufficient.credibility_score,
        len(insufficient.missing_evidence),
    )

    print(
        "conflict:",
        len(conflict.conflicts),
        conflict.credibility_score,
    )


if __name__ == "__main__":
    asyncio.run(main())