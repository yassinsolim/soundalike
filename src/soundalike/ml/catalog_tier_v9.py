"""Collect secret-safe evidence for the linked Vercel project's memory tier."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import requests

from .catalog_list_gold_v9 import canonical_bytes, sha256_bytes, sha256_path, write_json


DOC_URL = "https://vercel.com/docs/functions/limitations"
GITHUB_DEPLOYMENTS_URL = (
    "https://api.github.com/repos/yassinsolim/soundalike/deployments?per_page=20"
)


def _safe_cli(command: Sequence[str]) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
    except OSError as exc:
        return {
            "command": " ".join(command),
            "returncode": None,
            "status": "failed",
            "reason": type(exc).__name__,
        }
    combined = "\n".join((result.stdout, result.stderr)).strip()
    reason = next(
        (
            line.strip() for line in combined.splitlines()
            if "credentials" in line.casefold() or "token" in line.casefold()
        ),
        "completed" if result.returncode == 0 else "command failed",
    )
    return {
        "command": " ".join(command),
        "returncode": int(result.returncode),
        "status": "passed" if result.returncode == 0 else "failed",
        "reason": reason,
    }


def collect_tier_evidence(
    *,
    project_path: Any,
    resource_path: Any,
) -> Dict[str, Any]:
    project = json.loads(Path(project_path).read_text(encoding="utf-8"))
    resource = json.loads(Path(resource_path).read_text(encoding="utf-8"))
    credential_names = sorted(
        name for name in os.environ if name.startswith(("VERCEL", "NOW_"))
    )
    cli = [
        _safe_cli(["vercel.cmd", "whoami"]),
        _safe_cli([
            "vercel.cmd", "project", "inspect", project["projectName"], "--yes"
        ]),
    ]
    project_url = (
        f"https://api.vercel.com/v9/projects/{project['projectId']}"
        f"?teamId={project['orgId']}"
    )
    try:
        api_response = requests.get(project_url, timeout=20)
        api_attempt = {
            "method": "GET",
            "endpoint": "/v9/projects/<linked-project>?teamId=<linked-team>",
            "status": int(api_response.status_code),
            "authenticated": False,
            "outcome": (
                "project metadata returned"
                if api_response.status_code == 200
                else "forbidden without credentials"
            ),
        }
    except Exception as exc:
        api_attempt = {
            "method": "GET",
            "endpoint": "/v9/projects/<linked-project>?teamId=<linked-team>",
            "status": None,
            "authenticated": False,
            "outcome": type(exc).__name__,
        }
    try:
        response = requests.get(GITHUB_DEPLOYMENTS_URL, timeout=20)
        deployments = response.json() if response.status_code == 200 else []
        production = next(
            (
                item for item in deployments
                if item.get("environment") == "Production"
            ),
            None,
        )
        github = {
            "status": int(response.status_code),
            "vercel_bot_deployment_found": bool(
                production
                and production.get("creator", {}).get("login") == "vercel[bot]"
            ),
            "latest_public_production": (
                {
                    "created_at": production.get("created_at"),
                    "sha": production.get("sha"),
                    "creator": production.get("creator", {}).get("login"),
                }
                if production else None
            ),
            "tier_or_memory_exposed": False,
        }
    except Exception as exc:
        github = {
            "status": None,
            "error_class": type(exc).__name__,
            "tier_or_memory_exposed": False,
        }
    evidence: Dict[str, Any] = {
        "schema_version": 9,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "linked_project": {
            "projectName": project["projectName"],
            "projectId": project["projectId"],
            "orgId": project["orgId"],
            "metadata_sha256": sha256_path(project_path),
        },
        "credential_environment_variable_names": credential_names,
        "credentials_or_token_values_recorded": False,
        "cli_attempts": cli,
        "project_api_attempt": api_attempt,
        "github_public_deployment_evidence": github,
        "official_documentation": {
            "url": DOC_URL,
            "last_updated": "2026-07-01",
            "documented_memory_limits_bytes": {
                "Hobby_default_and_maximum": 2 * 1024**3,
                "Pro_Enterprise_default": 2 * 1024**3,
                "Pro_Enterprise_maximum": 4 * 1024**3,
            },
            "project_specific_tier_exposed": False,
        },
        "candidate_resources": {
            "unchanged_asset_set_from_v8_measurement": True,
            "resource_artifact": str(Path(resource_path)),
            "resource_artifact_sha256": sha256_path(resource_path),
            "peak_rss_bytes": int(resource["peak_rss_bytes"]),
            "load_seconds": float(resource["load_seconds"]),
            "warm_p95_ms": float(resource["warm_latency"]["p95_ms"]),
            "graph_bytes": int(resource["assets"]["graph"]["bytes"]),
            "index_bytes": int(resource["assets"]["index"]["bytes"]),
            "v9_adds_runtime_arrays": False,
        },
        "project_tier": "unknown",
        "actual_memory_limit_bytes": None,
        "tier_verified": False,
        "passed": False,
        "fail_closed_reason": (
            "Linked project identity and real Vercel deployment are verified, "
            "but local metadata and public GitHub deployments expose no plan; "
            "CLI has no credentials and the project API returns 403. Official "
            "limits differ by plan, so no plan or memory limit is assumed."
        ),
    }
    evidence["content_sha256"] = sha256_bytes(canonical_bytes(evidence))
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="webapp/.vercel/project.json")
    parser.add_argument(
        "--resource",
        default=(
            ".goals/human-quality-recommendations/artifacts/"
            "catalog-gated-resources-v8.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            ".goals/human-quality-recommendations/artifacts/"
            "catalog-vercel-tier-evidence-v9.json"
        ),
    )
    args = parser.parse_args(argv)
    evidence = collect_tier_evidence(
        project_path=args.project, resource_path=args.resource
    )
    write_json(args.output, evidence)
    print(json.dumps({
        "tier_verified": evidence["tier_verified"],
        "passed": evidence["passed"],
        "project_tier": evidence["project_tier"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
