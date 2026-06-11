from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"


def load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(ROOT / ".env")


def run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=check,
    )


def ensure_repo() -> None:
    result = run_git(["rev-parse", "--is-inside-work-tree"], check=False)
    if result.returncode == 0 and result.stdout.strip() == "true":
        return

    init = run_git(["init", "-b", "main"], check=False)
    if init.returncode != 0:
        run_git(["init"])
        run_git(["branch", "-M", "main"], check=False)


def ensure_identity() -> None:
    name = run_git(["config", "--get", "user.name"], check=False)
    email = run_git(["config", "--get", "user.email"], check=False)
    if name.returncode != 0 or not name.stdout.strip():
        run_git(["config", "user.name", "Codex Auto Version"])
    if email.returncode != 0 or not email.stdout.strip():
        run_git(["config", "user.email", "codex@example.local"])


def read_version() -> str | None:
    if not VERSION_FILE.exists():
        return None
    value = VERSION_FILE.read_text(encoding="utf-8").strip()
    return value or None


def bump_version(current: str | None, bump: str) -> str | None:
    if bump == "none":
        return current
    if current is None:
        return "0.1.0"
    try:
        major, minor, patch = [int(part) for part in current.split(".")]
    except ValueError as exc:
        raise SystemExit(f"Invalid VERSION value: {current}") from exc
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise SystemExit(f"Unknown bump type: {bump}")


def update_changelog(version: str | None, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = f"## {version or 'unversioned'} - {timestamp}"
    entry = f"{title}\n\n- {message}\n\n"
    previous = CHANGELOG_FILE.read_text(encoding="utf-8") if CHANGELOG_FILE.exists() else "# Changelog\n\n"
    if previous.startswith("# Changelog\n\n"):
        CHANGELOG_FILE.write_text("# Changelog\n\n" + entry + previous[len("# Changelog\n\n") :], encoding="utf-8")
    else:
        CHANGELOG_FILE.write_text("# Changelog\n\n" + entry + previous, encoding="utf-8")


def has_changes() -> bool:
    result = run_git(["status", "--porcelain"])
    return bool(result.stdout.strip())


def tag_exists(tag: str) -> bool:
    result = run_git(["rev-parse", "-q", "--verify", f"refs/tags/{tag}"], check=False)
    return result.returncode == 0


def current_branch() -> str:
    result = run_git(["branch", "--show-current"], check=False)
    branch = result.stdout.strip()
    if branch:
        return branch
    return "main"


def configure_remote(remote_name: str, remote_url: str) -> None:
    existing = run_git(["remote", "get-url", remote_name], check=False)
    if existing.returncode == 0:
        if existing.stdout.strip() != remote_url:
            run_git(["remote", "set-url", remote_name, remote_url])
        return
    run_git(["remote", "add", remote_name, remote_url])


def push_remote(remote_name: str) -> None:
    branch = current_branch()
    run_git(["push", "-u", remote_name, branch, "--follow-tags"])


def main() -> int:
    load_env_file()
    parser = argparse.ArgumentParser(description="Automate local git version commits and tags.")
    parser.add_argument("--bump", choices=["major", "minor", "patch", "none"], default="patch")
    parser.add_argument("--message", default="")
    parser.add_argument("--no-tag", action="store_true")
    parser.add_argument("--remote-name", default="origin")
    parser.add_argument("--remote-url", default=os.getenv("GITHUB_REMOTE_URL", "").strip())
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_repo()
    ensure_identity()

    current_version = read_version()
    new_version = bump_version(current_version, args.bump)
    message = args.message.strip()
    if not message:
        message = f"Update project version {new_version or current_version or 'unversioned'}"

    if args.dry_run:
        print(f"Repository: {ROOT}")
        print(f"Current version: {current_version or '(none)'}")
        print(f"Next version: {new_version or '(none)'}")
        print(f"Commit message: {message}")
        print(f"Tag: {'disabled' if args.no_tag or not new_version else 'v' + new_version}")
        print(f"Remote: {args.remote_name} {args.remote_url or '(not configured)'}")
        print(f"Push: {'yes' if args.push else 'no'}")
        return 0

    if args.remote_url:
        configure_remote(args.remote_name, args.remote_url)

    if new_version:
        VERSION_FILE.write_text(new_version + "\n", encoding="utf-8")
    update_changelog(new_version, message)

    if not has_changes():
        print("No changes to commit.")
        return 0

    run_git(["add", "--all"])
    if not has_changes():
        print("No staged changes to commit.")
        return 0

    version_prefix = f"v{new_version}: " if new_version else ""
    run_git(["commit", "-m", version_prefix + message])

    if new_version and not args.no_tag:
        tag = f"v{new_version}"
        if tag_exists(tag):
            print(f"Tag {tag} already exists; skipping tag creation.")
        else:
            run_git(["tag", "-a", tag, "-m", version_prefix + message])

    if args.push:
        if not args.remote_url and run_git(["remote", "get-url", args.remote_name], check=False).returncode != 0:
            raise SystemExit(
                f"Remote {args.remote_name} is not configured. Pass --remote-url or set GITHUB_REMOTE_URL."
            )
        push_remote(args.remote_name)

    print("Automated git versioning complete.")
    if new_version:
        print(f"Version: {new_version}")
    print(run_git(["status", "--short"]).stdout.strip() or "Working tree clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
