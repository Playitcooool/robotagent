"""
Context File Loader for robot_context.md.

Inspired by Claude Code's claudemd.ts - loads project-level context file.
"""

from pathlib import Path
from typing import Optional
import re
import fnmatch

FRONTMATTER_RE = re.compile(r'^---\n(.*?)\n---\n', re.DOTALL)


class ContextLoader:
    """Load robot_context.md from project root with frontmatter path filtering."""

    def __init__(self, root_dir: str | None = None):
        self._root_dir = Path(root_dir) if root_dir else Path.cwd()
        self._cache: Optional[str] = None

    def load_context(self, current_path: str | None = None) -> str:
        """
        Load robot_context.md with path-based conditional filtering.

        Args:
            current_path: Optional path to match against frontmatter 'paths' globs.

        Returns:
            The body content of robot_context.md (after frontmatter), or empty string.
        """
        if self._cache is not None:
            return self._cache

        context_file = self._root_dir / "robot_context.md"

        try:
            content = context_file.read_text(encoding='utf-8')
        except FileNotFoundError:
            return ""

        frontmatter, body = self._parse_frontmatter(content)

        # Path-based conditional loading
        if frontmatter.get('paths') and current_path:
            if not self._match_paths(frontmatter['paths'], current_path):
                self._cache = ""
                return ""

        self._cache = body
        return body

    def clear_cache(self) -> None:
        """Clear cache to force reload on next call."""
        self._cache = None

    def _parse_frontmatter(self, content: str) -> tuple[dict, str]:
        """Parse YAML frontmatter from content."""
        import yaml
        match = FRONTMATTER_RE.match(content)
        if not match:
            return {}, content
        frontmatter = yaml.safe_load(match.group(1)) or {}
        body = content[match.end():]
        return frontmatter, body

    def _match_paths(self, patterns: list[str], path: str) -> bool:
        """Check if path matches any of the glob patterns."""
        if not patterns:
            return True
        return any(fnmatch.fnmatch(path, p) for p in patterns)

