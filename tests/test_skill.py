from pathlib import Path

from rkgk.domain.models.vocabulary import describe_vocabulary

REPOSITORY_ROOT = Path(__file__).parent.parent
AGENT_SKILL_PATH = REPOSITORY_ROOT / ".agents" / "skills" / "extract" / "SKILL.md"
CLAUDE_SKILL_PATH = REPOSITORY_ROOT / ".claude" / "skills" / "extract" / "SKILL.md"


def read_agent_skill() -> str:
    return AGENT_SKILL_PATH.read_text(encoding="utf-8")


def test_the_skill_declares_its_name_in_the_frontmatter() -> None:
    text = read_agent_skill()
    assert text.startswith("---\n")
    assert "\nname: rkgk-extract\n" in text
    assert "\ndescription: " in text


def test_the_skill_embeds_the_vocabulary_verbatim() -> None:
    assert describe_vocabulary() in read_agent_skill()


def test_the_skill_tells_the_agent_which_commands_to_run() -> None:
    text = read_agent_skill()
    assert "uv run rkgk extract schema" in text
    assert "uv run rkgk extract validate" in text
    assert "uv run rkgk extract save" in text


def test_the_copy_for_claude_code_is_identical_to_the_agent_skill() -> None:
    assert CLAUDE_SKILL_PATH.read_bytes() == AGENT_SKILL_PATH.read_bytes()
