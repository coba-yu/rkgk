import pytest

from rkgk.domain.repositories.paper import PaperRepositoryError
from rkgk.usecase.load_paper import LoadPaperUseCase
from tests.usecase.fakes import FakePaperRepository, build_paper

PAPER = build_paper(1, "intro\n")


def test_execute_returns_the_paper_the_repository_holds() -> None:
    use_case = LoadPaperUseCase(FakePaperRepository({1: PAPER}))
    assert use_case.execute(1) == PAPER


def test_execute_propagates_the_repository_error_for_an_unknown_paper() -> None:
    use_case = LoadPaperUseCase(FakePaperRepository({1: PAPER}))
    with pytest.raises(PaperRepositoryError, match="paper 2"):
        use_case.execute(2)
