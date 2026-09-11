"""Port that the domain needs to read and write the concept normalization from the outside world."""

from typing import Protocol

from rkgk.domain.models.concept_normalization import ConceptNormalization
from rkgk.domain.repositories.base import RepositoryError


class ConceptNormalizationRepositoryError(RepositoryError):
    """Base of every failure the concept normalization repository reports.

    There is no paper id here, unlike the extraction repository, because a normalization covers every paper at once.
    """


class ConceptNormalizationNotFoundError(ConceptNormalizationRepositoryError):
    """The concepts have not been normalized yet, so running the normalization is the remedy."""


class ConceptNormalizationArtifactUnreadableError(ConceptNormalizationRepositoryError):
    """The artifact exists but cannot be read or written, so the environment is the remedy."""


class ConceptNormalizationArtifactInvalidError(ConceptNormalizationRepositoryError):
    """The artifact was read but is not a valid normalization, so a new normalization run is the remedy."""


class ConceptNormalizationRepository(Protocol):
    def save(self, normalization: ConceptNormalization) -> None: ...

    def find(self) -> ConceptNormalization: ...
