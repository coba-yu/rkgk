"""Ports that the domain needs from the outside world.

The domain states what it wants to read, not where the bytes live, so an implementation may back these calls
with files, S3, or a database without the use cases noticing.
"""


class RepositoryError(Exception):
    """Constructor shared by every repository error family.

    Subclasses tell the caller what kind of failure it is, not which step of an implementation hit it,
    so a file-backed and an S3-backed repository raise the same classes.
    `location` is an attribute so a caller can render it without parsing the message.
    """

    def __init__(self, message: str, *, location: str | None = None) -> None:
        super().__init__(message)
        self.location = location
