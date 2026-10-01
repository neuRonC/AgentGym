"""ALFWorld service package with an import-side-effect-free entrypoint."""


def launch() -> None:
    from .launch import launch as run

    run()

__all__ = ["launch"]
