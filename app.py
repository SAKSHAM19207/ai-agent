"""Compatibility entrypoint for validators expecting a top-level app.py."""

__all__ = ["app", "main"]


def __getattr__(name: str):
    if name in {"app", "main"}:
        from server.app import app, main

        return {"app": app, "main": main}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    from server.app import main

    main()
