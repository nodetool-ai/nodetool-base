class Environment:
    """Minimal environment helper.

    The production gate is always false in tests so local-only nodes can run.
    """

    @staticmethod
    def is_production() -> bool:
        return False

