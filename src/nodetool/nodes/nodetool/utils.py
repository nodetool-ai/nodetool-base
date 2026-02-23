import datetime

def generate_timestamped_name(pattern: str) -> str:
    """
    Generates a filename based on a pattern that may include strftime format codes.

    Args:
        pattern: The filename pattern which may contain date/time format codes
                (e.g., "%Y-%m-%d-output.txt").

    Returns:
        The pattern with format codes replaced by the current date and time.
    """
    return datetime.datetime.now().strftime(pattern)
