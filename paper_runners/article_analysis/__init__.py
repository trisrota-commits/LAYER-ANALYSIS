"""Byte-preserved article sources; explicit, import-safe analytical adapter."""


def run(repo, out):
    """Recompute published prediction analyses into a new output directory."""
    from analysis.revision_article import run as implementation
    return implementation(repo, out)
