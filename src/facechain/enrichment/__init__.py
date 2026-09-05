"""Profile enrichment and cross-platform discovery.

This package runs *after* the core verification pipeline (runner.run()) has
identified at least one verified candidate.  It never modifies the core
pipeline's output — it only adds a ``profile_graph`` to the Case.

Entry point: ``enrich_case(case, embedding, input_hashes, emit)``
"""
