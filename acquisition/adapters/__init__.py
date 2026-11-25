"""Adapters — one per SOURCE TYPE (api / dump / crawl).

A new source type needs a new adapter; a new source of a known type only
needs config. The adapter contract: fetch raw rows into the run dir
(resumable), or stream rows through the counted chain in a single pass.
"""
