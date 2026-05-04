"""Shared pytest configuration and fixtures."""

import pytest


# Make all async tests use asyncio by default
pytest_plugins = ["pytest_asyncio"]
