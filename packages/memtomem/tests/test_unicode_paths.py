"""Regression tests for Unicode path normalization (NFC/NFD) in web routes.

On macOS APFS, non-ASCII paths (e.g. Korean from Google Drive, accented Latin)
can be stored in NFD (decomposed) form on disk, while users type NFC (composed)
form. Without normalization, Path.resolve() on the two forms produces different
strings, causing a 403 "Path is not an indexed source file" even though the
file was indexed.

These tests verify that norm_path() applies NFC normalization so that both
forms compare equal.
"""

from __future__ import annotations

import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from memtomem.models import Chunk, ChunkMetadata
from memtomem.storage.sqlite_helpers import norm_path
from memtomem.web.app import create_app


# ---------------------------------------------------------------------------
# NFC / NFD test data
# ---------------------------------------------------------------------------

# Use explicit Unicode escapes to avoid editor mangling.
# "e\u0301" = e + combining acute accent (NFD)
# "\u00e9" = e with acute accent precomposed (NFC)
NFC_CHAR = "\u00e9"
NFD_CHAR = "e\u0301"

NFC_PATH = f"/tmp/t{NFC_CHAR}st-dir"
NFD_PATH = f"/tmp/t{NFD_CHAR}st-dir"

# Sanity checks at module level
assert NFC_PATH != NFD_PATH, "NFC and NFD strings must differ byte-wise"
assert norm_path(Path(NFC_PATH)) == norm_path(Path(NFD_PATH)), (
    "norm_path must normalize NFC and NFD to the same string"
)


# ---------------------------------------------------------------------------
# Chunk factory
# ---------------------------------------------------------------------------

def _make_chunk(source: str) -> Chunk:
    return Chunk(
        content="test content",
        metadata=ChunkMetadata(
            source_file=Path(source),
            heading_hierarchy=("H1",),
            tags=("tag1",),
            namespace="default",
            start_line=1,
            end_line=5,
        ),
        id=uuid.uuid4(),
        content_hash="abc",
        embedding=[0.1] * 768,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# App fixture: storage has NFC paths, user sends NFD
# ---------------------------------------------------------------------------

@pytest.fixture
def app_nfc_stored():
    app = create_app(lifespan=None)

    storage = AsyncMock()
    storage.get_all_source_files = AsyncMock(return_value=[Path(NFC_PATH)])
    storage.list_chunks_by_source = AsyncMock(return_value=[_make_chunk(NFC_PATH)])
    storage.delete_by_source = AsyncMock(return_value=1)
    storage.get_source_files_with_counts = AsyncMock(
        return_value=[(Path(NFC_PATH), 3, "2026-01-01T00:00:00", "default", 100, 50, 200)]
    )
    storage.get_stats = AsyncMock(return_value={"total_chunks": 3, "total_sources": 1})

    app.state.storage = storage
    app.state.config = type("Cfg", (), {"storage": type("S", (), {"backend": "sqlite"})()})()
    return app


@pytest.fixture
async def client_nfc(app_nfc_stored):
    transport = ASGITransport(app=app_nfc_stored)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# App fixture: storage has NFD paths, user sends NFC
# ---------------------------------------------------------------------------

@pytest.fixture
def app_nfd_stored():
    app = create_app(lifespan=None)

    storage = AsyncMock()
    storage.get_all_source_files = AsyncMock(return_value=[Path(NFD_PATH)])
    storage.list_chunks_by_source = AsyncMock(return_value=[_make_chunk(NFD_PATH)])
    storage.delete_by_source = AsyncMock(return_value=1)

    app.state.storage = storage
    app.state.config = type("Cfg", (), {"storage": type("S", (), {"backend": "sqlite"})()})()
    return app


@pytest.fixture
async def client_nfd(app_nfd_stored):
    transport = ASGITransport(app=app_nfd_stored)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Tests: user sends NFD, storage has NFC
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_source_nfd_matches_nfc(client_nfc):
    """DELETE /api/sources with NFD path should match NFC-indexed storage."""
    resp = await client_nfc.delete("/api/sources", params={"path": NFD_PATH})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_list_chunks_nfd_matches_nfc(client_nfc):
    """GET /api/chunks with NFD source path should match NFC-indexed storage."""
    resp = await client_nfc.get("/api/chunks", params={"source": NFD_PATH})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"


# ---------------------------------------------------------------------------
# Tests: user sends NFC, storage has NFD
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_source_nfc_matches_nfd(client_nfd):
    """DELETE /api/sources with NFC path should match NFD-indexed storage."""
    resp = await client_nfd.delete("/api/sources", params={"path": NFC_PATH})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_list_chunks_nfc_matches_nfd(client_nfd):
    """GET /api/chunks with NFC source path should match NFD-indexed storage."""
    resp = await client_nfd.get("/api/chunks", params={"source": NFC_PATH})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"


# ---------------------------------------------------------------------------
# Unit tests: norm_path always produces NFC
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_str",
    [
        NFC_CHAR,
        NFD_CHAR,
        NFC_PATH,
        NFD_PATH,
    ],
)
def test_norm_path_produces_nfc(input_str: str):
    """norm_path() output must be in NFC form."""
    result = norm_path(Path(input_str))
    assert result == unicodedata.normalize("NFC", result), (
        f"norm_path({input_str!r}) produced non-NFC result: {result!r}"
    )


def test_norm_path_nfc_nfd_equivalence():
    """Paths differing only in Unicode form must normalize identically."""
    pairs = [
        (NFC_CHAR, NFD_CHAR),
        (NFC_PATH, NFD_PATH),
    ]
    for nfc, nfd in pairs:
        assert norm_path(Path(nfc)) == norm_path(Path(nfd)), (
            f"norm_path mismatch: {nfc!r} vs {nfd!r}"
        )
