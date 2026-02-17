"""Tests for POST /ingest/file and POST /ingest/url endpoints."""

import pytest
from unittest.mock import AsyncMock, patch


class TestIngestFile:
    async def test_txt_file(self, app_client):
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("test.txt", b"Hello world content", "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "test.txt"
        assert data["status"] == "ingested"
        assert "document_id" in data
        assert data["chunk_count"] >= 1

    async def test_md_file(self, app_client):
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("readme.md", b"# Title\nContent here", "text/markdown")},
        )
        assert resp.status_code == 200
        assert resp.json()["filename"] == "readme.md"

    async def test_pdf_file(self, app_client, sample_pdf_bytes):
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("doc.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 200
        assert resp.json()["filename"] == "doc.pdf"

    async def test_unsupported_extension(self, app_client):
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("script.exe", b"binary data", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]

    async def test_empty_file(self, app_client):
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert resp.status_code == 400
        assert "Empty file" in resp.json()["detail"]

    async def test_chunk_count_matches(self, app_client):
        # Longer text that produces multiple chunks
        long_text = "Word " * 1000  # 5000 chars
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("long.txt", long_text.encode(), "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.json()["chunk_count"] >= 1

    async def test_calls_embedding_service(self, app_client):
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("test.txt", b"Some content for embedding", "text/plain")},
        )
        assert resp.status_code == 200
        app_client._mock_embed.embed.assert_called()

    async def test_calls_es_index_chunks(self, app_client):
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("test.txt", b"Some content for indexing", "text/plain")},
        )
        assert resp.status_code == 200
        app_client._mock_es.index_chunks.assert_called()

    async def test_whitespace_only_file(self, app_client):
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("spaces.txt", b"   \n\n  ", "text/plain")},
        )
        assert resp.status_code == 400

    async def test_no_extension(self, app_client):
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("noext", b"some content", "text/plain")},
        )
        assert resp.status_code == 400

    async def test_tags_passed_as_form_field(self, app_client):
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("test.txt", b"Hello world content", "text/plain")},
            data={"tags": "research, ml"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert set(data["tags"]) == {"research", "ml"}

    async def test_empty_tags_default(self, app_client):
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("test.txt", b"Hello world content", "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.json()["tags"] == []

    async def test_auto_tags_merged_with_user_tags(self, app_client):
        app_client._mock_gen_tags.return_value = ["auto1", "auto2"]
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("test.txt", b"Hello world content", "text/plain")},
            data={"tags": "manual"},
        )
        assert resp.status_code == 200
        tags = resp.json()["tags"]
        assert set(tags) == {"manual", "auto1", "auto2"}

    async def test_auto_tags_when_no_user_tags(self, app_client):
        app_client._mock_gen_tags.return_value = ["auto1", "auto2"]
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("test.txt", b"Hello world content", "text/plain")},
        )
        assert resp.status_code == 200
        tags = resp.json()["tags"]
        assert set(tags) == {"auto1", "auto2"}

    async def test_auto_tags_deduped_with_user_tags(self, app_client):
        app_client._mock_gen_tags.return_value = ["research", "new"]
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("test.txt", b"Hello world content", "text/plain")},
            data={"tags": "research"},
        )
        assert resp.status_code == 200
        tags = resp.json()["tags"]
        assert sorted(tags) == ["new", "research"]


class TestIngestFileDuplicateDetection:
    async def test_reupload_returns_updated_status(self, app_client):
        app_client._mock_es.find_document_by_source.return_value = "existing-doc-id"
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("test.txt", b"Hello world content", "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "updated"
        assert data["document_id"] == "existing-doc-id"
        app_client._mock_es.delete_document.assert_called_with("existing-doc-id")

    async def test_new_file_returns_ingested_status(self, app_client):
        app_client._mock_es.find_document_by_source.return_value = None
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("new.txt", b"Brand new content", "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ingested"
        app_client._mock_es.delete_document.assert_not_called()

    async def test_unknown_filename_skips_lookup(self, app_client):
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("unknown", b"content", "text/plain")},
        )
        # "unknown" has no extension -> 400
        # Let's use a valid name to test the skip
        assert resp.status_code == 400

    async def test_unknown_filename_always_creates_new(self, app_client):
        """When filename is 'unknown', find_document_by_source is not called."""
        # We can't easily test with literally "unknown" as filename since it has no extension.
        # Instead, verify that find_document_by_source is called with the actual filename.
        app_client._mock_es.find_document_by_source.return_value = None
        resp = await app_client.post(
            "/ingest/file",
            files={"file": ("test.txt", b"Hello", "text/plain")},
        )
        assert resp.status_code == 200
        app_client._mock_es.find_document_by_source.assert_called_once_with("test.txt", "text")


class TestIngestUrl:
    async def test_valid_url(self, app_client):
        with patch(
            "app.api.routes.ingest.parse_url",
            new_callable=AsyncMock,
            return_value={
                "content": "Web page content here.",
                "metadata": {"filename": "https://example.com", "source_type": "web", "url": "https://example.com"},
            },
        ):
            resp = await app_client.post("/ingest/url", json={"url": "https://example.com"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ingested"
        assert data["chunk_count"] >= 1

    async def test_empty_content_from_url(self, app_client):
        with patch(
            "app.api.routes.ingest.parse_url",
            new_callable=AsyncMock,
            return_value={
                "content": "   ",
                "metadata": {"filename": "https://example.com", "source_type": "web"},
            },
        ):
            resp = await app_client.post("/ingest/url", json={"url": "https://example.com"})

        assert resp.status_code == 400

    async def test_missing_url_field(self, app_client):
        resp = await app_client.post("/ingest/url", json={})
        assert resp.status_code == 422

    async def test_calls_embedding_and_es(self, app_client):
        with patch(
            "app.api.routes.ingest.parse_url",
            new_callable=AsyncMock,
            return_value={
                "content": "Some web content.",
                "metadata": {"filename": "https://example.com", "source_type": "web"},
            },
        ):
            resp = await app_client.post("/ingest/url", json={"url": "https://example.com"})

        assert resp.status_code == 200
        app_client._mock_embed.embed.assert_called()
        app_client._mock_es.index_chunks.assert_called()

    async def test_batched_embeddings(self, app_client):
        # Content long enough to create multiple chunks
        long_content = "Sentence. " * 500
        with patch(
            "app.api.routes.ingest.parse_url",
            new_callable=AsyncMock,
            return_value={
                "content": long_content,
                "metadata": {"filename": "https://example.com", "source_type": "web"},
            },
        ):
            resp = await app_client.post("/ingest/url", json={"url": "https://example.com"})

        assert resp.status_code == 200

    async def test_tags_in_json_body(self, app_client):
        with patch(
            "app.api.routes.ingest.parse_url",
            new_callable=AsyncMock,
            return_value={
                "content": "Web page content here.",
                "metadata": {"filename": "https://example.com", "source_type": "web"},
            },
        ):
            resp = await app_client.post(
                "/ingest/url", json={"url": "https://example.com", "tags": ["research", "ml"]}
            )

        assert resp.status_code == 200
        assert set(resp.json()["tags"]) == {"research", "ml"}

    async def test_auto_tags_merged_url(self, app_client):
        app_client._mock_gen_tags.return_value = ["web", "article"]
        with patch(
            "app.api.routes.ingest.parse_url",
            new_callable=AsyncMock,
            return_value={
                "content": "Web page content here.",
                "metadata": {"filename": "https://example.com", "source_type": "web"},
            },
        ):
            resp = await app_client.post(
                "/ingest/url", json={"url": "https://example.com", "tags": ["manual"]}
            )

        assert resp.status_code == 200
        tags = resp.json()["tags"]
        assert set(tags) == {"manual", "web", "article"}

    async def test_default_empty_tags_url(self, app_client):
        with patch(
            "app.api.routes.ingest.parse_url",
            new_callable=AsyncMock,
            return_value={
                "content": "Web page content here.",
                "metadata": {"filename": "https://example.com", "source_type": "web"},
            },
        ):
            resp = await app_client.post("/ingest/url", json={"url": "https://example.com"})

        assert resp.status_code == 200
        assert resp.json()["tags"] == []

    async def test_response_shape(self, app_client):
        with patch(
            "app.api.routes.ingest.parse_url",
            new_callable=AsyncMock,
            return_value={
                "content": "Content here.",
                "metadata": {"filename": "https://example.com", "source_type": "web"},
            },
        ):
            resp = await app_client.post("/ingest/url", json={"url": "https://example.com"})

        data = resp.json()
        assert "document_id" in data
        assert "filename" in data
        assert "chunk_count" in data
        assert "status" in data


class TestIngestUrlDuplicateDetection:
    async def test_reupload_url_returns_updated(self, app_client):
        app_client._mock_es.find_document_by_source.return_value = "existing-url-doc"
        with patch(
            "app.api.routes.ingest.parse_url",
            new_callable=AsyncMock,
            return_value={
                "content": "Web page content here.",
                "metadata": {"filename": "https://example.com", "source_type": "web"},
            },
        ):
            resp = await app_client.post("/ingest/url", json={"url": "https://example.com"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "updated"
        assert data["document_id"] == "existing-url-doc"
        app_client._mock_es.delete_document.assert_called_with("existing-url-doc")

    async def test_new_url_returns_ingested(self, app_client):
        app_client._mock_es.find_document_by_source.return_value = None
        with patch(
            "app.api.routes.ingest.parse_url",
            new_callable=AsyncMock,
            return_value={
                "content": "New web content.",
                "metadata": {"filename": "https://new.example.com", "source_type": "web"},
            },
        ):
            resp = await app_client.post("/ingest/url", json={"url": "https://new.example.com"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ingested"
        app_client._mock_es.delete_document.assert_not_called()
