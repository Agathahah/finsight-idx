"""Header/footer removal, OCR fallback and the index manifest."""

import json
from types import SimpleNamespace

import pytest

from src.nlp import pdf_extractor as pe
from src.nlp.pdf_extractor import PAGE_BREAK, find_boilerplate, looks_like_prose, strip_boilerplate
from src.rag import manifest as mf


def _pages(n=20):
    pages = []
    for i in range(1, n + 1):
        nav = "Ikhtisar Data Keuangan Laporan Manajemen Profi l Perusahaan" if i % 2 else \
              "Ikhtisar Data Keuangan Laporan Manajemen Profil Perusahaan"
        pages.append(
            f"{nav}\n(Dalam jutaan Rupiah, kecuali dinyatakan lain)\nKredit menurut sektor\n"
            f"Isi unik halaman {i} tentang kredit sektor {i * 7}.\n"
            f"Penjelasan lanjutan.\nSumber: catatan 12\n"
            f"Laporan Tahunan 2023PT Bank Central Asia Tbk {i + 2}\n{i + 2}"
        )
    return pages


# -- boilerplate ------------------------------------------------------------

def test_running_headers_found_across_ligature_and_number_variants():
    bp = find_boilerplate(_pages())
    texts = [strip_boilerplate(p, bp)[0] for p in _pages()]
    assert not any("Ikhtisar Data Keuangan" in t for t in texts)
    assert not any("Bank Central Asia Tbk" in t for t in texts)
    assert all("Isi unik halaman" in t for t in texts)


def test_unit_lines_are_kept_even_when_repeated():
    bp = find_boilerplate(_pages())
    assert all("Dalam jutaan Rupiah" in strip_boilerplate(p, bp)[0] for p in _pages())


def test_short_documents_are_left_alone():
    assert find_boilerplate(_pages(5)) == set()


def test_body_lines_that_differ_only_in_numbers_are_kept():
    bp = find_boilerplate(_pages())
    assert all("Isi unik halaman" in strip_boilerplate(p, bp)[0] for p in _pages())


def test_only_edge_page_numbers_are_removed():
    text = "12\nTotal kredit\n2023\n250\n13"
    out, removed = strip_boilerplate(text, set())
    assert out == "Total kredit\n2023\n250"
    assert removed == 2


# -- OCR --------------------------------------------------------------------

PROSE = ("Kami telah mengaudit laporan keuangan konsolidasian PT Contoh Tbk dan entitas "
         "anaknya yang terdiri dari laporan posisi keuangan konsolidasian tanggal 31 Desember "
         "2023 serta laporan laba rugi dan penghasilan komprehensif lain konsolidasian.")


def test_prose_filter_rejects_photo_noise():
    assert looks_like_prose(PROSE)
    assert not looks_like_prose("01")
    assert not looks_like_prose("|| ~~ // 12 34 56 ## ** " * 20)


@pytest.fixture
def scanned_pdf(tmp_path):
    """A two-page PDF made only of images: no text layer at all."""
    from PIL import Image
    path = tmp_path / "scan.pdf"
    imgs = [Image.new("RGB", (400, 500), "white") for _ in range(2)]
    imgs[0].save(path, save_all=True, append_images=imgs[1:])
    return path


def test_ocr_fills_pages_without_text_layer(scanned_pdf, monkeypatch):
    monkeypatch.setattr(pe, "ocr_available", lambda: True)
    monkeypatch.setattr(pe, "ocr_page_image", lambda image, lang: PROSE)
    doc = pe.PDFExtractor(ocr="auto", min_section_chars=50).extract(scanned_pdf)
    assert [p.source for p in doc.pages] == ["ocr", "ocr"]
    assert doc.metadata["pages_ocr"] == [1, 2]
    assert doc.metadata["pages_empty"] == []


def test_ocr_off_reports_unsearchable_document(scanned_pdf):
    with pytest.raises(ValueError, match="FINSIGHT_OCR"):
        pe.PDFExtractor(ocr="off").extract(scanned_pdf)


def test_ocr_required_fails_loudly_without_tesseract(scanned_pdf, monkeypatch):
    monkeypatch.setattr(pe, "ocr_available", lambda: False)
    with pytest.raises(RuntimeError, match="brew install tesseract"):
        pe.PDFExtractor(ocr="required").extract(scanned_pdf)


def test_chunks_from_ocr_pages_are_marked():
    from src.rag.indexer import RAGIndexer, TextChunker
    from src.rag.retriever import RetrievedChunk

    rag = RAGIndexer.__new__(RAGIndexer)
    rag.chunker = TextChunker(256, 64)
    section = SimpleNamespace(text=PAGE_BREAK.join([PROSE * 2, PROSE * 2]), start_page=10,
                              title="Laporan Auditor", section_type="financial")
    doc = SimpleNamespace(sections=[section], pages=[SimpleNamespace(page_number=10, source="text"),
                                                     SimpleNamespace(page_number=11, source="ocr")])
    chunks = rag._build_chunks(doc, "d", "BBCA", 2023, "x.pdf")
    assert {c.page_number: c.extraction for c in chunks} == {10: "text", 11: "ocr"}
    assert chunks[-1].to_metadata()["extraction"] == "ocr"
    cited = RetrievedChunk(chunk_id="1", text="t", page_number=11, section_title="", section_type="",
                           source_file="x.pdf", company="BBCA", year=2023, extraction="ocr")
    assert cited.citation() == "[BBCA, 2023, hal. 11, OCR]"


# -- manifest ---------------------------------------------------------------

def _record(tmp_path, doc_id="a", embed="m1", size=256):
    return mf.record_document(tmp_path, doc_id, {"source_file": "x.pdf"},
                              embed_model=embed, chunk_size=size, chunk_overlap=64)


def test_manifest_round_trip(tmp_path):
    _record(tmp_path)
    m = mf.check_compatible(tmp_path, "m1", strict=True)
    assert m["schema_version"] == mf.INDEX_SCHEMA_VERSION
    assert m["documents"]["a"]["source_file"] == "x.pdf"


def test_missing_manifest_is_fatal_only_in_strict_mode(tmp_path):
    assert mf.check_compatible(tmp_path, "m1", strict=False) is None
    with pytest.raises(mf.IndexVersionError):
        mf.check_compatible(tmp_path, "m1", strict=True)


def test_wrong_embedding_model_is_refused(tmp_path):
    _record(tmp_path, embed="m1")
    with pytest.raises(mf.IndexVersionError, match="not be comparable"):
        mf.check_compatible(tmp_path, "m2", strict=False)


def test_old_schema_is_refused(tmp_path):
    _record(tmp_path)
    path = mf.manifest_path(tmp_path)
    data = json.loads(path.read_text())
    data["schema_version"] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(mf.IndexVersionError, match="Re-index"):
        mf.check_compatible(tmp_path, "m1", strict=False)


def test_changing_settings_with_other_documents_present_is_refused(tmp_path):
    _record(tmp_path, doc_id="a")
    _record(tmp_path, doc_id="b")
    with pytest.raises(mf.IndexVersionError, match="re-index every document"):
        _record(tmp_path, doc_id="a", size=512)


def test_reindexing_the_only_document_may_change_settings(tmp_path):
    _record(tmp_path, doc_id="a")
    assert _record(tmp_path, doc_id="a", size=512)["chunk_size"] == 512
