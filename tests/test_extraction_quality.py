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
    assert chunks[0].text.startswith("[Laporan Auditor · hal. 10]\n")
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


def test_page_context_describes_tables_only():
    from src.rag.indexer import contextual_text, page_context
    table = ("Ikhtisar Data Keuangan\n(dalam miliar Rupiah) 2023 2022 2021\nTotal Aset 1.408.107 1.314.732\n"
             + "baris\n" * 20 + "Jumlah Rekening (dalam ribuan) 37.632")
    assert page_context(table) == "dalam miliar Rupiah · kolom 2023 2022 2021"
    assert page_context("Pada tahun 2023 laba naik. Target 2023 tercapai.") == ""
    assert page_context("(Dalam jutaanRupiah, kecuali dinyatakan lain)\n2023 2022") == \
        "Dalam jutaan Rupiah, kecuali dinyatakan lain · kolom 2023 2022"
    assert contextual_text("Total Aset 1.408.107", "Ikhtisar Data Keuangan", 18, "dalam miliar Rupiah") == \
        "[Ikhtisar Data Keuangan · hal. 18 · dalam miliar Rupiah]\nTotal Aset 1.408.107"


# -- table rows -------------------------------------------------------------

TABLE_PAGE = """Ikhtisar Data Keuangan
(dalam miliar Rupiah) 2023 2022 2021
Posisi Keuangan
Total Aset 1.408.107 1.314.732 1.228.345
Total Kredit1 810.392 711.262 636.987
CAR Tier 1 28,3% 24,8% 24,7%
Beban Operasional (37.503) (32.483) (30.308)
Rasio Kredit terhadap Dana Pihak Ketiga (Loan to Deposit Ratio
70,2% 65,2% 62,0%
Giro pada bank-bank lain -setelah dikurangi cadangan
(31 Desember 2022: Rp 743) 40,43 5.614.353 4.751.916 1.000
Jumlah pendapatan bunga 87.397.774 72.241.191 Pemilik entitas induk 48.639.122 40.735.722 1
Laba Bersih per Saham 395 330"""


def test_table_rows_become_labelled_sentences():
    from src.rag.indexer import table_rows
    rows = table_rows(TABLE_PAGE)
    assert rows[0] == "Total Aset: 2023 = 1.408.107; 2022 = 1.314.732; 2021 = 1.228.345 (dalam miliar Rupiah)"
    assert rows[1].startswith("Total Kredit: 2023 = 810.392")          # footnote digit removed
    assert rows[2].startswith("CAR Tier 1: 2023 = 28,3%")                # "Tier 1" kept
    assert rows[3].startswith("Beban Operasional: 2023 = (37.503)")
    assert rows[4].startswith("Rasio Kredit terhadap Dana Pihak Ketiga (Loan to Deposit Ratio: 2023 = 70,2%")
    assert rows[5].startswith("Giro pada bank-bank lain -setelah dikurangi cadangan: 2023 = 5.614.353")
    assert not any("Pemilik entitas induk" in r for r in rows)           # side-by-side tables skipped
    assert not any(r.startswith("Laba Bersih per Saham") for r in rows)  # 2 values for 3 columns: skipped


def test_pages_without_year_columns_give_no_rows():
    from src.rag.indexer import table_rows
    assert table_rows("Laba naik 19,4% menjadi Rp48,6 triliun pada 2023.\nNPL 1,9% 1,7% 2,2%") == []


def test_table_row_chunks_are_added_with_their_type():
    from src.rag.indexer import RAGIndexer, TextChunker
    rag = RAGIndexer.__new__(RAGIndexer)
    rag.chunker = TextChunker(256, 64)
    section = SimpleNamespace(text=TABLE_PAGE, start_page=18, end_page=18,
                              title="Ikhtisar Data Keuangan", section_type="financial")
    doc = SimpleNamespace(sections=[section], pages=[SimpleNamespace(page_number=18, source="text", text=TABLE_PAGE)])
    chunks = rag._build_chunks(doc, "d", "BBCA", 2023, "x.pdf")
    rows = [c for c in chunks if c.chunk_type == "table_row"]
    assert len(rows) == 6 and all(c.page_number == 18 for c in rows)
    assert rows[0].text.startswith("[Ikhtisar Data Keuangan · hal. 18]\nTotal Aset: 2023 = 1.408.107")
    assert rows[0].to_metadata()["chunk_type"] == "table_row"
    assert len({c.chunk_id for c in chunks}) == len(chunks)
