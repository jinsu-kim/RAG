from pathlib import Path
import pdfplumber


def load_pdf(pdf_path: str | Path) -> list[dict]:

    pdf_path = Path(pdf_path)
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            pages.append({
                "source": pdf_path.name,
                "page":   page_num,
                "text":   text,
            })

    return pages


def load_all_pdfs(data_dir: str | Path, target_files: set[str] | None = None) -> list[dict]:

    pdfs_path = Path(data_dir)

    all_pages = []

    for pdf_path in sorted(pdfs_path.glob("*.pdf")):

        if target_files is not None and pdf_path.name not in target_files:
            continue

        pages = load_pdf(pdf_path)
        all_pages.extend(pages)

    return all_pages
