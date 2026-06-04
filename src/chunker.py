def chunk_text(text:str, chunk_size:int, overlap:int) -> list[str]:

    if chunk_size <= overlap or chunk_size < 0 or overlap < 0:
        raise ValueError("Invalid parameters")

    text = text.strip()

    if not text:
        return []

    chunk_list = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunk_list.append(chunk)

        start = end - overlap

    return chunk_list


def chunk_pages(args, pages: list[dict]) ->  list[dict]:

    all_chunks = []

    for page in pages:
        source   = page["source"]
        page_num = page["page"]
        text     = page.get("text", "")

        page_chunks = chunk_text(text=text, chunk_size=args.chunk_size, overlap=args.overlap)

        for idx, texts in enumerate(page_chunks):

            chunk_id = f"{source}_p{page_num}_c{idx}"
            all_chunks.append(
                {"chunk_id":chunk_id, "source":source, "page":page_num, "chunk_text":texts}
            )

    return all_chunks
