from collections import defaultdict
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


INPUT_DIR = Path("./dataset")
OUTPUT_DIR = Path("./output_md")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

loader = DirectoryLoader(
    path=str(INPUT_DIR),
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    use_multithreading=True,
    silent_errors=False,
)

docs = loader.load()

# Gom các trang theo từng file PDF
documents_by_source = defaultdict(list)

for doc in docs:
    source = doc.metadata.get("source", "unknown.pdf")
    documents_by_source[source].append(doc)

# Xuất mỗi PDF thành một file Markdown
for source, pages in documents_by_source.items():
    source_path = Path(source)
    output_path = OUTPUT_DIR / f"{source_path.stem}.md"

    # Sắp xếp đúng thứ tự trang
    pages.sort(key=lambda doc: doc.metadata.get("page", 0))

    with output_path.open("w", encoding="utf-8") as file:
        file.write(f"# {source_path.stem}\n\n")

        for doc in pages:
            page_number = doc.metadata.get("page", 0) + 1
            content = doc.page_content.strip()

            file.write(f"<!-- page: {page_number} -->\n\n")

            if content:
                file.write(content)
            else:
                file.write("*Trang này không trích xuất được văn bản.*")

            file.write("\n\n")

    print(f"Đã tạo: {output_path}")

print(f"\nTổng số trang đã đọc: {len(docs)}")
print(f"Thư mục kết quả: {OUTPUT_DIR.resolve()}")