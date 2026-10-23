
from pathlib import Path

import chromadb


for directory in [
    Path("data/chroma"),
    Path("data/chroma_multilingual"),
]:
    print("\n" + "=" * 65)
    print(f"BANCO: {directory.resolve()}")

    if not directory.exists():
        print("Pasta não encontrada.")
        continue

    client = chromadb.PersistentClient(path=str(directory))

    for collection in client.list_collections():
        print(f"\nColeção: {collection.name}")
        print(f"Total de chunks: {collection.count()}")

        data = collection.get(include=["metadatas"])

        filenames = set()
        sources = set()

        for metadata in data.get("metadatas") or []:
            if not metadata:
                continue

            if metadata.get("filename"):
                filenames.add(metadata["filename"])

            if metadata.get("source"):
                sources.add(metadata["source"])

        print("Arquivos encontrados:")
        for filename in sorted(filenames):
            print(f"  - {filename}")

        print("Identificadores de origem:")
        for source in sorted(sources):
            print(f"  - {source}")
