import os
from .parser import ResumeParser

class ResumeLoader:
    def __init__(self):
        self.parser = ResumeParser()

    def load_folder(self, path: str) -> list[str]:
        documents = []

        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)

            if filename.endswith(".pdf"):
                chunks = self.parser.parse_pdf(full_path)
                chunk_index = 0

                for chunk in chunks:
                    documents.append({
                        "chunk": chunk,
                        "source": filename,
                        "id": f"{filename}_{chunk_index}"
                    })
                    chunk_index += 1
                
        return documents