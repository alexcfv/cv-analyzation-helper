import os
from .parser import ResumeParser

class ResumeLoader:
    def __init__(self):
        self.parser = ResumeParser()

    def load_folder(self, path: str):
        documents = []

        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)

            if filename.endswith(".pdf"):
                text = self.parser.parse_pdf(full_path)

                documents.append({
                    "text": text,
                    "source": filename
                })
                
        return documents