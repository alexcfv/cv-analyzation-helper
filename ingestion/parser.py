import pdfplumber

class ResumeParser:
    def parse_pdf(self, file_path: str) -> str:
        text = ""

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

        return self.clean_text(text)