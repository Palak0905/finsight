import io
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF file given as bytes."""
    output = io.StringIO()
    with io.BytesIO(file_bytes) as pdf_file:
        extract_text_to_fp(
            pdf_file,
            output,
            laparams=LAParams(),
            output_type="text",
            codec="utf-8"
        )
    text = output.getvalue()
    # Basic cleanup
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)
