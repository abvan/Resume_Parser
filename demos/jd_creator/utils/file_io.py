from fpdf import FPDF
import tempfile
import os

def save_jd_to_pdf(jd_text: str, filename: str = "job_description.pdf") -> str:
    """
    Save the job description text to a PDF and return the filepath.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Split by line and write each one
    for line in jd_text.split('\n'):
        pdf.multi_cell(0, 10, txt=line, align='L')

    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, filename)
    pdf.output(pdf_path)

    return pdf_path