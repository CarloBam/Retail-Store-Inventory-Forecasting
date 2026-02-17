import io
import pandas as pd

def create_word_report(df, insights, recommendations):
    """Generates a Word document report."""
    from docx import Document
    from docx.shared import Pt
    
    doc = Document()
    doc.add_heading('Retail Strategic Analysis Report', 0)
    
    doc.add_paragraph(f'Generated on {pd.Timestamp.now().date()}')
    
    # Exec Summary
    doc.add_heading('1. Executive Summary', level=1)
    doc.add_paragraph(f"Analyzed {len(df):,} transactions across {df['Category'].nunique()} categories.")
    
    if 'top_inelastic' in insights:
        doc.add_paragraph('Key Findings:', style='List Bullet')
        top = insights['top_inelastic'].iloc[0]['Category']
        doc.add_paragraph(f"Top Opportunity: '{top}' is inelastic (price insensitive).", style='List Bullet')

    # Recommendations
    doc.add_heading('2. Strategic Recommendations', level=1)
    for i, rec in enumerate(recommendations, 1):
        p = doc.add_paragraph()
        runner = p.add_run(f"{i}. {rec['Action']}")
        runner.bold = True
        doc.add_paragraph(f"Rationale: {rec['Rationale']}")
        doc.add_paragraph(f"Test: {rec['Test Design']}")
        
    stream = io.BytesIO()
    doc.save(stream)
    stream.seek(0)
    return stream

def create_pdf_report(df, insights, recommendations):
    """Generates a PDF report."""
    from fpdf import FPDF
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Retail Strategic Analysis', 0, 1, 'C')
            self.ln(10)
            
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(0, 10, f"Generated on {pd.Timestamp.now().date()}", 0, 1)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Executive Summary", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Analyzed {len(df):,} transactions across {df['Category'].nunique()} categories.")
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Strategic Recommendations", 0, 1)
    pdf.set_font("Arial", size=12)
    
    for i, rec in enumerate(recommendations, 1):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"{i}. {rec['Action']}", 0, 1)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"Rationale: {rec['Rationale']}")
        pdf.ln(2)
        
    # Output
    # FPDF output to string then bytes? Or directly to file?
    # output(dest='S') returns string (latin-1). 
    # Try generic approach for bytes
    try:
        return pdf.output(dest='S').encode('latin-1')
    except:
        return pdf.output(dest='S').encode('latin-1')
