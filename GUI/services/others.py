import io
from flask import send_file
import pandas as pd

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import io

def generate_pdf_response(title, data, filename):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # Add a title
    elements.append(Paragraph(title, styles['Title']))
    elements.append(Spacer(1, 12))

    if not data:
        elements.append(Paragraph("No data available.", styles['Normal']))
    else:
        # Remove 'id' field from each row (optional)
        data = [
            {k: v for k, v in row.items() if k != 'id'} for row in data
        ]

        # Create the header from the first row keys
        header = list(data[0].keys())
        table_data = [header]

        # Append each row's values in order of header
        for row in data:
            table_data.append([row[col] for col in header])

        # Create the table
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ]))

        elements.append(table)

    # Build PDF
    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{filename}.pdf",
        mimetype='application/pdf'
    )

def generate_excel_response(title, data, filename):
    # Assuming user_audits is a list of dicts
    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name=title)
    output.seek(0)
    return send_file(output, download_name=f"{filename}.xlsx", as_attachment=True, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
