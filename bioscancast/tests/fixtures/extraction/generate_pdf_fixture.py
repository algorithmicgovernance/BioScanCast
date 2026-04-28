"""Generate a realistic WHO Disease Outbreak News PDF fixture for testing.

Run once to create who_don_sample.pdf:
    python bioscancast/tests/fixtures/extraction/generate_pdf_fixture.py
"""

from __future__ import annotations

import os

import pymupdf


def generate_who_don_pdf(output_path: str) -> None:
    doc = pymupdf.open()
    doc.set_metadata(
        {
            "title": "Disease Outbreak News - Sudan Virus Disease in Uganda",
            "author": "World Health Organization",
            "creationDate": "D:20241201120000+00'00'",
        }
    )

    width, height = 595, 842  # A4
    margin = 50
    text_width = width - 2 * margin

    # ---- Page 1 ----
    page = doc.new_page(width=width, height=height)
    y = margin

    # Title (large bold)
    title_font_size = 18
    page.insert_text(
        (margin, y + title_font_size),
        "Disease Outbreak News",
        fontsize=title_font_size,
        fontname="helv",
    )
    y += title_font_size + 10

    subtitle_font_size = 14
    page.insert_text(
        (margin, y + subtitle_font_size),
        "Sudan Virus Disease in Uganda",
        fontsize=subtitle_font_size,
        fontname="helv",
    )
    y += subtitle_font_size + 8

    page.insert_text(
        (margin, y + 10),
        "1 December 2024",
        fontsize=10,
        fontname="helv",
    )
    y += 30

    # Section: Situation at a glance
    section_font_size = 14
    page.insert_text(
        (margin, y + section_font_size),
        "Situation at a glance",
        fontsize=section_font_size,
        fontname="helv",
    )
    y += section_font_size + 10

    body_font_size = 10
    paragraphs_p1 = [
        "On 15 November 2024, the Uganda Ministry of Health notified WHO of an outbreak of Sudan virus disease (SVD) in Kampala District. The index case was a 32-year-old male health worker at Mulago National Referral Hospital who developed symptoms on 8 November 2024.",
        "As of 30 November 2024, a total of 47 confirmed cases and 12 probable cases have been reported, with 9 confirmed deaths (case fatality ratio among confirmed cases: 19.1%). Cases have been reported from 5 districts: Kampala, Wakiso, Mubende, Kassanda, and Jinja.",
        "WHO assesses the risk at the national level as very high due to the confirmed cases among health workers, evidence of community transmission, and challenges in contact tracing. The risk at the regional level is assessed as high and at the global level as low.",
    ]

    for para in paragraphs_p1:
        lines = _wrap_text(para, body_font_size, text_width)
        for line in lines:
            page.insert_text(
                (margin, y + body_font_size),
                line,
                fontsize=body_font_size,
                fontname="helv",
            )
            y += body_font_size + 3
        y += 8

    # ---- Page 2: Epidemiological summary with table ----
    page2 = doc.new_page(width=width, height=height)
    y = margin

    page2.insert_text(
        (margin, y + section_font_size),
        "Epidemiological summary",
        fontsize=section_font_size,
        fontname="helv",
    )
    y += section_font_size + 10

    epi_text = "The outbreak was first detected in Kampala District among health workers at Mulago National Referral Hospital. Subsequent investigation identified cases linked to community exposure and nosocomial transmission. Active case finding and contact tracing are ongoing in all affected districts."
    for line in _wrap_text(epi_text, body_font_size, text_width):
        page2.insert_text(
            (margin, y + body_font_size),
            line,
            fontsize=body_font_size,
            fontname="helv",
        )
        y += body_font_size + 3
    y += 15

    # Table header
    page2.insert_text(
        (margin, y + 11),
        "Table 1: Confirmed and probable SVD cases by district, Uganda, Nov 2024",
        fontsize=9,
        fontname="helv",
    )
    y += 20

    # Draw table
    headers = ["District", "Confirmed", "Probable", "Deaths", "CFR (%)"]
    rows = [
        ["Kampala", "22", "5", "4", "18.2"],
        ["Wakiso", "11", "3", "2", "18.2"],
        ["Mubende", "8", "2", "2", "25.0"],
        ["Kassanda", "4", "1", "1", "25.0"],
        ["Jinja", "2", "1", "0", "0.0"],
        ["Total", "47", "12", "9", "19.1"],
    ]
    col_widths = [120, 80, 80, 80, 80]
    row_height = 18

    # Header row
    x = margin
    for i, header in enumerate(headers):
        page2.draw_rect(
            pymupdf.Rect(x, y, x + col_widths[i], y + row_height),
            color=(0, 0, 0),
            width=0.5,
        )
        page2.insert_text(
            (x + 3, y + 13),
            header,
            fontsize=9,
            fontname="helv",
        )
        x += col_widths[i]
    y += row_height

    for row in rows:
        x = margin
        for i, cell in enumerate(row):
            page2.draw_rect(
                pymupdf.Rect(x, y, x + col_widths[i], y + row_height),
                color=(0, 0, 0),
                width=0.5,
            )
            page2.insert_text(
                (x + 3, y + 13),
                cell,
                fontsize=9,
                fontname="helv",
            )
            x += col_widths[i]
        y += row_height

    y += 20

    page2.insert_text(
        (margin, y + section_font_size),
        "Public health response",
        fontsize=section_font_size,
        fontname="helv",
    )
    y += section_font_size + 10

    response_text = "The Government of Uganda activated its national Ebola/SVD response plan. An Incident Management System has been established at the Ministry of Health. WHO has deployed a surge team to support the response, including epidemiologists, laboratory experts, and infection prevention and control specialists."
    for line in _wrap_text(response_text, body_font_size, text_width):
        page2.insert_text(
            (margin, y + body_font_size),
            line,
            fontsize=body_font_size,
            fontname="helv",
        )
        y += body_font_size + 3

    # ---- Page 3: WHO risk assessment ----
    page3 = doc.new_page(width=width, height=height)
    y = margin

    page3.insert_text(
        (margin, y + section_font_size),
        "WHO risk assessment",
        fontsize=section_font_size,
        fontname="helv",
    )
    y += section_font_size + 10

    risk_paragraphs = [
        "Sudan virus disease is a severe and often fatal illness caused by infection with one of the Sudan virus species of the genus Ebolavirus. Outbreaks of SVD have occurred sporadically since 1976, primarily in East Africa.",
        "The current outbreak presents several concerning features: infection among health workers suggesting nosocomial transmission, spread to multiple districts including the capital city Kampala, and limited availability of specific therapeutics or licensed vaccines for Sudan virus.",
        "WHO continues to monitor the situation closely and is working with the Government of Uganda and partners to implement a comprehensive response. International Health Regulations procedures have been followed, and neighbouring countries have been alerted to enhance surveillance at points of entry.",
    ]

    for para in risk_paragraphs:
        for line in _wrap_text(para, body_font_size, text_width):
            page3.insert_text(
                (margin, y + body_font_size),
                line,
                fontsize=body_font_size,
                fontname="helv",
            )
            y += body_font_size + 3
        y += 8

    doc.save(output_path)
    doc.close()


def _wrap_text(text: str, font_size: float, max_width: float) -> list[str]:
    """Simple word-wrap approximation."""
    char_width = font_size * 0.5
    chars_per_line = int(max_width / char_width)
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= chars_per_line:
            current_line = f"{current_line} {word}" if current_line else word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "who_don_sample.pdf")
    generate_who_don_pdf(out)
    print(f"Generated {out}")
