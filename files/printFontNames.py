import fitz


def extract_unique_fonts(pdf_files):
    unique_fonts = set()

    for pdf_path in pdf_files:
        doc = fitz.open(pdf_path)

        for page in doc:
            text_dict = page.get_text("rawdict")

            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_name = span["font"]
                            if font_name not in unique_fonts:
                                print(font_name)
                                unique_fonts.add(font_name)


if __name__ == "__main__":
    pdf_files = [
        "./pdfs/mercedes-eqs-sedan-manual_1.pdf",
        "./pdfs/mercedes-eqs-sedan-manual_2.pdf",
        "./pdfs/mercedes-eqs-sedan-manual_3.pdf",
        "./pdfs/mercedes-eqs-sedan-manual_4.pdf",
        "./pdfs/mercedes-eqs-sedan-manual_5.pdf",
    ]

    extract_unique_fonts(pdf_files)
