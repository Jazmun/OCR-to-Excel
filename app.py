import streamlit as st
import pdf2image
import numpy as np
import pandas as pd
import easyocr
import re
import tempfile
import os

st.set_page_config(page_title="Check Grid Matrix Parser", layout="wide")

@st.cache_resource
def load_ocr_engine():
    # Loads English model. Set gpu=True if your execution environment features a CUDA GPU
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr_engine()

st.title("🖨️ Advanced Check Grid Matrix Parser")
st.write("Optimized 3x8 mapping pipeline to process check images and text blocks together.")

uploaded_file = st.file_uploader("Upload Scanned Check Document (PDF)", type=["pdf"])

def extract_check_face_metadata(text_lines):
    """
    Extracts the Payee name directly from the cropped check face.
    """
    full_string = " | ".join(text_lines)
    
    # Isolate Payee: Looks for text between standard check indicators
    payee_patterns = [
        r"(?:PAY TO THE ORDER OF|PAY TO THE|PAY TO)\s+([^|_\d$]+)",
        r"INC\.\s*\|\s*\|\s*([^|_\d$]+)"
    ]
    
    payee = "Unassigned / Handwritten"
    for pattern in payee_patterns:
        match = re.search(pattern, full_string, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" |-_")
            if len(candidate) > 3 and "LANDSCAPING" not in candidate.upper():
                payee = candidate
                break
                
    return payee

def extract_bank_block_metadata(text_lines):
    """
    Parses the structured bank text block directly below each check image.
    """
    full_string = " ".join(text_lines)
    
    # 1. Date (Looks for MM/DD/YYYY)
    date_match = re.search(r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})', full_string)
    date = date_match.group(1) if date_match else ""
    
    # 2. Check Number (Finds standalone 4-6 digit sequence matching statement page order)
    check_match = re.search(r'\b(4\d{4,5})\b', full_string)
    check_num = check_match.group(1) if check_match else ""
    
    # 3. Clean Amount (Finds standard decimal sequences like $1,045.00)
    amt_match = re.search(r'(\d{1,3}(?:,\d{3})*\.\d{2})', full_string)
    amount = f"${amt_match.group(1)}" if amt_match else ""
    
    return check_num, date, amount

if uploaded_file is not None:
    if st.button("🚀 Run Pipeline"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            temp_pdf_path = temp_pdf.name
            
        progress_bar = st.progress(0)
        status_message = st.empty()
        extracted_data = []
        
        try:
            info = pdf2image.pdfinfo_from_path(temp_pdf_path)
            total_pages = info["Pages"]
            
            for page_idx in range(1, total_pages + 1):
                status_message.text(f"Processing structural grids on Page {page_idx} of {total_pages}...")
                
                pages = pdf2image.convert_from_path(
                    temp_pdf_path, dpi=200, first_page=page_idx, last_page=page_idx
                )
                if not pages:
                    continue
                
                page_img = pages[0]
                width, height = page_img.size
                
                # 3 Columns wide, 8 Rows deep (Alternative Check Face & Bank Data Block layers)
                cols, rows = 3, 8
                cell_w = width // cols
                cell_h = height // rows
                
                # Step through the spreadsheet rows in pairs of 2
                for r in range(0, rows, 2):
                    for c in range(cols):
                        
                        # --- BOX A: THE CHECK IMAGE FACE ---
                        left_a, top_a = c * cell_w, r * cell_h
                        right_a, bottom_a = (c + 1) * cell_w, (r + 1) * cell_h
                        crop_check = page_img.crop((left_a, top_a, right_a, bottom_a))
                        
                        # --- BOX B: THE BANK STATEMENT TEXT BLOCK BELOW IT ---
                        left_b, top_b = c * cell_w, (r + 1) * cell_h
                        right_b, bottom_b = (c + 1) * cell_w, (r + 2) * cell_h
                        crop_bank_text = page_img.crop((left_b, top_b, right_b, bottom_b))
                        
                        # Execute Dual-Zone OCR Scans
                        lines_check = reader.readtext(np.array(crop_check), detail=0)
                        lines_bank = reader.readtext(np.array(crop_bank_text), detail=0)
                        
                        if lines_bank or lines_check:
                            payee = extract_check_face_metadata(lines_check)
                            check_num, clean_date, amount = extract_bank_block_metadata(lines_bank)
                            
                            extracted_data.append({
                                "Check Number": check_num if check_num else "Review Page",
                                "Date": clean_date if clean_date else "Review Page",
                                "Amount": amount if amount else "Review Page",
                                "Payee": payee,
                                "Statement Page": page_idx
                            })
                            
                progress_bar.progress(page_idx / total_pages)
                
            status_message.text("Compiling extracted structural data matrix...")
            
            # Formulate structured dataset preview
            df = pd.DataFrame(extracted_data)
            st.success(f"Processing Complete! Successfully extracted check matrices.")
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Clean Dataset to CSV",
                data=csv,
                file_name="bank_statement_extracted_checks.csv",
                mime="text/csv",
                type="primary"
            )
            
        except Exception as err:
            st.error(f"Pipeline processing failed: {err}")
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
