"""
Advanced PDF Processing Module
============================
Enhanced PDF processing with table extraction, structured content handling,
and improved metadata extraction for better RAG performance.

Features:
- Table extraction and conversion to markdown
- Structured content identification (headers, lists, tables)
- Enhanced metadata extraction
- Content type-aware chunking
- Table summarization for better searchability
- Standard PDF processing with semantic chunking
- File change detection and incremental processing
"""

import os
import re
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

# PDF processing libraries
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from operator import itemgetter

# NLP libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# LlamaIndex components for document creation
from llama_index.core import Document

# Set up logging first
logger = logging.getLogger(__name__)

# Table extraction libraries
try:
    import camelot
    CAMELOT_AVAILABLE = True
    # Uncomment next line to temporarily disable Camelot if issues persist
    # CAMELOT_AVAILABLE = False
except ImportError:
    CAMELOT_AVAILABLE = False
    logger.warning("Camelot not available - install with: pip install camelot-py")

try:
    import tabula
    import subprocess
    # Check if Java is available
    try:
        subprocess.run(['java', '-version'], capture_output=True, check=True)
        TABULA_AVAILABLE = True
        logger.info("Tabula available with Java support")
    except (subprocess.CalledProcessError, FileNotFoundError):
        TABULA_AVAILABLE = False
        logger.info("Java not available - using PDFplumber and Camelot for table extraction")
except ImportError:
    TABULA_AVAILABLE = False
    logger.warning("Tabula not available - install with: pip install tabula-py")


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass


# ———————— NLTK Setup ————————
def ensure_nltk_data():
    """Download required NLTK data if not present."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)


# ———————— Standard PDF Processing Functions ————————

def get_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of file for change detection."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def extract_pdf_structure(doc: fitz.Document) -> Dict[str, Any]:
    """Extract document structure information."""
    structure = {
        "page_count": len(doc),
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "subject": doc.metadata.get("subject", ""),
        "creator": doc.metadata.get("creator", ""),
        "producer": doc.metadata.get("producer", ""),
        "creation_date": doc.metadata.get("creationDate", ""),
        "modification_date": doc.metadata.get("modDate", ""),
        "toc": [],
        "has_images": False,
        "has_tables": False,
    }
    
    # Extract table of contents
    try:
        toc = doc.get_toc()
        structure["toc"] = [(level, title, page) for level, title, page in toc]
    except:
        pass
    
    return structure


def get_toc_with_hierarchy(doc: fitz.Document) -> List[Dict[str, Any]]:
    """
    Extracts the Table of Contents with proper level and hierarchy.
    Levels are 1-based.
    """
    toc_raw = doc.get_toc(simple=False)
    toc_structured = []
    
    if not toc_raw:
        return []

    for level, title, page_num, dest in toc_raw:
        # dest provides more accurate location info than page_num alone
        toc_structured.append({
            "level": level,
            "title": title.strip(),
            "page": page_num,
            "y_pos": dest.get("to", fitz.Point(0, 0)).y
        })
    return toc_structured

def build_section_map(toc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Builds a map of sections with their full hierarchical titles and page ranges.
    """
    section_map = []
    path = []

    for item in toc:
        level = item['level']
        title = item['title']
        page = item['page']

        # Adjust path to the current level
        path = path[:level-1]
        path.append(title)
        
        # Create full hierarchical title
        full_title = " > ".join(path)
        
        section_map.append({
            "page": page,
            "title": title,
            "full_title": full_title,
            "level": level,
            "y_pos": item.get('y_pos', 0)
        })

    # Sort by page then y-position to handle multiple sections on a single page
    section_map.sort(key=itemgetter('page', 'y_pos'))
    
    return section_map

def get_section_for_page(page_num: int, section_map: List[Dict[str, Any]]) -> str:
    """
    Finds the most specific section title for a given page number.
    """
    if not section_map:
        return f"Content from Page {page_num}"

    # Find all sections that start on or before the given page
    relevant_sections = [s for s in section_map if s['page'] <= page_num]
    
    if not relevant_sections:
        return f"Content from Page {page_num}"

    # The last section in the sorted list that starts on or before this page is the correct one
    # This assumes the section_map is sorted by page number.
    return relevant_sections[-1]['full_title']

def extract_sections_from_text(doc: fitz.Document) -> List[Dict[str, Any]]:
    """
    Extract section headers from text content using regex patterns.
    This is a fallback when PDF has no embedded ToC structure.
    """
    sections = []
    
    # Define patterns for different section levels
    section_patterns = [
        # Level 3: 5.4.1. Installing a QKD Blade
        (3, r'^\s*(\d+\.\d+\.\d+\.)\s+([A-Z][^\n\r]*)', r'^\s*\d+\.\d+\.\d+\.\s+'),
        # Level 2: 5.4. QKD Blade  
        (2, r'^\s*(\d+\.\d+\.)\s+([A-Z][^\n\r]*)', r'^\s*\d+\.\d+\.\s+'),
        # Level 1: 5. Installation
        (1, r'^\s*(\d+\.)\s+([A-Z][^\n\r]*)', r'^\s*\d+\.\s+')
    ]
    
    logger.info("Scanning PDF text for section headers...")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        lines = text.split('\n')
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Try each pattern level
            for level, pattern, validation_pattern in section_patterns:
                import re
                match = re.match(pattern, line, re.IGNORECASE | re.MULTILINE)
                if match:
                    number = match.group(1).strip()
                    title = match.group(2).strip()
                    
                    # Validate it's a real section header
                    if len(title) > 3 and len(title) < 100:  # Reasonable title length
                        section = {
                            'level': level,
                            'number': number,
                            'title': title,
                            'full_text': f"{number} {title}",
                            'page': page_num + 1,
                            'line_idx': line_idx
                        }
                        sections.append(section)
                        logger.debug(f"Found section: Level {level}, Page {page_num + 1}, '{number} {title}'")
    
    logger.info(f"Found {len(sections)} section headers via text analysis")
    return sections

def build_hierarchical_section_map(text_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a hierarchical section map from text-extracted sections.
    """
    if not text_sections:
        return []
    
    # Sort sections by page number and line index
    sorted_sections = sorted(text_sections, key=lambda x: (x['page'], x['line_idx']))
    
    section_map = []
    hierarchy = ['', '', '']  # Track current level 1, level 2, level 3 titles
    
    for section in sorted_sections:
        level = section['level']
        title = section['title']
        
        # Update hierarchy at current level and clear deeper levels
        hierarchy[level - 1] = title
        for i in range(level, 3):
            hierarchy[i] = ''
        
        # Build full hierarchical title
        path_parts = [part for part in hierarchy[:level] if part]
        full_title = ' > '.join(path_parts)
        
        section_entry = {
            'page': section['page'],
            'title': title,
            'full_title': full_title,
            'level': level,
            'number': section['number']
        }
        section_map.append(section_entry)
    
    return section_map

def get_section_for_page_text_based(page_num: int, text_section_map: List[Dict[str, Any]]) -> str:
    """
    Find the most specific section title for a given page using text-based section map.
    """
    if not text_section_map:
        return f"Content from Page {page_num}"
    
    # Find the last section that starts on or before this page
    applicable_sections = [s for s in text_section_map if s['page'] <= page_num]
    
    if not applicable_sections:
        return f"Content from Page {page_num}"
    
    # Get the section with the highest level (most specific)
    best_section = max(applicable_sections, key=lambda x: (x['page'], x['level']))
    return best_section['full_title']


def extract_text_with_structure(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    """Extract text from PDF with enhanced structure information."""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        
        # Try ToC-based extraction first
        toc = get_toc_with_hierarchy(doc)
        section_map = build_section_map(toc)
        
        # If no ToC found, use text-based section detection
        if not toc or not section_map:
            logger.info("No PDF ToC found, using text-based section detection")
            text_sections = extract_sections_from_text(doc)
            section_map = build_hierarchical_section_map(text_sections)
            structure = extract_pdf_structure(doc)
            structure["text_sections"] = text_sections
            structure["section_map"] = section_map
            structure["extraction_method"] = "text_based"
        else:
            structure = extract_pdf_structure(doc)
            structure["toc"] = toc
            structure["section_map"] = section_map
            structure["extraction_method"] = "toc_based"
        
        page_texts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text with position information
            text_dict = page.get_text("dict")
            page_text = ""
            
            # Process text blocks
            for block in text_dict["blocks"]:
                if "lines" in block:  # Text block
                    block_text = ""
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        block_text += line_text + " "
                    page_text += block_text + "\n"
            
            # Clean up text
            page_text = re.sub(r'\s+', ' ', page_text.strip())
            if page_text:
                page_texts.append({
                    "page_num": page_num + 1,
                    "text": page_text,
                    "char_count": len(page_text)
                })
                full_text += f"\n\n[Page {page_num + 1}]\n{page_text}"
        
        doc.close()
        
        # Add processing statistics
        structure["pages"] = page_texts
        structure["total_chars"] = len(full_text)
        structure["processing_date"] = datetime.now().isoformat()
        
        return full_text, structure
        
    except Exception as e:
        raise PDFProcessingError(f"Failed to extract text from {pdf_path}: {e}")




def semantic_chunk_text(text: str, chunk_size: int = 512, overlap: int = 64, min_chunk_size: int = 100) -> List[str]:
    """
    Perform semantic chunking that preserves sentence boundaries.
    """
    ensure_nltk_data()

    # Split into sentences
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Clean sentence
        sentence = sentence.strip()
        if not sentence:
            continue

        # Calculate potential chunk size
        potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

        # Chunking logic
        if len(potential_chunk) <= chunk_size:
            current_chunk = potential_chunk
        else:
            # Save current chunk if it meets minimum size
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())

            # Start new chunk with overlap
            if overlap > 0 and current_chunk:
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap//4:]) if len(words) > overlap//4 else ""
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                current_chunk = sentence

    # Add final chunk
    if len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk.strip())

    return chunks


def process_single_pdf(pdf_path: Path, existing_metadata: Dict, 
                      chunk_size: int = 512, overlap: int = 64) -> Tuple[List[Document], Dict[str, Any]]:
    """Process a single PDF file with enhanced error handling and structured metadata."""
    pdf_name = pdf_path.name
    file_hash = get_file_hash(str(pdf_path))
    
    # Check if file needs processing
    if pdf_name in existing_metadata:
        if existing_metadata[pdf_name].get("file_hash") == file_hash:
            logger.info(f"[DOC] {pdf_name}: Unchanged, skipping...")
            return [], existing_metadata[pdf_name]
    
    logger.info(f"[DOC] Processing {pdf_name}...")
    
    try:
        # Extract text and structure
        full_text, structure = extract_text_with_structure(str(pdf_path))
        
        if not full_text.strip():
            logger.warning(f"[DOC] {pdf_name}: No text content extracted")
            return [], {}
        
        # Create structured chunks with page-level metadata
        documents = create_structured_chunks_from_pdf(pdf_path, structure, chunk_size, overlap)
        
        if not documents:
            logger.warning(f"[DOC] {pdf_name}: No documents created")
            return [], {}
        
        # Update metadata
        pdf_metadata = {
            "file_hash": file_hash,
            "processing_date": datetime.now().isoformat(),
            "chunk_count": len(documents),
            "character_count": len(full_text),
            "structure": structure,
            "processing_method": "structured_pdf",
        }
        
        logger.info(f"✅ {pdf_name}: Created {len(documents)} structured chunks ({len(full_text):,} chars)")
        return documents, pdf_metadata
        
    except Exception as e:
        logger.error(f"❌ {pdf_name}: Processing failed - {e}")
        raise PDFProcessingError(f"Failed to process {pdf_name}: {e}")


def create_structured_chunks_from_pdf(pdf_path: Path, structure: Dict, chunk_size: int, overlap: int) -> List[Document]:
    """Create structured chunks from PDF with proper page and section metadata."""
    documents = []
    pdf_name = pdf_path.name
    section_map = structure.get("section_map", [])
    extraction_method = structure.get("extraction_method", "unknown")
    
    try:
        doc = fitz.open(str(pdf_path))
        
        for page_info in structure.get("pages", []):
            page_num = page_info["page_num"]
            page_text = page_info["text"]
            
            if not page_text.strip():
                continue
            
            # Use appropriate section detection method
            if extraction_method == "text_based":
                section_header = get_section_for_page_text_based(page_num, section_map)
            else:
                section_header = get_section_for_page(page_num, section_map)
            
            # Create chunks for this page
            chunks = semantic_chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
            
            for i, chunk in enumerate(chunks):
                # Create document with structured metadata
                doc_metadata = {
                    "source": pdf_name,
                    "source_file": pdf_name,
                    "page": page_num,
                    "page_number": page_num,
                    "section": section_header,
                    "section_header": section_header,
                    "chunk_index": i,
                    "processing_method": f"structured_pdf_{extraction_method}",
                    "processing_date": datetime.now().isoformat(),
                    "extraction_method": extraction_method
                }
                
                document = Document(
                    text=chunk,
                    metadata=doc_metadata
                )
                documents.append(document)
        
        doc.close()
        return documents
        
    except Exception as e:
        logger.error(f"Failed to create structured chunks from {pdf_name}: {e}")
        return []




class AdvancedPDFProcessor:
    """
    Advanced PDF processor with table extraction and structured content handling.
    """
    
    def __init__(self):
        self.ensure_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        
    def ensure_nltk_data(self):
        """Download required NLTK data if not present."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
    
    def extract_tables_camelot(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables using Camelot (more accurate for complex tables)."""
        if not CAMELOT_AVAILABLE:
            return []
        
        try:
            # Extract tables from specific page
            tables = camelot.read_pdf(str(pdf_path), pages=str(page_num + 1), flavor='lattice')
            
            extracted_tables = []
            for i, table in enumerate(tables):
                if table.df is not None and not table.df.empty:
                    # Convert to markdown format
                    markdown_table = self.dataframe_to_markdown(table.df)
                    
                    # Generate table summary
                    summary = self.generate_table_summary(table.df)
                    
                    extracted_tables.append({
                        'table_id': f'page_{page_num + 1}_table_{i + 1}',
                        'page_num': page_num + 1,
                        'markdown': markdown_table,
                        'summary': summary,
                        'shape': table.df.shape,
                        'accuracy': table.accuracy,
                        'method': 'camelot_lattice'
                    })
            
            return extracted_tables
            
        except Exception as e:
            logger.warning(f"Camelot table extraction failed for page {page_num + 1}: {e}")
            return []
    
    def extract_tables_tabula(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables using Tabula (good for simple tables)."""
        if not TABULA_AVAILABLE:
            return []
        
        try:
            # Extract tables from specific page with memory optimization
            tables = tabula.read_pdf(
                str(pdf_path), 
                pages=page_num + 1, 
                multiple_tables=True,
                silent=True,
                java_options=['-Dfile.encoding=UTF-8', '-Xms512m', '-Xmx4096m']
            )
            
            extracted_tables = []
            for i, table_df in enumerate(tables):
                if table_df is not None and not table_df.empty:
                    # Convert to markdown format
                    markdown_table = self.dataframe_to_markdown(table_df)
                    
                    # Generate table summary
                    summary = self.generate_table_summary(table_df)
                    
                    extracted_tables.append({
                        'table_id': f'page_{page_num + 1}_table_{i + 1}',
                        'page_num': page_num + 1,
                        'markdown': markdown_table,
                        'summary': summary,
                        'shape': table_df.shape,
                        'accuracy': 0.8,  # Default accuracy for tabula
                        'method': 'tabula'
                    })
            
            return extracted_tables
            
        except Exception as e:
            logger.warning(f"Tabula table extraction failed for page {page_num + 1}: {e}")
            return []
    
    def extract_tables_pdfplumber(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber (good for text-based tables)."""
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                if page_num >= len(pdf.pages):
                    return []
                
                page = pdf.pages[page_num]
                tables = page.extract_tables()
                
                extracted_tables = []
                for i, table in enumerate(tables):
                    if table and len(table) > 1:  # At least header + 1 row
                        # Convert to DataFrame
                        df = pd.DataFrame(table[1:], columns=table[0])
                        
                        # Clean DataFrame
                        df = df.dropna(how='all').fillna('')
                        
                        if not df.empty:
                            # Convert to markdown format
                            markdown_table = self.dataframe_to_markdown(df)
                            
                            # Generate table summary
                            summary = self.generate_table_summary(df)
                            
                            extracted_tables.append({
                                'table_id': f'page_{page_num + 1}_table_{i + 1}',
                                'page_num': page_num + 1,
                                'markdown': markdown_table,
                                'summary': summary,
                                'shape': df.shape,
                                'accuracy': 0.7,  # Default accuracy for pdfplumber
                                'method': 'pdfplumber'
                            })
                
                return extracted_tables
                
        except Exception as e:
            logger.warning(f"PDFplumber table extraction failed for page {page_num + 1}: {e}")
            return []
    
    def dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to markdown table format."""
        try:
            # Clean column names
            df.columns = [str(col).strip() for col in df.columns]
            
            # Clean data
            df = df.astype(str).replace('nan', '').replace('None', '')
            
            # Convert to markdown
            markdown = df.to_markdown(index=False, tablefmt='pipe')
            return markdown
            
        except Exception as e:
            logger.warning(f"Failed to convert DataFrame to markdown: {e}")
            return str(df)
    
    def generate_table_summary(self, df: pd.DataFrame) -> str:
        """Generate a searchable summary of table content."""
        try:
            rows, cols = df.shape
            
            # Get column names
            columns = [str(col).strip() for col in df.columns if str(col).strip()]
            
            # Extract key values (first few rows of each column)
            key_values = []
            for col in df.columns:
                # Convert column to string series first, then apply string operations
                col_series = df[col].dropna().astype(str)
                values = col_series.str.strip() if hasattr(col_series, 'str') else col_series
                values = values[values != ''].head(3)
                # Convert to list safely
                if hasattr(values, 'tolist'):
                    values_list = values.tolist()
                else:
                    values_list = list(values)
                key_values.extend(values_list)
            
            # Create summary
            summary_parts = [
                f"Table with {rows} rows and {cols} columns",
                f"Columns: {', '.join(columns)}" if columns else "",
                f"Key values: {', '.join(key_values[:10])}" if key_values else ""
            ]
            
            summary = ". ".join(part for part in summary_parts if part)
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to generate table summary: {e}")
            return f"Table with {df.shape[0]} rows and {df.shape[1]} columns"
    
    def extract_structured_content(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract structured content from PDF including text, tables, and metadata.
        """
        logger.info(f"🔍 Extracting structured content from {Path(pdf_path).name}")
        
        # Store current PDF info for reference
        self._current_pdf_path = str(pdf_path)
        self._current_pdf_name = Path(pdf_path).name
        
        structured_content = {
            'text_content': [],
            'tables': [],
            'metadata': {},
            'structure': {
                'total_pages': 0,
                'pages_with_tables': [],
                'content_types': []
            }
        }
        
        try:
            # Open PDF with PyMuPDF for text and metadata
            doc = fitz.open(str(pdf_path))
            structured_content['structure']['total_pages'] = len(doc)
            
            # Extract document metadata
            structured_content['metadata'] = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
            }
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                self._current_page = page_num + 1  # Store for fallback header detection
                
                # Extract text content
                page_text = page.get_text()
                
                if page_text.strip():
                    # Analyze content structure
                    content_analysis = self.analyze_page_content(page_text)
                    
                    # Enhanced page metadata with source information
                    page_data = {
                        'page_num': page_num + 1,
                        'text': page_text,
                        'word_count': len(page_text.split()),
                        'has_headers': content_analysis['has_headers'],
                        'has_lists': content_analysis['has_lists'],
                        'technical_terms': content_analysis['technical_terms'],
                        # ADD SOURCE FILE INFORMATION
                        'source_file': self._current_pdf_name,
                        'file_path': str(pdf_path)
                    }
                    
                    structured_content['text_content'].append(page_data)
                
                # Try multiple table extraction methods in priority order
                page_tables = []
                
                # Method 1: PDFplumber (most reliable, works without Java)
                plumber_tables = self.extract_tables_pdfplumber(pdf_path, page_num)
                page_tables.extend(plumber_tables)
                
                # Method 2: Camelot (most accurate for complex tables, if no PDFplumber results)
                if not page_tables and CAMELOT_AVAILABLE:
                    camelot_tables = self.extract_tables_camelot(pdf_path, page_num)
                    page_tables.extend(camelot_tables)
                
                # Method 3: Tabula (only if Java available and no other results)
                if not page_tables and TABULA_AVAILABLE:
                    tabula_tables = self.extract_tables_tabula(pdf_path, page_num)
                    page_tables.extend(tabula_tables)
                
                # Add tables to structured content
                if page_tables:
                    structured_content['tables'].extend(page_tables)
                    structured_content['structure']['pages_with_tables'].append(page_num + 1)
            
            # INTEGRATE TEXT-BASED SECTION DETECTION
            # Add our text-based section detection to the advanced processor
            doc = fitz.open(str(pdf_path))  # Reopen for section detection
            text_sections = extract_sections_from_text(doc)
            section_map = build_hierarchical_section_map(text_sections)
            doc.close()
            
            # Add section detection results to structured content
            structured_content['text_sections'] = text_sections
            structured_content['section_map'] = section_map
            structured_content['extraction_method'] = 'text_based' if section_map else 'fallback'
            
            # Store section map for chunk creation
            self._section_map = section_map
            
            logger.info(f"Found {len(text_sections)} text-based sections")
            if text_sections:
                installation_sections = [s for s in text_sections if 'install' in s['title'].lower()]
                logger.info(f"Found {len(installation_sections)} installation-related sections")
            
            # Analyze overall content structure
            structured_content['structure']['content_types'] = self.identify_content_types(structured_content)
            
            logger.info(f" Extracted {len(structured_content['text_content'])} pages of text and {len(structured_content['tables'])} tables")
            
            return structured_content
            
        except Exception as e:
            logger.error(f" Failed to extract structured content: {e}")
            raise
    
    def analyze_page_content(self, text: str) -> Dict[str, Any]:
        """Analyze page content to identify structure elements."""
        analysis = {
            'has_headers': False,
            'has_lists': False,
            'technical_terms': []
        }
        
        # Check for headers (numbered sections, capitalized lines)
        header_patterns = [
            r'^\d+\.[\d\.]*\s+[A-Z]',  # Numbered sections
            r'^[A-Z][A-Z\s]{5,}$',     # All caps headers
            r'^\d+\.\s+[A-Z]'          # Simple numbered items
        ]
        
        for pattern in header_patterns:
            if re.search(pattern, text, re.MULTILINE):
                analysis['has_headers'] = True
                break
        
        # Check for lists
        list_patterns = [
            r'^\s*[•\-\*]\s+',         # Bullet points
            r'^\s*\d+\.\s+',           # Numbered lists
            r'^\s*[a-z]\)\s+',         # Lettered lists
        ]
        
        for pattern in list_patterns:
            if re.search(pattern, text, re.MULTILINE):
                analysis['has_lists'] = True
                break
        
        # Extract technical terms
        qkd_terms = [
            'COW', 'QBER', 'visibility', 'quantum', 'photon', 'polarization',
            'key rate', 'loss budget', 'attenuation', 'detector', 'laser',
            'fiber', 'wavelength', 'dBm', 'Cerberis', 'SNMP', 'protocol',
            'interference', 'coherent', 'BB84', 'decoy', 'sifting', 'Alice',
            'Bob', 'Eve', 'cryptography', 'encryption', 'security'
        ]
        
        text_lower = text.lower()
        for term in qkd_terms:
            if term.lower() in text_lower:
                analysis['technical_terms'].append(term)
        
        return analysis
    
    def identify_content_types(self, structured_content: Dict[str, Any]) -> List[str]:
        """Identify the types of content in the document."""
        content_types = []
        
        # Check for tables
        if structured_content['tables']:
            content_types.append('tables')
        
        # Check for technical specifications
        text_content = ' '.join([page['text'] for page in structured_content['text_content']])
        if any(term in text_content.lower() for term in ['specification', 'requirement', 'parameter']):
            content_types.append('specifications')
        
        # Check for procedures/instructions
        if any(term in text_content.lower() for term in ['step', 'procedure', 'instruction', 'guide']):
            content_types.append('procedures')
        
        # Check for troubleshooting content
        if any(term in text_content.lower() for term in ['troubleshoot', 'problem', 'error', 'fault']):
            content_types.append('troubleshooting')
        
        return content_types
    
    def create_enhanced_chunks(self, structured_content: Dict[str, Any], 
                             chunk_size: int = 512, overlap: int = 64) -> List[Dict[str, Any]]:
        """
        Create enhanced chunks that preserve structure and handle different content types.
        """
        chunks = []
        
        # Process text content with structure-aware chunking
        for page_data in structured_content['text_content']:
            page_chunks = self.chunk_text_with_structure(
                page_data['text'], 
                page_data['page_num'],
                chunk_size, 
                overlap,
                page_data
            )
            chunks.extend(page_chunks)
        
        # Process tables as separate chunks
        for table_data in structured_content['tables']:
            table_chunk = self.create_table_chunk(table_data)
            chunks.append(table_chunk)
        
        return chunks
    
    def chunk_text_with_structure(self, text: str, page_num: int, 
                                 chunk_size: int, overlap: int,
                                 page_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create structure-aware text chunks with proper section header detection."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        # Use text-based section detection if available, otherwise fallback
        section_header = self._get_text_based_section_header(page_num)
        if not section_header or section_header.startswith("Content from Page"):
            # Fallback to old method, but filter out watermarks
            detected_header = self._detect_section_header_from_text(text)
            if detected_header and "Property of ID Quantique SA" not in detected_header:
                section_header = detected_header
            else:
                section_header = f"Content from Page {page_num}"
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk
                if len(current_chunk) >= 100:  # Minimum chunk size
                    chunk_data = {
                        'text': current_chunk,
                        'chunk_type': 'text',
                        'page_num': page_num,
                        'chunk_index': chunk_index,
                        'metadata': {
                            'has_headers': page_metadata.get('has_headers', False),
                            'has_lists': page_metadata.get('has_lists', False),
                            'technical_terms': page_metadata.get('technical_terms', []),
                            'word_count': len(current_chunk.split()),
                            'section_header': section_header,
                            'source_file': page_metadata.get('source_file', ''),
                            'page_number': page_num  # Alternative field name for compatibility
                        }
                    }
                    chunks.append(chunk_data)
                    chunk_index += 1
                
                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    words = current_chunk.split()
                    overlap_text = " ".join(words[-overlap//4:]) if len(words) > overlap//4 else ""
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if len(current_chunk) >= 100:
            chunk_data = {
                'text': current_chunk,
                'chunk_type': 'text',
                'page_num': page_num,
                'chunk_index': chunk_index,
                'metadata': {
                    'has_headers': page_metadata.get('has_headers', False),
                    'has_lists': page_metadata.get('has_lists', False),
                    'technical_terms': page_metadata.get('technical_terms', []),
                    'word_count': len(current_chunk.split()),
                    'section_header': section_header,
                    'source_file': page_metadata.get('source_file', ''),
                    'page_number': page_num  # Alternative field name for compatibility
                }
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def _detect_section_header_from_text(self, text: str) -> str:
        """
        Detect section header from page text using enhanced patterns.
        
        This method looks for common section header patterns in technical documents
        and returns the most likely section header for the page.
        """
        lines = text.split('\n')
        
        # Enhanced patterns for technical documents
        header_patterns = [
            # Numbered sections with various formats
            (r'^\s*(\d+\.[\d\.]*)\s+([A-Z][^.\n]{5,50})\s*$', 1),  # "1.2.3 Section Title"
            (r'^\s*(\d+)\s+([A-Z][A-Z\s]{5,50})\s*$', 1),          # "1 SECTION TITLE"
            (r'^\s*(Chapter\s+\d+[:\s]+[A-Z][^.\n]{5,50})\s*$', 1), # "Chapter 1: Title"
            (r'^\s*(Section\s+\d+[:\s]+[A-Z][^.\n]{5,50})\s*$', 1), # "Section 1: Title"
            
            # All caps headers
            (r'^\s*([A-Z][A-Z\s]{8,50})\s*$', 0.8),               # "SYSTEM OVERVIEW"
            
            # Title case headers
            (r'^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,8})\s*$', 0.7), # "System Overview"
            
            # QKD-specific headers
            (r'^\s*(.*(?:QKD|Quantum|Cerberis|Protocol|Configuration)[^.\n]{0,30})\s*$', 0.9),
            
            # Table headers
            (r'^\s*(Table\s+\d+[:\-\s]+[^.\n]{5,50})\s*$', 0.6),   # "Table 1: Description"
        ]
        
        best_header = ""
        best_score = 0
        
        # Check first 10 lines for headers (most likely to contain section headers)
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if not line or len(line) < 3:
                continue
            
            for pattern, score in header_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Adjust score based on position (earlier lines more likely to be headers)
                    position_weight = 1.0 - (i * 0.1)  # Reduce score for later lines
                    final_score = score * position_weight
                    
                    if final_score > best_score:
                        # Extract the header text (usually the full match or a specific group)
                        if match.groups():
                            header_text = match.group(1) if len(match.groups()) == 1 else " ".join(match.groups())
                        else:
                            header_text = match.group(0)
                        
                        best_header = header_text.strip()
                        best_score = final_score
        
        # If no specific header found, try to extract a meaningful title from content
        if not best_header or best_score < 0.3:
            # Look for any line that might be a title (short, descriptive)
            for line in lines[:5]:
                line = line.strip()
                if (line and 
                    10 <= len(line) <= 80 and  # Reasonable title length
                    not line.endswith('.') and  # Not a sentence
                    not line.startswith('•') and  # Not a bullet point
                    not re.match(r'^\d+\s', line) and  # Not a numbered list
                    any(char.isupper() for char in line)):  # Contains uppercase
                    
                    best_header = line
                    break
        
        # Final fallback
        if not best_header:
            best_header = f"Content from Page {getattr(self, '_current_page', 'Unknown')}"
        
        return best_header

    def _get_text_based_section_header(self, page_num: int) -> str:
        """
        Get section header using text-based section detection.
        This method accesses the section map built during extraction.
        """
        # Check if we have a section map from text-based detection
        if hasattr(self, '_section_map') and self._section_map:
            return get_section_for_page_text_based(page_num, self._section_map)
        return ""

    def create_table_chunk(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a specialized chunk for a table with enhanced searchability."""
        page_num = table_data.get('page_num', 'N/A')
        markdown_table = table_data.get('markdown', '')
        summary = table_data.get('summary', '')
        table_id = table_data.get('table_id', f'table_p{page_num}')

        # Enhanced table content processing for better searchability
        searchable_content = self._create_searchable_table_content(
            markdown_table, summary, page_num, table_id
        )

        # Create optimized metadata (keep it concise for LlamaIndex)
        enhanced_metadata = {
            "source_file": getattr(self, '_current_pdf_name', 'Unknown Source'),
            "file_path": getattr(self, '_current_pdf_path', ''),
            "page_num": page_num,
            "page_number": page_num,  # Alternative field name for compatibility
            "section_header": f"Table on Page {page_num}",
            "chunk_type": "table",
            "table_id": table_id,
            "content_type": "table_data",
            "processing_method": "phase2_advanced_table"
        }

        return {
            "text": searchable_content,
            "chunk_type": "table",
            "page_num": page_num,
            "chunk_index": table_data.get('table_index', 0),
            "metadata": enhanced_metadata
        }

    def _create_searchable_table_content(self, markdown_table: str, summary: str, 
                                       page_num: int, table_id: str) -> str:
        """Create highly searchable content from table data."""
        content_parts = []
        
        # Add structured header
        content_parts.append(f"TABLE: {table_id} on Page {page_num}")
        
        # Add summary for semantic search
        if summary:
            content_parts.append(f"SUMMARY: {summary}")
        
        # ENHANCED: Add flattened searchable text for better matching
        flattened_content = self._flatten_table_for_search(markdown_table)
        if flattened_content:
            content_parts.append(f"SEARCHABLE CONTENT: {flattened_content}")
        
        # Add the full markdown table
        if markdown_table:
            content_parts.append(f"TABLE CONTENT:\n{markdown_table}")
        
        # Extract and add searchable key-value pairs from table
        key_value_pairs = self._extract_table_key_values(markdown_table)
        if key_value_pairs:
            content_parts.append(f"KEY-VALUE PAIRS: {', '.join(key_value_pairs)}")
        
        # Add technical terms found in table
        technical_terms = self._extract_technical_terms_from_table(markdown_table)
        if technical_terms:
            content_parts.append(f"TECHNICAL TERMS: {', '.join(technical_terms)}")
        
        return "\n\n".join(content_parts)

    def _flatten_table_for_search(self, markdown_table: str) -> str:
        """Flatten table content into searchable text format."""
        if not markdown_table:
            return ""
        
        # Remove markdown table formatting and extract pure text
        lines = markdown_table.split('\n')
        searchable_terms = []
        
        for line in lines:
            if '|' in line and not line.strip().startswith('|---'):
                # Extract cell content
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                for cell in cells:
                    if cell and len(cell) > 1:
                        # Add the cell content directly
                        searchable_terms.append(cell)
                        
                        # For OID-like patterns, also add without dots for partial matching
                        if re.match(r'^\d+(\.\d+)+$', cell):
                            searchable_terms.append(cell.replace('.', ' '))
                        
                        # For technical terms, add variations
                        if re.match(r'^[a-zA-Z][a-zA-Z0-9]*$', cell):
                            # Add camelCase variations
                            camel_parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', cell).split()
                            if len(camel_parts) > 1:
                                searchable_terms.extend(camel_parts)
        
        return ' '.join(searchable_terms)

    def _extract_table_key_values(self, markdown_table: str) -> List[str]:
        """Extract key-value pairs from table for enhanced searchability."""
        if not markdown_table:
            return []
        
        key_value_pairs = []
        lines = markdown_table.split('\n')
        
        # Skip header separator line (contains |---|---|)
        data_lines = [line for line in lines if line.strip() and not line.strip().startswith('|---')]
        
        if len(data_lines) < 2:  # Need at least header + 1 data row
            return key_value_pairs
        
        # Parse header
        header_line = data_lines[0]
        headers = [col.strip() for col in header_line.split('|') if col.strip()]
        
        # Parse data rows and create key-value pairs
        for line in data_lines[1:]:
            if '|' in line:
                values = [col.strip() for col in line.split('|') if col.strip()]
                
                # Create key-value pairs for each column
                for i, value in enumerate(values):
                    if i < len(headers) and value and value != '-' and len(value) > 1:
                        # Create searchable key-value pair
                        key_value_pairs.append(f"{headers[i]}={value}")
                        
                        # Also add just the value for direct searches
                        if not value.isdigit():  # Skip pure numbers
                            key_value_pairs.append(value)
        
        return key_value_pairs[:20]  # Limit to prevent bloat

    def _extract_technical_terms_from_table(self, markdown_table: str) -> List[str]:
        """Extract technical terms from table content."""
        if not markdown_table:
            return []
        
        # QKD-specific technical patterns
        technical_patterns = [
            r'\b\d+\.\d+\.\d+\.\d+\.\d+\.\d+\.\d+\.\d+\.\d+\.\d+\b',  # OID patterns
            r'\bqkd\w*\b',  # QKD-related terms
            r'\b\w*[Rr]atio\b',  # Ratio terms
            r'\b\w*[Rr]ate\b',   # Rate terms
            r'\b\w*[Bb]udget\b', # Budget terms
            r'\b\w*[Pp]ower\b',  # Power terms
            r'\b\w*[Ll]oss\b',   # Loss terms
            r'\bdB[m]?\b',       # dB, dBm units
            r'\b\d+nm\b',        # Wavelength
            r'\bCOW\b',          # COW protocol
            r'\bQBER\b',         # QBER
            r'\bSNMP\b'          # SNMP
        ]
        
        technical_terms = []
        table_text = markdown_table.lower()
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, table_text, re.IGNORECASE)
            technical_terms.extend(matches)
        
        # Remove duplicates and return
        return list(set(technical_terms))[:10]  # Limit to prevent bloat


# Convenience function for integration
def create_enhanced_pdf_chunks(pdf_path: str, chunk_size: int = 512,
                              overlap: int = 64) -> List[Dict[str, Any]]:
    """
    Create enhanced chunks from PDF with table-aware processing.
    
    Args:
        pdf_path: Path to PDF file
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks
        
    Returns:
        List of enhanced chunks with metadata
    """
    processor = AdvancedPDFProcessor()
    structured_content = processor.extract_structured_content(pdf_path)
    return processor.create_enhanced_chunks(structured_content, chunk_size, overlap)

 