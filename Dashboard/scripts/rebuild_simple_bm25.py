"""
Manual BM25 rebuild script.
Run this when you want to rebuild the index.

Usage:
    python rebuild_simple_bm25.py
"""

import logging
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Dashboard.services.unified_db_manager import UnifiedDBManager
from Dashboard.services.simple_bm25_manager import SimpleBM25Manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def rebuild_bm25_index():
    """
    Rebuild BM25 index manually.
    """
    logger.info("=" * 50)
    logger.info("Starting BM25 Index Rebuild")
    logger.info("=" * 50)

    try:
        # Initialize database manager
        logger.info("Connecting to database...")
        db_manager = UnifiedDBManager()

        # Initialize BM25 manager
        logger.info("Initializing BM25 manager...")
        bm25_manager = SimpleBM25Manager(db_manager)

        # Check if index exists
        if bm25_manager.bm25_index:
            logger.info(f"Existing index found with {len(bm25_manager.chunk_ids)} chunks")
            response = input("Replace existing index? (y/n): ")
            if response.lower() != 'y':
                logger.info("Rebuild cancelled by user")
                return

        # Rebuild index
        logger.info("Starting index rebuild...")
        num_chunks = bm25_manager.rebuild_index()

        if num_chunks > 0:
            logger.info("=" * 50)
            logger.info(f"✅ BM25 index rebuilt successfully!")
            logger.info(f"   Total chunks indexed: {num_chunks}")
            logger.info(f"   Index saved to: {bm25_manager.index_path}")
            logger.info("=" * 50)
        else:
            logger.warning("No chunks found to index")

        return num_chunks

    except Exception as e:
        logger.error(f"❌ Rebuild failed: {e}")
        logger.error(f"Error details: ", exc_info=True)
        return 0

if __name__ == "__main__":
    rebuild_bm25_index()