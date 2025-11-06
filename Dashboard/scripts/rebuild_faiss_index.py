"""
Manual FAISS GPU index rebuild script.
Run this when you want to rebuild the FAISS index.

Usage:
    python rebuild_faiss_index.py
"""

import logging
import sys
import os
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Dashboard.services.unified_db_manager import UnifiedDBManager
from Dashboard.services.faiss_gpu_manager import FAISSManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def rebuild_faiss_index():
    """
    Rebuild FAISS index manually.
    """
    logger.info("=" * 60)
    logger.info("FAISS Index Rebuild")
    logger.info("=" * 60)

    try:
        # Initialize database manager
        logger.info("Connecting to database...")
        db_manager = UnifiedDBManager()

        # Initialize FAISS manager
        logger.info("Initializing FAISS manager...")
        faiss_manager = FAISSManager(db_manager)

        # Check current index status
        current_stats = faiss_manager.get_index_stats()
        logger.info(f"Current index status: {current_stats}")

        if current_stats['status'] == 'ready':
            logger.info(f"Existing index found with {current_stats['num_vectors']} vectors")
            logger.info(f"Index type: {current_stats.get('index_type', 'Unknown')}")
            logger.info(f"GPU enabled: {current_stats.get('use_gpu', False)}")

            response = input("\nReplace existing index? (y/n): ")
            if response.lower() != 'y':
                logger.info("Rebuild cancelled by user")
                return

        # Start rebuild
        logger.info("\nStarting FAISS index rebuild...")
        logger.info("This may take 1-2 minutes for 500k vectors...")

        start_time = time.time()

        # Rebuild index
        result = faiss_manager.rebuild_index()

        if result['status'] == 'success':
            duration = time.time() - start_time

            logger.info("\n" + "=" * 60)
            logger.info("✅ FAISS index rebuilt successfully!")
            logger.info("=" * 60)
            logger.info(f"Statistics:")
            logger.info(f"  Vectors indexed: {result['vectors_indexed']:,}")
            logger.info(f"  Time taken: {duration:.1f} seconds")
            logger.info(f"  Processing speed: {result['vectors_per_second']:.0f} vectors/sec")
            logger.info(f"  Index type: {result.get('index_type', 'IVFFlat')}")
            logger.info(f"  Index saved to: {faiss_manager.index_path}")
            logger.info("=" * 60)

            # Verify index
            new_stats = faiss_manager.get_index_stats()
            logger.info("\nVerification:")
            logger.info(f"  Index loaded: {'✓' if new_stats['status'] == 'ready' else '✗'}")
            logger.info(f"  Vectors available: {new_stats.get('num_vectors', 0):,}")
            logger.info(f"  Memory usage: {new_stats.get('memory_usage_mb', 0):.1f} MB")

        else:
            logger.error(f"❌ Rebuild failed: {result.get('message', 'Unknown error')}")

        return result

    except Exception as e:
        logger.error(f"❌ Rebuild failed with exception: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

def check_faiss_status():
    """Check FAISS index status without rebuilding."""
    try:
        db_manager = UnifiedDBManager()
        faiss_manager = FAISSManager(db_manager)

        stats = faiss_manager.get_index_stats()

        logger.info("\n" + "=" * 60)
        logger.info("FAISS Index Status")
        logger.info("=" * 60)

        if stats['status'] == 'ready':
            logger.info(f"✓ Index is ready")
            logger.info(f"  Vectors: {stats['num_vectors']:,}")
            logger.info(f"  Dimension: {stats['dimension']}")
            logger.info(f"  GPU enabled: {stats['use_gpu']}")
            logger.info(f"  Memory: {stats['memory_usage_mb']:.1f} MB")
        else:
            logger.info(f"✗ Index not built")
            logger.info(f"  Run rebuild to create index")

        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to check status: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='FAISS GPU Index Management')
    parser.add_argument('--status', action='store_true',
                       help='Check index status without rebuilding')
    parser.add_argument('--force', action='store_true',
                       help='Force rebuild without confirmation')

    args = parser.parse_args()

    if args.status:
        check_faiss_status()
    else:
        if args.force:
            logger.info("Force rebuild mode - skipping confirmation")
        rebuild_faiss_index()