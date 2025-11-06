-- ============================================
-- Vector Storage Tables for RAG System
-- ============================================

USE `qkd`;

-- Documents table for storing document metadata
CREATE TABLE IF NOT EXISTS `documents` (
    `id` INT(11) NOT NULL AUTO_INCREMENT,
    `filename` VARCHAR(255) NOT NULL,
    `display_name` VARCHAR(255) NOT NULL,
    `file_path` TEXT,
    `chunk_count` INT(11) DEFAULT 0,
    `status` VARCHAR(20) DEFAULT 'active',
    `metadata` JSON,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    INDEX `idx_filename` (`filename`),
    INDEX `idx_status` (`status`),
    INDEX `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- Document chunks table for storing text chunks with embeddings
CREATE TABLE IF NOT EXISTS `document_chunks` (
    `id` INT(11) NOT NULL AUTO_INCREMENT,
    `document_id` INT(11) NOT NULL,
    `chunk_index` INT(11) NOT NULL,
    `content` TEXT NOT NULL,
    `embedding` JSON COMMENT 'Vector embedding as JSON array',
    `metadata` JSON,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    INDEX `idx_document_id` (`document_id`),
    INDEX `idx_chunk_index` (`chunk_index`),
    FOREIGN KEY (`document_id`) REFERENCES `documents`(`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

