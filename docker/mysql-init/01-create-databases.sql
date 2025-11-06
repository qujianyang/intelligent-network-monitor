-- ============================================
-- Create Databases for QKD System
-- ============================================

-- Create main QKD database
CREATE DATABASE IF NOT EXISTS `qkd`
    DEFAULT CHARACTER SET utf8mb4
    COLLATE utf8mb4_general_ci;

-- Create ML database
CREATE DATABASE IF NOT EXISTS `qkd_ml`
    DEFAULT CHARACTER SET utf8mb4
    COLLATE utf8mb4_general_ci;

-- Grant privileges to root user (for Docker environment)
GRANT ALL PRIVILEGES ON qkd.* TO 'root'@'%';
GRANT ALL PRIVILEGES ON qkd_ml.* TO 'root'@'%';
FLUSH PRIVILEGES;