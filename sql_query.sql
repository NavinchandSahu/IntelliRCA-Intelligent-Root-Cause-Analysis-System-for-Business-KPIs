CREATE DATABASE IF NOT EXISTS rca_system;
USE rca_system;
CREATE TABLE IF NOT EXISTS upload_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    upload_date DATE NOT NULL,
    file_path VARCHAR(500) NOT NULL
);
ALTER TABLE upload_history
ADD COLUMN upload_time TIME AFTER upload_date;
DROP DATABASE IF EXISTS rca_system;

