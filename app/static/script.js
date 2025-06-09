// Simple form submission feedback
const form = document.getElementById('uploadForm');
const loader = document.getElementById('loader');
const statusMessage = document.getElementById('statusMessage');
const uploadArea = document.querySelector('.upload-area');
const fileInput = document.getElementById('file');
const uploadText = document.querySelector('.upload-text');
const uploadHint = document.querySelector('.upload-hint');

form.addEventListener('submit', function() {
    loader.style.display = 'block';
    statusMessage.innerHTML = '';
});

// Drag and drop functionality
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary)';
    uploadArea.style.backgroundColor = 'rgba(30, 41, 59, 0.7)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'rgba(108, 99, 255, 0.3)';
    uploadArea.style.backgroundColor = 'rgba(30, 41, 59, 0.5)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'rgba(108, 99, 255, 0.3)';
    uploadArea.style.backgroundColor = 'rgba(30, 41, 59, 0.5)';
    
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        updateFileNameDisplay();
    }
});

// Show selected file name
fileInput.addEventListener('change', function() {
    if (this.files.length) {
        updateFileNameDisplay();
    }
});

function updateFileNameDisplay() {
    if (fileInput.files.length) {
        uploadText.textContent = fileInput.files[0].name;
        uploadHint.textContent = 'Ready to process';
        
        // Add success visual feedback
        uploadArea.style.borderColor = 'var(--success)';
        setTimeout(() => {
            uploadArea.style.borderColor = 'rgba(108, 99, 255, 0.3)';
        }, 1000);
    }
}

// Display any flash messages from Flask
document.addEventListener('DOMContentLoaded', function() {
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(message => {
        setTimeout(() => {
            message.style.opacity = '0';
            setTimeout(() => message.remove(), 500);
        }, 3000);
    });
});