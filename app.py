import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from huggingface_hub import HfApi, upload_folder
import threading
import time
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'py', 'js', 'html', 'css', 'json', 'md', 'zip'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for tracking upload progress
upload_progress = {}
upload_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_zip(zip_path, extract_to):
    """Extract ZIP file to specified directory"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def create_zip_from_files(files, zip_path):
    """Create a ZIP file from uploaded files"""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            if file.filename:
                # Save file temporarily
                temp_path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
                file.save(temp_path)
                
                # Add to ZIP with the original filename
                zipf.write(temp_path, file.filename)
                
                # Clean up temp file
                os.remove(temp_path)

def upload_to_huggingface(upload_id, folder_path, repo_id, hf_token, commit_message, repo_type="space"):
    """Upload folder to Hugging Face in a separate thread"""
    try:
        logger.info(f"Starting upload {upload_id} to {repo_id}")
        upload_status[upload_id] = "Initializing..."
        upload_progress[upload_id] = 0
        
        # Initialize API
        logger.info(f"Initializing HfApi for {upload_id}")
        api = HfApi(token=hf_token)
        
        # Test token validity
        try:
            logger.info(f"Testing token validity for {upload_id}")
            user_info = api.whoami()
            logger.info(f"Token valid for user: {user_info.get('name', 'Unknown')}")
        except Exception as e:
            logger.error(f"Token validation failed for {upload_id}: {str(e)}")
            upload_status[upload_id] = f"Invalid token: {str(e)}"
            return
        
        upload_status[upload_id] = "Uploading to Hugging Face..."
        upload_progress[upload_id] = 25
        
        logger.info(f"Starting upload for {upload_id}: {folder_path} -> {repo_id}")
        
        # Always upload as folder to maintain directory structure
        if os.path.isfile(folder_path):
            logger.info(f"Converting single file to folder structure: {folder_path}")
            # Create a temporary folder for single file
            temp_folder = os.path.join(os.path.dirname(folder_path), f"temp_folder_{upload_id}")
            os.makedirs(temp_folder, exist_ok=True)
            
            # Copy the file to the temporary folder
            filename = os.path.basename(folder_path)
            temp_file_path = os.path.join(temp_folder, filename)
            shutil.copy2(folder_path, temp_file_path)
            
            # Upload the temporary folder
            upload_folder(
                folder_path=temp_folder,
                repo_id=repo_id,
                repo_type=repo_type,
                token=hf_token,
                commit_message=commit_message,
                ignore_patterns=[".git", "__pycache__", "*.pyc", ".env", "*.log"]
            )
            
            # Clean up temporary folder
            shutil.rmtree(temp_folder)
            
        else:
            logger.info(f"Uploading folder: {folder_path}")
            # Folder upload
            upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                repo_type=repo_type,
                token=hf_token,
                commit_message=commit_message,
                ignore_patterns=[".git", "__pycache__", "*.pyc", ".env", "*.log"]
            )
        
        upload_progress[upload_id] = 100
        upload_status[upload_id] = "Upload completed successfully!"
        logger.info(f"Upload {upload_id} completed successfully")
        
    except Exception as e:
        error_msg = f"Upload error: {str(e)}"
        logger.error(f"Upload {upload_id} failed: {error_msg}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        upload_status[upload_id] = error_msg
        upload_progress[upload_id] = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Get form data
        hf_token = request.form.get('hf_token')
        repo_id = request.form.get('repo_id')
        repo_type = request.form.get('repo_type', 'space')
        commit_message = request.form.get('commit_message', 'Upload via Flask App')
        
        logger.info(f"Upload request: repo_id={repo_id}, repo_type={repo_type}")
        
        if not hf_token or not repo_id:
            return jsonify({'error': 'Hugging Face token and repository ID are required'}), 400
        
        # Check if files were uploaded
        if 'files' not in request.files:
            return jsonify({'error': 'No files selected'}), 400
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Generate unique upload ID and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_id = f"upload_{timestamp}"
        
        # Handle multiple files and folder structures
        if len(files) == 1 and not any('/' in file.filename for file in files):
            # Single file upload
            file = files[0]
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = f"{timestamp}_{filename}"
                
                # Save uploaded file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                logger.info(f"File saved: {file_path}")
                
                # Handle ZIP files
                if filename.lower().endswith('.zip'):
                    extract_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"extracted_{timestamp}")
                    os.makedirs(extract_folder, exist_ok=True)
                    extract_zip(file_path, extract_folder)
                    upload_path = extract_folder
                    logger.info(f"ZIP extracted to: {extract_folder}")
                else:
                    upload_path = file_path
            else:
                return jsonify({'error': 'File type not allowed'}), 400
        else:
            # Multiple files or folder structure - create folder structure
            folder_path = os.path.join(app.config['UPLOAD_FOLDER'], f"folder_{timestamp}")
            os.makedirs(folder_path, exist_ok=True)
            
            # Save all files to the folder, preserving directory structure
            for file in files:
                if file and file.filename and allowed_file(file.filename):
                    # Use the original filename which may include directory structure
                    filename = file.filename
                    # Ensure the filename is safe but preserve directory structure
                    safe_filename = filename.replace('\\', '/').strip('/')
                    file_path = os.path.join(folder_path, safe_filename)
                    
                    # Create subdirectories if needed (for folder uploads)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    file.save(file_path)
                    logger.info(f"File saved: {file_path}")
                elif file and file.filename:
                    return jsonify({'error': f'File type not allowed: {file.filename}'}), 400
            
            upload_path = folder_path
            logger.info(f"Folder created: {folder_path}")
        
        # Start upload in background thread
        thread = threading.Thread(
            target=upload_to_huggingface,
            args=(upload_id, upload_path, repo_id, hf_token, commit_message, repo_type)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'upload_id': upload_id,
            'message': 'Upload started successfully!'
        })
        
    except Exception as e:
        logger.error(f"Upload endpoint error: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/progress/<upload_id>')
def get_progress(upload_id):
    """Get upload progress"""
    progress = upload_progress.get(upload_id, 0)
    status = upload_status.get(upload_id, "Unknown")
    
    return jsonify({
        'progress': progress,
        'status': status,
        'completed': progress == 100 and 'Error' not in status
    })

@app.route('/cleanup')
def cleanup():
    """Clean up old uploaded files"""
    try:
        # Remove files older than 1 hour
        cutoff_time = time.time() - 3600  # 1 hour
        
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.getctime(file_path) < cutoff_time:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        
        return jsonify({'message': 'Cleanup completed'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))  # Hugging Face Spaces uses port 7860
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)