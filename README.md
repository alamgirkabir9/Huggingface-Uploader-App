# HuggingFace Uploader

A Flask web application for uploading files and folders to Hugging Face Spaces, Models, and Datasets.

## Features

- 🌐 **Web-based Interface**: Clean, modern UI with drag-and-drop support
- 📁 **ZIP Support**: Upload entire folders by compressing them to ZIP
- 🚀 **Multiple Repository Types**: Support for Spaces, Models, and Datasets
- 📊 **Progress Tracking**: Real-time upload progress monitoring
- 🔄 **Background Processing**: Non-blocking uploads with threading
- 🧹 **Auto Cleanup**: Automatic cleanup of temporary files
- 📱 **Responsive Design**: Works on desktop and mobile devices
  ## 🌐 Live Demo  
**Deployed on huggingface**: [(https://huggingface.co/spaces/Alamgirapi/HuggingFace_Uploader)]

## Installation

1. **Clone or download the files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure

```
huggingface-uploader/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── templates/
│   └── index.html     # Web interface template
└── uploads/           # Temporary upload directory (auto-created)
```

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:5000`

3. **Fill in the form**:
   - **Hugging Face Token**: Get from [HuggingFace Settings](https://huggingface.co/settings/tokens)
   - **Repository ID**: Format as `username/repository-name`
   - **Repository Type**: Choose between Space, Model, or Dataset
   - **Commit Message**: Describe your upload
   - **File**: Select a file or ZIP folder

4. **Upload**: Click "Upload to HuggingFace" and monitor progress

## Supported File Types

- **Archives**: ZIP
- **Code**: Python (.py), JavaScript (.js), HTML (.html), CSS (.css), JSON (.json)
- **Documents**: Text (.txt), Markdown (.md), PDF (.pdf)
- **Images**: PNG, JPG, JPEG, GIF

## Configuration

### Environment Variables

You can customize the application by modifying these variables in `app.py`:

```python
UPLOAD_FOLDER = 'uploads'                    # Temporary upload directory
MAX_CONTENT_LENGTH = 500 * 1024 * 1024     # Max file size (500MB)
SECRET_KEY = 'your-secret-key-change-this'  # Flask secret key
```

### Security

- Change the `SECRET_KEY` in production
- Consider adding authentication for production use
- Implement rate limiting for public deployments

## API Endpoints

- `GET /` - Main upload interface
- `POST /upload` - Handle file uploads
- `GET /progress/<upload_id>` - Get upload progress
- `GET /cleanup` - Manual cleanup of old files

## Error Handling

The application includes comprehensive error handling for:
- Invalid file types
- Missing required fields
- HuggingFace API errors
- Network connectivity issues
- File size limitations

## Auto-Cleanup

The application automatically cleans up temporary files older than 1 hour. You can also trigger manual cleanup by visiting `/cleanup`.

## Development

To run in development mode:
```bash
python app.py
```

The application runs on `http://localhost:5000` with debug mode enabled.

## Deployment

For production deployment:

1. Set `debug=False` in `app.py`
2. Change the `SECRET_KEY` to a secure random value
3. Configure a proper web server (e.g., nginx + gunicorn)
4. Set up proper file permissions
5. Consider implementing authentication

## Troubleshooting

### Common Issues

1. **"No module named 'huggingface_hub'"**
   - Run: `pip install -r requirements.txt`

2. **"Permission denied" errors**
   - Ensure the application has write permissions to the uploads directory

3. **Upload fails with authentication error**
   - Verify your HuggingFace token is correct and has proper permissions

4. **File too large**
   - Check if your file exceeds the 500MB limit

### Logs

The application logs errors to the console. Check the terminal output for detailed error messages.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For support, please:
1. Check the troubleshooting section above
2. Review the HuggingFace documentation
3. Open an issue on the project repository
