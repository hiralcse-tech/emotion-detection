def handle_model_loading():
    """Handle model file upload or download from Hugging Face"""
    model_path = "emotion_model.pth"
    
    if not os.path.exists(model_path):
        st.warning("⚠️ Model file not found!")
        
        # Create tabs for different options
        tab1, tab2, tab3 = st.tabs(["📥 Download from URL", "📁 Upload File", "🔗 Hugging Face"])
        
        # Tab 1: Direct URL download
        with tab1:
            st.markdown("### Download from Direct URL")
            model_url = st.text_input(
                "Enter model URL:", 
                placeholder="https://huggingface.co/username/model/resolve/main/emotion_model.pth",
                key="url_input"
            )
            
            if model_url and st.button("📥 Download Model", key="download_url"):
                download_model_from_url(model_url, model_path)
        
        # Tab 2: File upload
        with tab2:
            st.markdown("### Upload Model File")
            st.info("Max file size: 200MB")
            uploaded_file = st.file_uploader(
                "Upload emotion_model.pth",
                type=['pth', 'pt'],
                key="file_upload"
            )
            
            if uploaded_file is not None:
                with open(model_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                st.success("✅ Model uploaded successfully!")
                st.rerun()
        
        # Tab 3: Hugging Face specific
        with tab3:
            st.markdown("### Download from Hugging Face")
            st.code("""
Format: https://huggingface.co/USERNAME/MODEL_NAME/resolve/main/FILENAME.pth
Example: https://huggingface.co/username/emotion-model/resolve/main/emotion_model.pth
            """)
            
            hf_username = st.text_input("Hugging Face Username:", key="hf_user")
            hf_model = st.text_input("Model Name:", key="hf_model")
            hf_file = st.text_input("File Name:", value="emotion_model.pth", key="hf_file")
            
            if hf_username and hf_model and st.button("📥 Download from Hugging Face", key="download_hf"):
                hf_url = f"https://huggingface.co/{hf_username}/{hf_model}/resolve/main/{hf_file}"
                download_model_from_url(hf_url, model_path)
        
        return None
    
    # Load the model
    with st.spinner("Loading emotion detection model..."):
        model = load_model(model_path)
    
    if model is None:
        st.error("Failed to load model. Please check the model file.")
        if st.button("🗑️ Delete corrupted model file"):
            os.remove(model_path)
            st.rerun()
        return None
    
    return model

def download_model_from_url(url, model_path):
    """Download model from URL with progress bar"""
    try:
        import requests
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Download with progress
        with open(model_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = downloaded / total_size
                    progress_bar.progress(progress)
                    status_text.text(f"Downloading: {downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB")
        
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Model downloaded successfully! Please rerun the app.")
        st.rerun()
        
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        st.info("""
        Make sure your URL is correct. It should be a direct download link ending with .pth or .pt
        
        Correct format: https://huggingface.co/USERNAME/MODEL/resolve/main/FILENAME.pth
        """)
