import tempfile

def save_temp_file(uploaded_file, suffix=".jpg"):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(uploaded_file.getbuffer())
    tmp_file.flush()
    return tmp_file.name
