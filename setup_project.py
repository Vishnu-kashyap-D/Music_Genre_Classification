import subprocess
import sys

def install_libraries():
    """Installs required Python libraries using pip."""
    try:
        print("Installing required Python libraries...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow", "librosa", "numpy", "matplotlib"])
        print("All libraries installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during library installation: {e}")
        print("Please try running 'pip install tensorflow librosa numpy matplotlib' manually.")

def main():
    install_libraries()
    
    gtzan_dataset_url = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
    print("\nGTZAN dataset information:")
    print(f"  URL: {gtzan_dataset_url}")
    print("  This is the standard dataset for music genre classification.")
    print("  You will need to download and extract this file.")
    print("  The file is a compressed archive (.tar.gz) that contains the audio files.")

if __name__ == "__main__":
    main()
