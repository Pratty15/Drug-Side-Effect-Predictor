import os

# Get path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(BASE_DIR, 'models')

if os.path.exists(folder):
    total_size = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))
    print(f"Total disk space used by models: {total_size / (1024**2):.2f} MB")
else:
    print(f"Models directory not found at: {folder}")
