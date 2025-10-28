import os

folder = 'models/'
total_size = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder))
print(f"Total disk space used by models: {total_size / (1024**2):.2f} MB")
