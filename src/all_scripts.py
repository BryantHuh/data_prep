import os

def list_all_files(base_path):
    all_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            rel_dir = os.path.relpath(root, base_path)
            rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
            all_files.append(rel_file)
    return all_files

if __name__ == "__main__":
    base_directory = "."  # oder z.B. "./src", "./notebooks", etc.
    file_list = list_all_files(base_directory)

    print("Gefundene Dateien:")
    for f in file_list:
        print(f)

    # Optional: In Datei speichern
    with open("file_index.txt", "w", encoding="utf-8") as f:
        for path in file_list:
            f.write(path + "\n")
