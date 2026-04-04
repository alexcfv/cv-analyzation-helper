from ingestion.loader import ResumeLoader

resume_loader = ResumeLoader()

dir_path = input("enter dir path for the resumes: ")

print(resume_loader.load_folder(dir_path))