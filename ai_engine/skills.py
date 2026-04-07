import re

COMMON_SKILLS = [
    "python", "java", "c++", "c", "sql", "mysql", "postgresql",
    "machine learning", "deep learning", "data science",
    "flask", "django", "fastapi",
    "html", "css", "javascript", "react", "node",
    "aws", "docker", "kubernetes",
    "linux", "git", "tensorflow", "pytorch"
]

def extract_skills(text):
    text = text.lower()
    found_skills = set()

    for skill in COMMON_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found_skills.add(skill.title())

    return sorted(list(found_skills))
