from resume_parser.extract_text import extract_text
from ai_engine.semantic_matcher import match_resume
from ai_engine.skills import extract_skills
from ai_engine.education import extract_education
from ai_engine.experience import extract_experience

from datetime import datetime, date, timedelta
from flask import Flask, render_template, request, redirect, session, flash, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import exists, and_
from sqlalchemy.orm import joinedload
from flask_mail import Mail, Message
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import re
import json

# -------------------- APP CONFIG --------------------

app = Flask(__name__)
app.secret_key = "offline_hr_portal_secret"

# -------------------- EMAIL CONFIGURATION --------------------
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'chinmayjandrapalli68@gmail.com'
app.config['MAIL_PASSWORD'] = 'auxp uykf ywyt dajo'
app.config['MAIL_DEFAULT_SENDER'] = ('HR Team', 'chinmayjandrapalli68@gmail.com')

mail = Mail(app)

# -------------------- DATABASE CONFIG (SQLITE) --------------------

basedir = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, "resume_system.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# -------------------- DATABASE MODELS --------------------

class HRUser(db.Model):
    __tablename__ = "hr_user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)


class CandidateUser(db.Model):
    __tablename__ = "candidate_user"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class JobAnalysis(db.Model):
    __tablename__ = "job_analysis"
    id = db.Column(db.Integer, primary_key=True)
    job_title = db.Column(db.String(200))
    required_skills = db.Column(db.Text)
    min_experience = db.Column(db.Integer)
    deadline = db.Column(db.Date)
    status = db.Column(db.String(20), default="Active")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class JobApplication(db.Model):
    __tablename__ = "job_application"
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey("job_analysis.id"), nullable=False)
    candidate_id = db.Column(db.Integer, db.ForeignKey("candidate_user.id"), nullable=False)
    candidate_name = db.Column(db.String(150), nullable=False)
    resume_file = db.Column(db.String(300), nullable=False)
    applied_at = db.Column(db.DateTime, default=datetime.utcnow)
    withdrawn_at = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(20), default="Active")


class CandidateResult(db.Model):
    __tablename__ = "candidate_result"
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey("job_analysis.id"))
    candidate_id = db.Column(db.Integer, db.ForeignKey("candidate_user.id"), nullable=False)
    candidate_name = db.Column(db.String(200))
    score = db.Column(db.Float)
    skills = db.Column(db.Text)
    education = db.Column(db.Text)
    experience = db.Column(db.Integer)
    resume_file = db.Column(db.String(300))
    status = db.Column(db.String(20), default="Pending")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class InterviewSchedule(db.Model):
    __tablename__ = "interview_schedule"
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey("job_analysis.id"), nullable=False)
    candidate_id = db.Column(db.Integer, db.ForeignKey("candidate_user.id"), nullable=False)
    candidate_name = db.Column(db.String(200), nullable=False)
    candidate_email = db.Column(db.String(150), nullable=False)
    interview_date = db.Column(db.DateTime, nullable=False)
    interview_type = db.Column(db.String(50), default="Video Call")
    interview_link = db.Column(db.String(500), nullable=True)
    location = db.Column(db.String(300), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default="Scheduled")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    sent_at = db.Column(db.DateTime, nullable=True)
    
    job = db.relationship('JobAnalysis', backref='interviews')

# -------------------- ENHANCED SKILL DATABASE --------------------

SKILL_DATABASE = {
    'technical': [
        'python', 'java', 'javascript', 'typescript', 'react', 'node', 'nodejs', 'sql', 'mysql', 'postgresql',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'github', 'gitlab', 'linux', 'bash',
        'html', 'css', 'sass', 'scss', 'bootstrap', 'tailwind', 'jquery', 'ajax', 'json', 'xml',
        'mongodb', 'redis', 'elasticsearch', 'kafka', 'rabbitmq', 'nginx', 'apache',
        'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
        'django', 'flask', 'fastapi', 'spring', 'springboot', 'hibernate', 'express', 'next.js', 'nuxt.js',
        'php', 'laravel', 'symfony', 'ruby', 'rails', 'go', 'golang', 'rust', 'c', 'c++', 'csharp', '.net',
        'swift', 'kotlin', 'flutter', 'react native', 'ionic', 'cordova',
        'ansible', 'terraform', 'jenkins', 'circleci', 'travis', 'github actions', 'gitlab ci',
        'prometheus', 'grafana', 'elk stack', 'splunk', 'new relic', 'datadog',
        'graphql', 'rest api', 'soap', 'grpc', 'websocket', 'oauth', 'jwt', 'sso', 'ldap',
        'cicd', 'devops', 'microservices', 'serverless', 'lambda', 'ecs', 'eks', 'gke'
    ],
    'soft': ['communication', 'leadership', 'teamwork', 'problem solving', 'critical thinking', 'time management', 'adaptability', 'creativity', 'emotional intelligence'],
    'domain': ['machine learning', 'deep learning', 'data science', 'web development', 'cloud computing', 'devops', 'cybersecurity', 'blockchain', 'iot', 'big data', 'nlp', 'computer vision'],
    'alternatives': {
        'react': ['reactjs', 'react.js', 'react native', 'next.js', 'gatsby', 'preact', 'frontend', 'frontend development', 'ui development'],
        'python': ['django', 'flask', 'fastapi', 'python3', 'py', 'data analysis', 'scripting', 'automation'],
        'aws': ['amazon web services', 'ec2', 's3', 'rds', 'lambda', 'cloudformation', 'cloudfront', 'route53', 'azure', 'gcp', 'cloud infrastructure', 'cloud platforms'],
        'docker': ['containerization', 'containers', 'kubernetes', 'k8s', 'container orchestration', 'podman', 'cri-o'],
        'sql': ['mysql', 'postgresql', 'sqlite', 'database', 'rdbms', 'relational database', 'nosql', 'mongodb', 'query optimization', 'database design'],
        'git': ['github', 'gitlab', 'bitbucket', 'version control', 'source control', 'svn', 'mercurial'],
        'javascript': ['js', 'es6', 'es7', 'typescript', 'ts', 'node', 'nodejs', 'frontend', 'backend'],
        'machine learning': ['ml', 'deep learning', 'ai', 'artificial intelligence', 'tensorflow', 'pytorch', 'scikit-learn', 'predictive modeling', 'neural networks'],
        'node': ['nodejs', 'node.js', 'express', 'npm', 'backend javascript'],
        'kubernetes': ['k8s', 'container orchestration', 'docker swarm', 'openshift', 'rancher'],
        'devops': ['cicd', 'continuous integration', 'continuous deployment', 'infrastructure as code', 'iac', 'site reliability', 'sre']
    },
    'learning_paths': {
        'python': {
            'beginner': ['Python for Everybody (Coursera)', 'Automate the Boring Stuff', 'Codecademy Python'],
            'intermediate': ['Django for Beginners', 'Flask Web Development', 'Python Testing'],
            'advanced': ['Advanced Python Patterns', 'System Design with Python', 'ML with Python']
        },
        'react': {
            'beginner': ['React Official Docs', 'Scrimba React Course', 'React for Beginners'],
            'intermediate': ['Advanced React Patterns', 'React Performance', 'Testing React Apps'],
            'advanced': ['React Architecture', 'React Native', 'Next.js Mastery']
        },
        'aws': {
            'beginner': ['AWS Cloud Practitioner', 'AWS Free Tier Labs', 'Cloud Computing Basics'],
            'intermediate': ['AWS Solutions Architect', 'AWS Developer Associate', 'Serverless Architecture'],
            'advanced': ['AWS DevOps Engineer', 'AWS Security Specialty', 'Multi-Cloud Strategies']
        },
        'machine learning': {
            'beginner': ['ML Crash Course (Google)', 'Andrew Ng ML Course', 'Kaggle Learn'],
            'intermediate': ['Deep Learning Specialization', 'ML Engineering', 'MLOps'],
            'advanced': ['Advanced ML (MIT)', 'Research Papers Implementation', 'Production ML Systems']
        }
    },
    'market_demand': {
        'python': 10, 'machine learning': 10, 'aws': 10, 'docker': 9, 'kubernetes': 9,
        'react': 10, 'sql': 9, 'javascript': 9, 'typescript': 9, 'data science': 9, 'devops': 9,
        'java': 8, 'c++': 7, 'golang': 9, 'rust': 8, 'angular': 7, 'vue': 8, 'node': 9,
        'git': 8, 'github': 8, 'ci/cd': 9, 'microservices': 8, 'cloud computing': 9,
        'tensorflow': 9, 'pytorch': 9, 'nlp': 9, 'computer vision': 8, 'blockchain': 6
    },
    'learning_time': {
        'python': {'beginner': 40, 'intermediate': 100, 'expert': 300},
        'react': {'beginner': 60, 'intermediate': 150, 'expert': 400},
        'aws': {'beginner': 80, 'intermediate': 200, 'expert': 500},
        'machine learning': {'beginner': 100, 'intermediate': 300, 'expert': 800},
        'docker': {'beginner': 20, 'intermediate': 60, 'expert': 150},
        'sql': {'beginner': 30, 'intermediate': 80, 'expert': 200},
        'kubernetes': {'beginner': 40, 'intermediate': 120, 'expert': 300},
        'javascript': {'beginner': 40, 'intermediate': 100, 'expert': 250}
    },
    'priority': {
        'critical': ['python', 'machine learning', 'aws', 'sql', 'docker', 'kubernetes', 'react', 'javascript', 'git'],
        'important': ['typescript', 'node', 'java', 'golang', 'ci/cd', 'microservices', 'cloud computing', 'devops'],
        'nice_to_have': ['angular', 'vue', 'php', 'ruby', 'c++', 'rust', 'blockchain', 'iot']
    }
}

# -------------------- JOB EXPIRY FUNCTIONS --------------------

def check_and_close_expired_jobs():
    today = date.today()
    expired_jobs = JobAnalysis.query.filter(
        JobAnalysis.deadline < today,
        JobAnalysis.status == 'Active'
    ).all()
    
    for job in expired_jobs:
        job.status = 'Closed'
    
    if expired_jobs:
        db.session.commit()
        print(f"[AUTO-CLOSE] Closed {len(expired_jobs)} expired job(s)")
    
    return len(expired_jobs)


def get_job_status_badge(job):
    if job.status == 'Closed':
        return 'closed'
    elif job.deadline and job.deadline < date.today():
        return 'expired'
    elif job.deadline:
        days_left = (job.deadline - date.today()).days
        if days_left <= 3:
            return 'urgent'
        elif days_left <= 7:
            return 'warning'
    return 'active'


def get_days_remaining(job):
    if not job.deadline:
        return None
    if job.status == 'Closed':
        return -1
    days = (job.deadline - date.today()).days
    return days if days >= 0 else -1

# -------------------- EMAIL HELPER FUNCTIONS --------------------

def send_email(to_email, subject, body, html_body=None):
    try:
        msg = Message(
            subject=subject,
            recipients=[to_email],
            body=body,
            html=html_body
        )
        mail.send(msg)
        print(f"[EMAIL SENT] To: {to_email}, Subject: {subject}")
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send email to {to_email}: {str(e)}")
        return False


def send_application_status_email(candidate_email, candidate_name, job_title, status, interview_details=None):
    if status == "Accepted":
        subject = f"Congratulations! You've been selected for {job_title}"
        body = f"""
Dear {candidate_name},

Great news! We are pleased to inform you that your application for the position of "{job_title}" has been ACCEPTED.

We were impressed with your qualifications and experience, and we believe you would be a valuable addition to our team.

"""
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #22c55e, #16a34a); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
        .content {{ background: #f9fafb; padding: 30px; border-radius: 0 0 10px 10px; }}
        .success-badge {{ background: #dcfce7; color: #166534; padding: 10px 20px; border-radius: 20px; display: inline-block; font-weight: bold; margin: 10px 0; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎉 Congratulations!</h1>
        </div>
        <div class="content">
            <p>Dear <strong>{candidate_name}</strong>,</p>
            
            <p>Great news! We are pleased to inform you that your application has been:</p>
            
            <div class="success-badge">✓ ACCEPTED</div>
            
            <p><strong>Position:</strong> {job_title}</p>
            
            <p>We were impressed with your qualifications and experience, and we believe you would be a valuable addition to our team.</p>
            
            <p><strong>Next Steps:</strong><br>
            Our HR team will contact you shortly regarding the next steps in the hiring process.</p>
            
            <div class="footer">
                <p>Best regards,<br>
                <strong>HR Team</strong><br>
                Resume Intelligence System</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        if interview_details:
            body += f"\nInterview Scheduled:\nDate: {interview_details['date']}\nType: {interview_details['type']}"
            if interview_details.get('link'):
                body += f"\nLink: {interview_details['link']}"
            if interview_details.get('location'):
                body += f"\nLocation: {interview_details['location']}"
    
    else:
        subject = f"Update on your application for {job_title}"
        body = f"""
Dear {candidate_name},

Thank you for your interest in the "{job_title}" position and for taking the time to apply.

After careful consideration of all applications, we regret to inform you that we have decided to move forward with other candidates whose qualifications more closely match our current needs.

This decision was not easy, as we received many strong applications. We encourage you to apply for future openings that match your skills and experience.

We wish you all the best in your job search and future career endeavors.

Best regards,
HR Team
Resume Intelligence System
"""
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #64748b, #475569); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
        .content {{ background: #f9fafb; padding: 30px; border-radius: 0 0 10px 10px; }}
        .status-badge {{ background: #fee2e2; color: #991b1b; padding: 10px 20px; border-radius: 20px; display: inline-block; font-weight: bold; margin: 10px 0; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Application Update</h1>
        </div>
        <div class="content">
            <p>Dear <strong>{candidate_name}</strong>,</p>
            
            <p>Thank you for your interest in the position and for taking the time to apply.</p>
            
            <p>After careful consideration, we regret to inform you that your application has been:</p>
            
            <div class="status-badge">Not Selected</div>
            
            <p><strong>Position:</strong> {job_title}</p>
            
            <p>This decision was not easy, as we received many strong applications. We encourage you to apply for future openings that match your skills and experience.</p>
            
            <p>We wish you all the best in your job search and future career endeavors.</p>
            
            <div class="footer">
                <p>Best regards,<br>
                <strong>HR Team</strong><br>
                Resume Intelligence System</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    return send_email(candidate_email, subject, body, html_body)


def send_interview_email(candidate_email, candidate_name, job_title, interview_details, is_update=False):
    action = "Updated" if is_update else "Scheduled"
    subject = f"Interview {action}: {job_title} Position"
    
    date_str = interview_details['date'].strftime('%A, %B %d, %Y at %I:%M %p')
    
    body = f"""
Dear {candidate_name},

Your interview for the "{job_title}" position has been {action.lower()}.

Interview Details:
Date & Time: {date_str}
Type: {interview_details['type']}
"""
    
    if interview_details.get('location'):
        body += f"Location: {interview_details['location']}\n"
    if interview_details.get('link'):
        body += f"Meeting Link: {interview_details['link']}\n"
    
    if interview_details.get('notes'):
        body += f"\nAdditional Notes: {interview_details['notes']}\n"
    
    body += """
Please confirm your attendance by replying to this email.

If you need to reschedule, please contact us at least 24 hours in advance.

Best regards,
HR Team
"""
    
    html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
        .content {{ background: #f9fafb; padding: 30px; border-radius: 0 0 10px 10px; }}
        .detail-box {{ background: white; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #3b82f6; }}
        .detail-row {{ margin: 10px 0; }}
        .detail-label {{ font-weight: 600; color: #6b7280; display: inline-block; width: 120px; }}
        .detail-value {{ color: #1f293b; }}
        .btn {{ background: #3b82f6; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block; margin: 10px 0; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📅 Interview {action}</h1>
        </div>
        <div class="content">
            <p>Dear <strong>{candidate_name}</strong>,</p>
            
            <p>Your interview for the <strong>{job_title}</strong> position has been <strong>{action.lower()}</strong>.</p>
            
            <div class="detail-box">
                <div class="detail-row">
                    <span class="detail-label">Date & Time:</span>
                    <span class="detail-value">{date_str}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Type:</span>
                    <span class="detail-value">{interview_details['type']}</span>
                </div>
                {'<div class="detail-row"><span class="detail-label">Location:</span><span class="detail-value">' + interview_details['location'] + '</span></div>' if interview_details.get('location') else ''}
                {'<div class="detail-row"><span class="detail-label">Meeting Link:</span><span class="detail-value"><a href="' + interview_details['link'] + '">' + interview_details['link'] + '</a></span></div>' if interview_details.get('link') else ''}
                {'<div class="detail-row"><span class="detail-label">Notes:</span><span class="detail-value">' + interview_details['notes'] + '</span></div>' if interview_details.get('notes') else ''}
            </div>
            
            <p>Please confirm your attendance by replying to this email.</p>
            
            <p><em>If you need to reschedule, please contact us at least 24 hours in advance.</em></p>
            
            <div class="footer">
                <p>Best regards,<br>
                <strong>HR Team</strong><br>
                Resume Intelligence System</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    return send_email(candidate_email, subject, body, html_body)

# -------------------- IMPROVED AI SCORING FUNCTIONS --------------------

def preprocess_text(text):
    """Clean and normalize text for better matching"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text


def extract_section(text, section_name):
    """Extract specific section from resume with improved accuracy"""
    lines = text.split('\n')
    section_lines = []
    capture = False
    
    section_patterns = {
        'experience': ['experience', 'work experience', 'employment', 'work history', 'professional experience', 'career history'],
        'education': ['education', 'academic background', 'qualifications', 'academic qualifications', 'degrees'],
        'skills': ['skills', 'technical skills', 'core competencies', 'expertise', 'proficiencies', 'technologies'],
        'projects': ['projects', 'personal projects', 'professional projects', 'key projects'],
        'certifications': ['certifications', 'certificates', 'professional certifications', 'accreditations']
    }
    
    patterns = section_patterns.get(section_name, [section_name])
    
    for line in lines:
        clean = line.strip().lower()
        
        if not capture:
            for pattern in patterns:
                if clean.startswith(pattern) or pattern in clean[:len(pattern) + 5]:
                    capture = True
                    break
            continue
        
        for other_section, other_patterns in section_patterns.items():
            if other_section != section_name:
                for other_pattern in other_patterns:
                    if clean.startswith(other_pattern) or other_pattern in clean[:len(other_pattern) + 5]:
                        return ' '.join(section_lines)
        
        if capture and line.strip():
            section_lines.append(line.strip())
    
    return ' '.join(section_lines)


def calculate_tfidf_similarity(resume_text, job_text):
    """Calculate TF-IDF based cosine similarity for keyword matching"""
    try:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        texts = [preprocess_text(resume_text), preprocess_text(job_text)]
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity * 100)
    except Exception as e:
        print(f"TF-IDF calculation error: {e}")
        return 0.0


def extract_skills_precise(resume_text, job_skills):
    """Extract skills with precise matching and context analysis"""
    resume_lower = resume_text.lower()
    job_skills_lower = [s.lower().strip() for s in job_skills]
    
    synonyms = SKILL_DATABASE['alternatives']
    
    matched_skills = {}
    total_confidence = 0
    skill_details = []
    
    for skill in job_skills_lower:
        skill_info = {
            'skill': skill,
            'matched': False,
            'confidence': 0,
            'match_type': 'none',
            'proficiency': 'beginner',
            'context': ''
        }
        
        if re.search(r"\b" + re.escape(skill) + r"\b", resume_lower):
            skill_info['matched'] = True
            skill_info['confidence'] = 1.0
            skill_info['match_type'] = 'exact'
        
        elif skill in resume_lower:
            skill_info['matched'] = True
            skill_info['confidence'] = 0.7
            skill_info['match_type'] = 'partial'
        
        if not skill_info['matched'] and skill in synonyms:
            for syn in synonyms[skill]:
                if re.search(r"\b" + re.escape(syn) + r"\b", resume_lower):
                    skill_info['matched'] = True
                    skill_info['confidence'] = 0.85
                    skill_info['match_type'] = f'synonym ({syn})'
                    break
                elif syn in resume_lower:
                    skill_info['matched'] = True
                    skill_info['confidence'] = 0.6
                    skill_info['match_type'] = f'partial_synonym ({syn})'
                    break
        
        if skill_info['matched']:
            skill_info['proficiency'] = determine_proficiency(resume_text, skill)
            proficiency_multiplier = {
                'expert': 1.0,
                'advanced': 0.9,
                'intermediate': 0.8,
                'beginner': 0.6
            }
            skill_info['confidence'] *= proficiency_multiplier.get(skill_info['proficiency'], 0.7)
        
        matched_skills[skill] = skill_info['confidence']
        total_confidence += skill_info['confidence']
        skill_details.append(skill_info)
    
    if job_skills:
        skill_score = (total_confidence / len(job_skills)) * 100
    else:
        skill_score = 100
    
    return round(skill_score, 2), matched_skills, skill_details


def determine_proficiency(resume_text, skill):
    """Determine skill proficiency level from context"""
    skill_pattern = r"\b" + re.escape(skill) + r"\b"
    matches = list(re.finditer(skill_pattern, resume_text.lower()))
    
    if not matches:
        return 'beginner'
    
    contexts = []
    for match in matches:
        start = max(0, match.start() - 100)
        end = min(len(resume_text), match.end() + 100)
        contexts.append(resume_text[start:end].lower())
    
    expert_indicators = ['expert', 'advanced', 'senior', 'lead', 'architect', 'specialist', '5+ years', '6+ years', '7+ years', '8+ years', '9+ years', '10+ years']
    advanced_indicators = ['advanced', 'experienced', 'proficient', 'skilled', '4 years', '5 years', 'strong']
    intermediate_indicators = ['intermediate', '3 years', '2+ years', 'familiar', 'working knowledge']
    
    all_context = ' '.join(contexts)
    
    years_pattern = r'(\d+)\+?\s*years?.*?' + re.escape(skill)
    years_match = re.search(years_pattern, resume_text.lower())
    
    if years_match:
        years = int(years_match.group(1))
        if years >= 5:
            return 'expert'
        elif years >= 3:
            return 'advanced'
        elif years >= 1:
            return 'intermediate'
    
    for indicator in expert_indicators:
        if indicator in all_context:
            return 'expert'
    
    for indicator in advanced_indicators:
        if indicator in all_context:
            return 'advanced'
    
    for indicator in intermediate_indicators:
        if indicator in all_context:
            return 'intermediate'
    
    return 'beginner'


def calculate_experience_score_improved(resume_text, min_years, job_title):
    """Calculate experience score with job relevance weighting"""
    exp_section = extract_section(resume_text, 'experience')
    
    years_matches = re.findall(r"(\d+)\s*\+?\s*years?", exp_section.lower())
    total_years = max([int(y) for y in years_matches]) if years_matches else 0
    
    if min_years > 0:
        base_score = min((total_years / min_years) * 100, 150)
    else:
        base_score = 100 if total_years > 0 else 50
    
    job_keywords = set(job_title.lower().split())
    relevance_score = 0
    
    for keyword in job_keywords:
        if len(keyword) > 3:
            count = exp_section.lower().count(keyword)
            relevance_score += min(count * 15, 30)
    
    seniority_indicators = ['senior', 'lead', 'principal', 'staff', 'manager', 'director', 'head of', 'architect']
    seniority_score = 0
    for indicator in seniority_indicators:
        if indicator in exp_section.lower():
            seniority_score = 20
            break
    
    final_score = min(base_score + relevance_score + seniority_score, 150)
    return round(final_score, 2), total_years


def calculate_education_score_improved(edu_level, min_exp):
    """Calculate education score based on experience requirements"""
    expected_level = 2
    
    if min_exp >= 7:
        expected_level = 4
    elif min_exp >= 5:
        expected_level = 3
    elif min_exp >= 3:
        expected_level = 3
    
    if edu_level >= expected_level:
        return 100
    elif edu_level == expected_level - 1:
        return 85
    elif edu_level == expected_level - 2:
        return 70
    else:
        return max(edu_level * 20, 40)


def extract_education_level_improved(text):
    """Extract education level with improved accuracy"""
    education_section = extract_section(text, 'education')
    text_lower = education_section.lower() if education_section else text.lower()
    
    education_patterns = {
        5: ['phd', 'doctorate', 'ph.d', 'doctor of philosophy', 'd.phil', 'ed.d'],
        4: ['masters', 'master', 'mba', 'm.s', 'm.sc', 'm.tech', 'm.e', 'm.eng', 'ma', 'm.a', 'post graduate', 'postgraduate', 'm.b.a'],
        3: ['bachelors', 'bachelor', 'b.tech', 'b.e', 'b.eng', 'b.s', 'b.sc', 'b.a', 'undergraduate', 'graduate', 'ba', 'bs', 'be', 'btech'],
        2: ['diploma', 'associate', 'a.s', 'a.a', 'certification', 'certificate', 'vocational'],
        1: ['high school', 'secondary', 'hsc', 'ssc', '12th', '10+2']
    }
    
    max_level = 1
    
    for level, patterns in education_patterns.items():
        for pattern in patterns:
            if re.search(r"\b" + re.escape(pattern) + r"\b", text_lower):
                max_level = max(max_level, level)
                break
    
    gpa_pattern = r'(?:gpa|cgpa)[\s:]*(\d+\.?\d*)'
    gpa_match = re.search(gpa_pattern, text_lower)
    if gpa_match:
        gpa = float(gpa_match.group(1))
        if gpa > 10:
            gpa = gpa / 10
        if gpa >= 3.5 or (gpa >= 7 and gpa <= 10):
            max_level += 0.5
    
    return min(max_level, 5)


def calculate_final_score_improved(resume_text, job_text, job_skills, min_experience, job_title):
    """Calculate final score with improved precision using hybrid approach"""
    
    semantic_raw = match_resume(job_text, resume_text)
    semantic_score = min(semantic_raw * 1.1, 100)
    
    tfidf_score = calculate_tfidf_similarity(resume_text, job_text)
    
    skills_score, matched_skills, skill_details = extract_skills_precise(resume_text, job_skills)
    
    exp_score, total_years = calculate_experience_score_improved(resume_text, min_experience, job_title)
    
    edu_level = extract_education_level_improved(resume_text)
    edu_score = calculate_education_score_improved(edu_level, min_experience)
    
    final_score = (
        (semantic_score * 0.30) +
        (tfidf_score * 0.15) +
        (skills_score * 0.30) +
        (min(exp_score, 100) * 0.20) +
        (edu_score * 0.05)
    )
    
    final_score = max(0, min(final_score, 100))
    
    breakdown = {
        'semantic': round(semantic_score, 1),
        'tfidf': round(tfidf_score, 1),
        'skills': round(skills_score, 1),
        'experience': round(min(exp_score, 100), 1),
        'education': round(edu_score, 1),
        'years': total_years,
        'education_level': edu_level,
        'matched_skills': matched_skills,
        'skill_details': skill_details
    }
    
    return round(final_score, 2), breakdown

# -------------------- ENHANCED SKILL GAP ANALYSIS FUNCTIONS --------------------

def analyze_skill_gap(resume_text, job_skills):
    resume_lower = resume_text.lower()
    job_skills_lower = [s.lower().strip() for s in job_skills]
    
    synonyms = SKILL_DATABASE['alternatives']
    
    matched_skills = []
    missing_skills = []
    partial_matches = []
    skill_gaps_detailed = []
    
    for skill in job_skills_lower:
        skill_data = {
            'name': skill.title(),
            'level': 'beginner',
            'confidence': 0,
            'priority': classify_skill_priority(skill),
            'market_demand': SKILL_DATABASE['market_demand'].get(skill, 5),
            'alternatives': SKILL_DATABASE['alternatives'].get(skill, []),
            'learning_time': SKILL_DATABASE['learning_time'].get(skill, {'beginner': 50, 'intermediate': 100, 'expert': 250}),
            'learning_path': SKILL_DATABASE['learning_paths'].get(skill, {}),
            'category': classify_skill_category(skill)
        }
        
        exact_pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(exact_pattern, resume_lower):
            skill_data['confidence'] = 100
            skill_data['level'] = determine_proficiency(resume_lower, skill)
            skill_data['proficiency_score'] = calculate_proficiency_score(resume_text, skill)
            matched_skills.append(skill_data)
            continue
            
        if skill in resume_lower:
            skill_data['confidence'] = 70
            skill_data['level'] = determine_proficiency(resume_lower, skill)
            skill_data['proficiency_score'] = calculate_proficiency_score(resume_text, skill) * 0.7
            matched_skills.append(skill_data)
            continue
            
        found_synonym = False
        if skill in synonyms:
            for syn in synonyms[skill]:
                syn_pattern = r"\b" + re.escape(syn) + r"\b"
                if re.search(syn_pattern, resume_lower):
                    skill_data['confidence'] = 80
                    skill_data['level'] = determine_proficiency(resume_lower, syn)
                    skill_data['name'] = f"{skill.title()} (via {syn})"
                    skill_data['proficiency_score'] = calculate_proficiency_score(resume_text, syn) * 0.8
                    matched_skills.append(skill_data)
                    found_synonym = True
                    break
                elif syn in resume_lower:
                    skill_data['confidence'] = 50
                    skill_data['level'] = 'beginner'
                    skill_data['name'] = f"{skill.title()} (via {syn})"
                    skill_data['proficiency_score'] = 30
                    partial_matches.append(skill_data)
                    found_synonym = True
                    break
        
        if not found_synonym:
            skill_data['gap_impact'] = calculate_gap_impact(skill_data)
            missing_skills.append(skill.title())
            skill_gaps_detailed.append(skill_data)
    
    total_required = len(job_skills_lower)
    matched_count = len([s for s in matched_skills if s['confidence'] >= 70])
    match_percentage = round((matched_count / total_required) * 100) if total_required > 0 else 0
    
    critical_gaps = [s for s in skill_gaps_detailed if s['priority'] == 'critical']
    total_learning_time = estimate_total_learning_time(skill_gaps_detailed)
    roi_score = calculate_roi_score(skill_gaps_detailed)
    
    return {
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'partial_matches': partial_matches,
        'skill_gaps_detailed': skill_gaps_detailed,
        'critical_gaps': critical_gaps,
        'match_percentage': match_percentage,
        'total_required': total_required,
        'total_matched': len(matched_skills),
        'total_learning_time': total_learning_time,
        'roi_score': roi_score,
        'priority_breakdown': {
            'critical': len([s for s in skill_gaps_detailed if s['priority'] == 'critical']),
            'important': len([s for s in skill_gaps_detailed if s['priority'] == 'important']),
            'nice_to_have': len([s for s in skill_gaps_detailed if s['priority'] == 'nice_to_have'])
        }
    }


def classify_skill_priority(skill):
    skill_lower = skill.lower()
    if skill_lower in SKILL_DATABASE['priority']['critical']:
        return 'critical'
    elif skill_lower in SKILL_DATABASE['priority']['important']:
        return 'important'
    else:
        return 'nice_to_have'


def classify_skill_category(skill):
    skill_lower = skill.lower()
    for category, skills in SKILL_DATABASE.items():
        if category in ['technical', 'soft', 'domain'] and skill_lower in skills:
            return category
    return 'technical'


def calculate_proficiency_score(resume_text, skill):
    skill_lower = skill.lower()
    resume_lower = resume_text.lower()
    
    mentions = resume_lower.count(skill_lower)
    
    expertise_score = 0
    context_window = 200
    
    for match in re.finditer(r"\b" + re.escape(skill_lower) + r"\b", resume_lower):
        start = max(0, match.start() - context_window)
        end = min(len(resume_lower), match.end() + context_window)
        context = resume_lower[start:end]
        
        if any(word in context for word in ['expert', 'advanced', 'senior', 'lead', 'architect']):
            expertise_score += 25
        elif any(word in context for word in ['intermediate', 'experienced', 'proficient']):
            expertise_score += 15
        elif any(word in context for word in ['beginner', 'basic', 'familiar']):
            expertise_score += 5
    
    years_match = re.search(r'(\d+)\s*\+?\s*years?.*?' + re.escape(skill_lower), resume_lower)
    if years_match:
        years = int(years_match.group(1))
        expertise_score += min(years * 5, 30)
    
    base_score = min(mentions * 10, 40)
    return min(base_score + expertise_score, 100)


def calculate_gap_impact(skill_data):
    priority_multiplier = {
        'critical': 3,
        'important': 2,
        'nice_to_have': 1
    }
    
    demand_score = skill_data['market_demand']
    priority_score = priority_multiplier.get(skill_data['priority'], 1)
    
    return demand_score * priority_score


def estimate_total_learning_time(skill_gaps):
    total_hours = 0
    for gap in skill_gaps:
        level = gap.get('target_level', 'intermediate')
        time_needed = gap['learning_time'].get(level, 100)
        total_hours += time_needed
    
    total_days = total_hours / 4
    return {
        'hours': total_hours,
        'days': round(total_days),
        'weeks': round(total_days / 5, 1)
    }


def calculate_roi_score(skill_gaps):
    if not skill_gaps:
        return 0
    
    total_impact = sum(calculate_gap_impact(gap) for gap in skill_gaps)
    total_time = sum(gap['learning_time'].get('intermediate', 100) for gap in skill_gaps)
    
    roi = total_impact / (total_time / 10)
    return round(min(roi, 100), 1)


def determine_skill_level(resume_text, skill):
    exp_indicators = {
        'expert': ['expert', 'advanced', '5+ years', '5 years', '6 years', '7 years', '8 years', '9 years', '10 years', 'senior', 'lead', 'architect'],
        'intermediate': ['intermediate', '3 years', '4 years', '2+ years', 'experienced', 'proficient'],
        'beginner': ['beginner', 'basic', 'familiar', '1 year', '2 years', 'entry', 'junior', 'learning']
    }
    
    skill_pos = resume_text.find(skill)
    if skill_pos == -1:
        return 'beginner'
    
    start = max(0, skill_pos - 100)
    end = min(len(resume_text), skill_pos + 100)
    context = resume_text[start:end]
    
    for level, indicators in exp_indicators.items():
        for indicator in indicators:
            if indicator in context:
                return level
    
    years_match = re.search(r'(\d+)\s*\+?\s*years?.*?' + re.escape(skill), resume_text)
    if years_match:
        years = int(years_match.group(1))
        if years >= 5:
            return 'expert'
        elif years >= 3:
            return 'intermediate'
    
    return 'beginner'

# -------------------- ROUTES --------------------

@app.route("/")
def index():
    check_and_close_expired_jobs()
    return render_template("index.html")


@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        user = HRUser.query.filter_by(username=request.form["username"]).first()
        if user and check_password_hash(user.password_hash, request.form["password"]):
            session["hr_logged_in"] = True
            return redirect("/hr-dashboard")
        flash("Invalid username or password")
    return render_template("login.html")


@app.route("/hr-dashboard")
def hr_dashboard():
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")

    check_and_close_expired_jobs()
    
    jobs = JobAnalysis.query.order_by(JobAnalysis.created_at.desc()).all()
    
    jobs_with_meta = []
    for job in jobs:
        days_remaining = get_days_remaining(job)
        jobs_with_meta.append({
            'job': job,
            'days_remaining': days_remaining if days_remaining is not None else -999,
            'has_deadline': job.deadline is not None,
            'status_badge': get_job_status_badge(job)
        })
    
    upcoming_interviews = InterviewSchedule.query.filter(
        InterviewSchedule.status == 'Scheduled',
        InterviewSchedule.interview_date > datetime.now()
    ).count()
    
    return render_template("hr_dashboard.html", 
                         jobs_with_meta=jobs_with_meta,
                         upcoming_interviews=upcoming_interviews)


@app.route("/create-job", methods=["GET", "POST"])
def create_job():
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")

    if request.method == "POST":
        deadline_str = request.form.get("deadline")
        deadline = datetime.strptime(deadline_str, "%Y-%m-%d").date() if deadline_str else None
        
        db.session.add(JobAnalysis(
            job_title=request.form["job_title"],
            required_skills=request.form["required_skills"],
            min_experience=int(request.form["min_experience"]),
            deadline=deadline,
            status="Active"
        ))
        db.session.commit()
        flash("Job created successfully")
        return redirect("/hr-dashboard")

    return render_template("create_job.html", today=date.today().isoformat())


@app.route("/hr/close-job/<int:job_id>")
def close_job_manually(job_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    job = JobAnalysis.query.get_or_404(job_id)
    job.status = 'Closed'
    db.session.commit()
    
    flash(f"Job '{job.job_title}' has been closed")
    return redirect("/hr-dashboard")


@app.route("/hr/reopen-job/<int:job_id>")
def reopen_job(job_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    job = JobAnalysis.query.get_or_404(job_id)
    
    new_deadline = date.today() + timedelta(days=7)
    job.deadline = new_deadline
    job.status = 'Active'
    db.session.commit()
    
    flash(f"Job '{job.job_title}' reopened until {new_deadline}")
    return redirect("/hr-dashboard")


@app.route("/hr/delete-job/<int:job_id>")
def delete_job(job_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    job = JobAnalysis.query.get_or_404(job_id)
    
    JobApplication.query.filter_by(job_id=job_id).delete()
    CandidateResult.query.filter_by(job_id=job_id).delete()
    
    db.session.delete(job)
    db.session.commit()
    
    flash(f"Job '{job.job_title}' has been deleted")
    return redirect("/hr-dashboard")


@app.route("/candidate-login", methods=["GET", "POST"])
def candidate_login():
    if request.method == "POST":
        candidate = CandidateUser.query.filter_by(username=request.form["username"]).first()
        if candidate and check_password_hash(candidate.password_hash, request.form["password"]):
            session["candidate_logged_in"] = True
            session["candidate_id"] = candidate.id
            session["candidate_name"] = candidate.name
            return redirect("/candidate-dashboard")
        flash("Invalid username or password")
    return render_template("candidate_login.html")


@app.route("/candidate-register", methods=["GET", "POST"])
def candidate_register():
    if request.method == "POST":
        if request.form["password"] != request.form["confirm_password"]:
            flash("Passwords do not match")
            return redirect("/candidate-register")

        try:
            db.session.add(CandidateUser(
                name=request.form["name"],
                phone=request.form["phone"],
                email=request.form["email"],
                username=request.form["username"],
                password_hash=generate_password_hash(request.form["password"])
            ))
            db.session.commit()
            flash("Account created successfully")
            return redirect("/candidate-login")
        except Exception as e:
            db.session.rollback()
            flash("Username or email already exists")
            return redirect("/candidate-register")

    return render_template("candidate_register.html")


@app.route("/candidate-dashboard")
def candidate_dashboard():
    if not session.get("candidate_logged_in"):
        return redirect("/candidate-login")

    check_and_close_expired_jobs()
    
    candidate_id = session.get("candidate_id")
    
    applications = db.session.query(
        JobAnalysis.job_title,
        CandidateResult.score,
        CandidateResult.status,
        JobApplication.id,
        JobApplication.withdrawn_at,
        JobApplication.status.label('app_status')
    ).join(
        JobApplication, JobApplication.job_id == JobAnalysis.id
    ).outerjoin(
        CandidateResult, 
        (CandidateResult.job_id == JobAnalysis.id) & 
        (CandidateResult.candidate_id == candidate_id)
    ).filter(
        JobApplication.candidate_id == candidate_id
    ).order_by(JobApplication.applied_at.desc()).all()

    applied_job_ids = db.session.query(JobApplication.job_id).filter(
        JobApplication.candidate_id == candidate_id,
        JobApplication.status != 'Withdrawn'
    ).all()
    applied_job_ids = [j[0] for j in applied_job_ids]

    jobs = JobAnalysis.query.filter(
        JobAnalysis.status == 'Active',
        (JobAnalysis.deadline >= date.today()) | (JobAnalysis.deadline == None)
    ).order_by(JobAnalysis.created_at.desc()).all()

    jobs_with_status = []
    for job in jobs:
        job.already_applied = job.id in applied_job_ids
        jobs_with_status.append(job)

    return render_template(
        "candidate_dashboard.html",
        jobs=jobs_with_status,
        applications=applications,
        today=date.today()
    )


@app.route("/withdraw-application/<int:application_id>", methods=["POST"])
def withdraw_application(application_id):
    if not session.get("candidate_logged_in"):
        return redirect("/candidate-login")
    
    application = JobApplication.query.get_or_404(application_id)
    
    if application.candidate_id != session.get("candidate_id"):
        flash("Unauthorized action")
        return redirect("/candidate-dashboard")
    
    existing_result = CandidateResult.query.filter_by(
        job_id=application.job_id,
        candidate_id=application.candidate_id
    ).first()
    
    if existing_result and existing_result.status in ['Accepted', 'Rejected']:
        flash("Cannot withdraw - decision already made")
        return redirect("/candidate-dashboard")
    
    application.withdrawn_at = datetime.utcnow()
    application.status = "Withdrawn"
    
    if existing_result and existing_result.status == 'Pending':
        db.session.delete(existing_result)
    
    db.session.commit()
    
    flash("Application withdrawn successfully")
    return redirect("/candidate-dashboard")


@app.route("/apply/<int:job_id>", methods=["GET", "POST"])
def apply_job(job_id):
    if not session.get("candidate_logged_in"):
        return redirect("/candidate-login")

    job = JobAnalysis.query.get_or_404(job_id)
    
    if job.status == 'Closed':
        flash("This job posting has been closed")
        return redirect("/candidate-dashboard")
    
    if job.deadline and job.deadline < date.today():
        flash("The deadline for this job has passed")
        return redirect("/candidate-dashboard")

    candidate = CandidateUser.query.get(session["candidate_id"])

    if request.method == "POST":
        file = request.files.get("resume")
        if not file or file.filename == "":
            flash("Please upload resume")
            return redirect(request.url)

        os.makedirs("uploads", exist_ok=True)
        file.save(os.path.join("uploads", file.filename))

        db.session.add(JobApplication(
            job_id=job.id,
            candidate_id=candidate.id,
            candidate_name=candidate.name,
            resume_file=file.filename
        ))
        db.session.commit()

        flash("Application submitted successfully!")
        return redirect("/candidate-dashboard")

    return render_template("apply_job.html", job=job, candidate=candidate)


@app.route("/hr/applications/<int:job_id>")
def hr_view_applications(job_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")

    job = JobAnalysis.query.get_or_404(job_id)
    applications = JobApplication.query.filter_by(job_id=job_id, status='Active').all()
    return render_template("hr_applications.html", job=job, applications=applications)


@app.route("/hr/decision/<int:result_id>/<string:decision>")
def hr_decision(result_id, decision):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")

    result = CandidateResult.query.get_or_404(result_id)
    job = JobAnalysis.query.get_or_404(result.job_id)
    
    candidate = CandidateUser.query.get(result.candidate_id)
    if not candidate:
        flash("Candidate not found")
        return redirect("/hr-dashboard")

    if decision.lower() == "accept":
        result.status = "Accepted"
        flash(f"{result.candidate_name} has been accepted")
        
        email_sent = send_application_status_email(
            candidate_email=candidate.email,
            candidate_name=result.candidate_name,
            job_title=job.job_title,
            status="Accepted"
        )
        
        if email_sent:
            flash(f"Acceptance email sent to {candidate.email}")
        else:
            flash("Failed to send email notification", "warning")
            
    elif decision.lower() == "reject":
        result.status = "Rejected"
        flash(f"{result.candidate_name} has been rejected")
        
        email_sent = send_application_status_email(
            candidate_email=candidate.email,
            candidate_name=result.candidate_name,
            job_title=job.job_title,
            status="Rejected"
        )
        
        if email_sent:
            flash(f"Rejection email sent to {candidate.email}")
        else:
            flash("Failed to send email notification", "warning")
    else:
        flash("Invalid decision")
        return redirect("/hr-dashboard")

    db.session.commit()
    return redirect(f"/hr/sort/{result.job_id}")


@app.route("/hr/bulk-reject/<int:job_id>", methods=["POST"])
def bulk_reject_remaining(job_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    selected_ids = request.form.getlist('candidate_ids')
    
    if selected_ids:
        candidates = CandidateResult.query.filter(
            CandidateResult.id.in_(selected_ids),
            CandidateResult.status == 'Pending'
        ).all()
    else:
        candidates = CandidateResult.query.filter_by(
            job_id=job_id,
            status='Pending'
        ).all()
    
    rejected_count = 0
    for candidate in candidates:
        candidate.status = 'Rejected'
        
        candidate_user = CandidateUser.query.get(candidate.candidate_id)
        if candidate_user:
            job = JobAnalysis.query.get(job_id)
            send_application_status_email(
                candidate_email=candidate_user.email,
                candidate_name=candidate.candidate_name,
                job_title=job.job_title,
                status="Rejected"
            )
        rejected_count += 1
    
    db.session.commit()
    
    if rejected_count > 0:
        flash(f"Successfully rejected {rejected_count} candidate(s) and sent emails")
    else:
        flash("No pending candidates to reject")
    
    return redirect(f"/hr/sort/{job_id}")


@app.route("/hr/skill-gap/<int:result_id>")
def skill_gap_analysis(result_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    result = CandidateResult.query.get_or_404(result_id)
    job = JobAnalysis.query.get_or_404(result.job_id)
    
    job_skills = [s.strip() for s in job.required_skills.split(",") if s.strip()]
    
    resume_path = os.path.join("uploads", result.resume_file)
    resume_text = extract_text(resume_path) if os.path.exists(resume_path) else ""
    
    analysis = analyze_skill_gap(resume_text, job_skills)
    
    return render_template("skill_gap.html",
                         candidate_name=result.candidate_name,
                         job_title=job.job_title,
                         job_id=job.id,
                         experience=result.experience or 0,
                         matched_skills=analysis['matched_skills'],
                         missing_skills=analysis['missing_skills'],
                         partial_matches=analysis['partial_matches'],
                         skill_gaps_detailed=analysis['skill_gaps_detailed'],
                         critical_gaps=analysis['critical_gaps'],
                         match_percentage=analysis['match_percentage'],
                         total_learning_time=analysis['total_learning_time'],
                         roi_score=analysis['roi_score'],
                         priority_breakdown=analysis['priority_breakdown'])


@app.route("/hr/schedule-interview/<int:result_id>", methods=["GET", "POST"])
def schedule_interview(result_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    result = CandidateResult.query.get_or_404(result_id)
    job = JobAnalysis.query.get_or_404(result.job_id)
    candidate = CandidateUser.query.get_or_404(result.candidate_id)
    
    if result.status != "Accepted":
        flash("Can only schedule interviews for accepted candidates")
        return redirect(f"/hr/sort/{result.job_id}")
    
    if request.method == "POST":
        interview_date_str = request.form.get("interview_date")
        interview_type = request.form.get("interview_type", "Video Call")
        interview_link = request.form.get("interview_link", "")
        location = request.form.get("location", "")
        notes = request.form.get("notes", "")
        
        try:
            interview_date = datetime.strptime(interview_date_str, "%Y-%m-%dT%H:%M")
            
            existing = InterviewSchedule.query.filter_by(
                job_id=job.id,
                candidate_id=candidate.id
            ).first()
            
            if existing:
                existing.interview_date = interview_date
                existing.interview_type = interview_type
                existing.interview_link = interview_link
                existing.location = location
                existing.notes = notes
                existing.status = "Scheduled"
                is_update = True
            else:
                new_interview = InterviewSchedule(
                    job_id=job.id,
                    candidate_id=candidate.id,
                    candidate_name=candidate.name,
                    candidate_email=candidate.email,
                    interview_date=interview_date,
                    interview_type=interview_type,
                    interview_link=interview_link,
                    location=location,
                    notes=notes
                )
                db.session.add(new_interview)
                is_update = False
            
            db.session.commit()
            
            interview_details = {
                'date': interview_date,
                'type': interview_type,
                'link': interview_link,
                'location': location,
                'notes': notes
            }
            
            email_sent = send_interview_email(
                candidate_email=candidate.email,
                candidate_name=candidate.name,
                job_title=job.job_title,
                interview_details=interview_details,
                is_update=is_update
            )
            
            if email_sent:
                flash(f"Interview {'updated' if is_update else 'scheduled'} and email sent to {candidate.email}")
            else:
                flash(f"Interview {'updated' if is_update else 'scheduled'} but email failed to send", "warning")
            
            return redirect(f"/hr/sort/{result.job_id}")
            
        except Exception as e:
            flash(f"Error scheduling interview: {str(e)}")
            return redirect(request.url)
    
    existing_interview = InterviewSchedule.query.filter_by(
        job_id=job.id,
        candidate_id=candidate.id
    ).first()
    
    default_date = ""
    if existing_interview:
        default_date = existing_interview.interview_date.strftime("%Y-%m-%dT%H:%M")
    else:
        tomorrow = datetime.now() + timedelta(days=1)
        default_date = tomorrow.replace(hour=10, minute=0).strftime("%Y-%m-%dT%H:%M")
    
    return render_template("schedule_interview.html",
                         result=result,
                         job=job,
                         candidate=candidate,
                         existing_interview=existing_interview,
                         default_date=default_date)


@app.route("/hr/interviews")
def view_interviews():
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    interviews = InterviewSchedule.query.options(
        joinedload(InterviewSchedule.job)
    ).order_by(InterviewSchedule.interview_date.asc()).all()
    
    now = datetime.now()
    
    upcoming = [i for i in interviews if i.status == "Scheduled" and i.interview_date > now]
    past = [i for i in interviews if i.status == "Completed" or (i.status == "Scheduled" and i.interview_date <= now)]
    cancelled = [i for i in interviews if i.status == "Cancelled"]
    
    return render_template("interviews.html",
                         upcoming=upcoming,
                         past=past,
                         cancelled=cancelled)


@app.route("/hr/cancel-interview/<int:interview_id>")
def cancel_interview(interview_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    interview = InterviewSchedule.query.get_or_404(interview_id)
    interview.status = "Cancelled"
    db.session.commit()
    
    subject = f"Interview Cancelled: {interview.candidate_name}"
    body = f"Your interview scheduled for {interview.interview_date} has been cancelled."
    send_email(interview.candidate_email, subject, body)
    
    flash("Interview cancelled and candidate notified")
    return redirect("/hr/interviews")


@app.route("/hr/complete-interview/<int:interview_id>")
def complete_interview(interview_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    interview = InterviewSchedule.query.get_or_404(interview_id)
    interview.status = "Completed"
    db.session.commit()
    
    flash("Interview marked as completed")
    return redirect("/hr/interviews")


@app.route("/hr/sort/<int:job_id>")
def hr_sort_resumes(job_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")

    job = JobAnalysis.query.get_or_404(job_id)
    
    result_exists = exists().where(
        and_(
            CandidateResult.job_id == JobApplication.job_id,
            CandidateResult.candidate_id == JobApplication.candidate_id
        )
    )
    
    applications_to_process = JobApplication.query.filter(
        JobApplication.job_id == job_id,
        JobApplication.status == 'Active',
        ~result_exists
    ).all()
    
    all_active_applications = JobApplication.query.filter_by(
        job_id=job_id, 
        status='Active'
    ).all()
    
    apps_needing_update = []
    for app in all_active_applications:
        existing_result = CandidateResult.query.filter_by(
            job_id=job_id,
            candidate_id=app.candidate_id
        ).first()
        
        if existing_result:
            if existing_result.resume_file != app.resume_file:
                apps_needing_update.append(app)
                db.session.delete(existing_result)
    
    if apps_needing_update:
        db.session.commit()
        applications_to_process.extend(apps_needing_update)
    
    if applications_to_process:
        job_skills = [s.strip() for s in job.required_skills.split(",")]
        job_title = job.job_title
        
        job_text = f"""
        Job Title: {job.job_title}
        Required Skills: {job.required_skills}
        Minimum Experience: {job.min_experience} years
        """

        for app_entry in applications_to_process:
            resume_path = os.path.join("uploads", app_entry.resume_file)
            
            if not os.path.exists(resume_path):
                continue
                
            resume_text = extract_text(resume_path)
            
            final_score, score_breakdown = calculate_final_score_improved(
                resume_text, 
                job_text, 
                job_skills,
                job.min_experience,
                job_title
            )
            
            skills = extract_skills(resume_text)
            education = extract_education(resume_text)
            
            new_result = CandidateResult(
                job_id=job.id,
                candidate_id=app_entry.candidate_id,
                candidate_name=app_entry.candidate_name,
                score=final_score,
                skills=", ".join(skills),
                education=" | ".join(education),
                experience=score_breakdown['years'],
                resume_file=app_entry.resume_file,
                status="Pending"
            )
            db.session.add(new_result)

        db.session.commit()

    db_results = CandidateResult.query.filter_by(job_id=job_id).order_by(CandidateResult.score.desc()).all()
    
    pending_count = sum(1 for r in db_results if r.status == 'Pending')
    accepted_count = sum(1 for r in db_results if r.status == 'Accepted')
    rejected_count = sum(1 for r in db_results if r.status == 'Rejected')
    
    # Calculate top score with rounding to 2 decimal places
    top_score = round(db_results[0].score, 2) if db_results else 0
    
    ranked_results = []
    for rank, r in enumerate(db_results, start=1):
        # Round all scores to 2 decimal places for clean display
        display_score = round(r.score, 2)
        
        # Create breakdown with rounded values
        breakdown = {
            'semantic': round(display_score * 0.30, 1),
            'tfidf': round(display_score * 0.15, 1),
            'skills': round(display_score * 0.30, 1),
            'experience': round(display_score * 0.20, 1),
            'education': round(display_score * 0.05, 1)
        }
        
        ranked_results.append({
            "id": r.id,
            "rank": rank,
            "name": r.candidate_name,
            "score": display_score,  # Now rounded to 2 decimals
            "breakdown": breakdown,
            "skills": r.skills.split(", ") if r.skills else [],
            "education": r.education.split(" | ") if r.education else [],
            "experience": r.experience,
            "resume_file": r.resume_file,
            "status": r.status
        })

    return render_template("results.html", 
                         results=ranked_results, 
                         job_title=job.job_title,
                         min_experience=job.min_experience,
                         job_id=job_id,
                         pending_count=pending_count,
                         accepted_count=accepted_count,
                         rejected_count=rejected_count,
                         top_score=top_score)  # Pass rounded top score


@app.route("/view_resume/<filename>")
def view_resume(filename):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    return send_from_directory("uploads", filename)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

        if not HRUser.query.first():
            db.session.add(HRUser(
                username="hradmin",
                password_hash=generate_password_hash("admin123")
            ))
            db.session.commit()

    app.run(debug=True)
