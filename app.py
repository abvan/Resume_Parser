from flask import Flask, render_template, request, send_file
import os
import docx2txt
import uuid
import pandas as pd
from groq import Groq
from skill_extractor import extract_skills
import json

app = Flask(__name__)
client = Groq(api_key="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")  

def extract_text(file_storage):
    if file_storage.filename.endswith(".txt"):
        return file_storage.read().decode("utf-8")
    elif file_storage.filename.endswith(".docx"):
        return docx2txt.process(file_storage)
    else:
        raise ValueError("Unsupported file type")

def compare(jd_text, resume_text):
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)

    prompt = f"""
Job Description Skills:\n{', '.join(jd_skills)}\n
Resume Skills:\n{', '.join(resume_skills)}\n

Compare the two sets of skills above. 
- Return a match percentage based on overlap.
- Decide if the resume should be 'Shortlist', 'Hold', or 'Reject'. (Reject <40%, Hold 40-65%, Shortlist >65%)
- List the top 5 most relevant skills (no suggestions or explanations).
Return a JSON with keys: match_percent, status, top_skills.
"""

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "Always return a JSON with exact keys: match_percent, status, top_skills. Be consistent and deterministic."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            top_p=1.0
        )

        response_text = response.choices[0].message.content
        json_str = response_text[response_text.find("{"):response_text.rfind("}") + 1]
        return json.loads(json_str)

    except Exception as e:
        print("Parsing Error:", e)
        return {
            "match_percent": 0,
            "status": "Reject",
            "top_skills": []
        }

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        jd_files = request.files.getlist("jd_files")
        resume_files = request.files.getlist("resume_files")

        for jd_file in jd_files:
            jd_text = extract_text(jd_file)
            for resume_file in resume_files:
                resume_text = extract_text(resume_file)
                result = compare(jd_text, resume_text)
                results.append({
                    "jd": jd_file.filename,
                    "resume": resume_file.filename,
                    "match_percent": result.get("match_percent", 0),
                    "status": result.get("status", "Reject"),
                    "top_skills": ", ".join(result.get("top_skills", []))
                })

        df = pd.DataFrame(results)
        filename = f"results_{uuid.uuid4().hex}.csv"
        path = os.path.join("static", filename)
        df.to_csv(path, index=False)

        return render_template("index.html", results=results, csv_file=filename)

    return render_template("index.html", results=None, csv_file=None)

@app.route("/download_csv/<filename>")
def download_csv(filename):
    return send_file(os.path.join("static", filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
