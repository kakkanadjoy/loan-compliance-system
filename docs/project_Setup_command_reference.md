# Project setup command reference (Windows / PowerShell)

Everything we used from zero to a working, tested, CI-verified project — organized so you can reuse it for any new project. Commands marked **[this project]** are specific to the loan-compliance system; everything else is generic.

---

## 1. One-time machine setup (never again after the first time)

```powershell
git --version                      # check Git is installed (install from git-scm.com if not)
python --version                   # check Python (3.10+; install from python.org, CHECK "Add to PATH")

git config --global user.name "Your Name"
git config --global user.email "you@youremail.com"   # same email as your GitHub account

# Allow venv activation scripts to run (one-time, answer Y if prompted)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 2. Start a new project folder

```powershell
cd C:\Users\YourName\projects      # or wherever you keep projects
mkdir my-new-project
cd my-new-project
```

Or open the folder in VS Code (File → Open Folder) and use its terminal: **Ctrl+`** — it opens already in the project folder.

---

## 3. Virtual environment (do this per project, before installing anything)

```powershell
python -m venv .venv               # create the venv (a private Python for this project)
.venv\Scripts\Activate.ps1         # activate it — prompt shows (.venv) when active
python -m pip install --upgrade pip
```

Remember:
- Re-activate every new terminal session (VS Code auto-activates if you accept its popup).
- Add a line containing `.venv/` to your `.gitignore` — never commit the venv.

---

## 4. Install dependencies

```powershell
pip install -r requirements.txt    # if the project has a requirements file
pip install <package>              # ad-hoc installs
pip install -U <package>           # upgrade a package
pip install "<package><2.1"        # pin below a version — QUOTES required in PowerShell
```

Lesson learned in this project: when a specific version fixes a bug
(xgboost <2.1 for shap compatibility), also pin it in `requirements.txt`
(e.g. `xgboost>=2.0,<2.1`) and commit — so every machine and CI gets the
working version, not just your laptop.

---

## 5. Put the project on GitHub (first time for a repo)

Local side:

```powershell
git init                           # turn the folder into a repository
git add .                          # stage all files (.gitignore exclusions respected)
git commit -m "Initial commit"     # save the first snapshot
git branch -M main                 # ensure the branch is named main
```

GitHub side: github.com → + → New repository → name it → Public/Private →
leave ALL checkboxes unchecked (no README, no .gitignore, no license) → Create.

Connect and push:

```powershell
git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin main            # authenticate via the browser popup
```

Common stumbles:
- Push "rejected" / "fetch first" → you initialized the GitHub repo with a README.
  Fix: `git pull origin main --rebase --allow-unrelated-histories` then push again.
- "Password authentication was removed" → use a Personal Access Token
  (GitHub → Settings → Developer settings → Tokens) as the password.

---

## 6. Everyday Git workflow (after the repo exists)

```powershell
git status                         # what changed?
git add <file>                     # stage a specific file (or git add . for all)
git commit -m "Describe the change"
git push                           # upload; triggers CI if the repo has a workflow
git pull                           # download changes (when working across machines)
git log --oneline                  # compact history
```

Then check the **Actions** tab on GitHub for the green checkmark.

---

## 7. [this project] Full build pipeline

Run from the project root with the venv active:

```powershell
pip install -r requirements.txt

# Generate ground-truth data
python backend\ml_training\synthetic_data.py --n 500     # 504 records (incl. 4 demos)
python backend\ml_training\render_documents.py --limit 50 # 54 PDF packets

# Train both models (must run from inside backend\)
cd backend
python ml_training\train_compliance_model.py   # expect r2≈0.94, miss_rate≈0.04
python ml_training\train_escalation_model.py   # expect auc≈0.99, recall=1.0

# Run the test suite (same as CI)
python -m pytest tests/ -q                     # expect: 4 passed
cd ..
```

---

## 8. Handy PowerShell basics used along the way

```powershell
cd <path>                          # change directory
cd ..                              # go up one level
dir                                # list files (equivalent of ls)
mkdir <name>                       # create a folder
```

---

## Quick-start recipe for any future project

```powershell
mkdir new-project; cd new-project
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# ...add your code, requirements.txt, .gitignore (include .venv/)...
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/new-project.git
git push -u origin main
```