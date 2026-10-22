"""
Exporte dans integration_info.txt ce qu'il faut pour adapter le Deployer a la branche d'integration.
Ne contient aucun secret : seuls les fichiers de code et de configuration du depot sont lus
(jamais .env). A lancer depuis la racine du depot, apres `git fetch` :

    python dump_branch_info.py
    python dump_branch_info.py origin/autre-branche       (autre branche, facultatif)

Ne pas enregistrer integration_info.txt dans Git : c'est un fichier temporaire.
"""
import re
import subprocess
import sys

BRANCH = sys.argv[1] if len(sys.argv) > 1 else "origin/data-checker+builder"
MAX_CHARS = 60_000
WANTED = re.compile(
    r"^(shared/.*\.(py|yaml)"
    r"|agents\.yaml|Makefile|requirements\.txt|Dockerfile.*|docker-compose.*"
    r"|agents/[^/]+/(agent|actions|analysis|stub)\.py"
    r"|agents/[^/]+/prompt\.md"
    r"|scenarios/.*|contracts/catalogue\.yaml"
    r"|harness/.*\.py|samples/(agents|schema)\..*)$")
SECRET = re.compile(r"(^|/)\.env($|\.)|\.pem$|\.key$")


def git(*args):
    out = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    return out.stdout if out.returncode == 0 else f"[erreur git : {out.stderr.strip()}]"


listing = git("ls-tree", "-r", "--name-only", BRANCH).splitlines()
if not listing or listing[0].startswith("[erreur"):
    sys.exit(f"Branche introuvable : {BRANCH}\nFais d'abord : git fetch\n{listing}")

lines = [f"BRANCHE : {BRANCH}", f"DERNIERS COMMITS :\n{git('log', '--oneline', '-8', BRANCH)}",
         "TOUS LES FICHIERS :", *listing]
sent = 0
for path in listing:
    if SECRET.search(path) or not WANTED.match(path):
        continue
    text = git("show", f"{BRANCH}:{path}")
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n[... tronque ...]"
    lines += ["", f"===== {path} =====", text]
    sent += 1

with open("integration_info.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print(f"OK : {len(listing)} fichiers listes, {sent} exportes dans integration_info.txt")
