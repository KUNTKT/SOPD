# Push to GitHub

This directory is already `git init`’d on branch `main`. Files are staged.
Set your git identity (this repo does not touch `git config`), then:

```bash
cd /home/yiyangba/ssopd-release
git commit -m "Initial public slice: scripts, configs, notes, UCE libraries."

# new empty GitHub repo, then:
git remote add origin git@github.com:<USER>/ssopd-agent-experiments.git
git push -u origin main
```

Or with GitHub CLI:

```bash
gh repo create ssopd-agent-experiments --public --source=. --remote=origin --push
```

Do not upload `/home/yiyangba/ssopd_paper_archive` itself (6GB+ rollouts and LoRAs).
