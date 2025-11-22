# Git Workflow Guide

## Daily Workflow - Basic Order of Operations

### 1. Starting Your Work Day
```bash
# Always pull latest changes first before starting work
git pull origin main
```

### 2. Making Changes
```bash
# After making changes, check what you've modified
git status

# See what files changed
git diff                    # See all changes
git diff filename.py        # See changes in specific file
```

### 3. Staging Changes
```bash
# Add specific files
git add path/to/file.py

# Add all changes in current directory
git add .

# Add all changes in entire repo
git add -A
```

### 4. Committing Changes
```bash
# Commit with a descriptive message
git commit -m "Your commit message describing the changes"

# Example:
git commit -m "Update salary updater script to filter players with >2 points"
```

### 5. Pushing Your Work
```bash
# Push to remote repository
git push origin main

# If this is your first push on a new branch:
git push -u origin branch-name
```

---

## Common Scenarios

### Scenario 1: Someone else pushed changes while you were working

**Problem:** You tried to push but got "Your branch is behind 'origin/main'"

**Solution:**
```bash
# 1. Pull and merge remote changes
git pull origin main

# If there are conflicts, git will tell you. Then:
# 2a. If no conflicts: git will auto-merge, then push
git push origin main

# 2b. If conflicts exist: Resolve them, then:
git add .
git commit -m "Merge remote changes and resolve conflicts"
git push origin main
```

### Scenario 2: Working on a feature branch (recommended for new features)

```bash
# 1. Create and switch to new branch
git checkout -b feature/new-script

# 2. Make your changes, commit
git add .
git commit -m "Add new feature"

# 3. Push branch to remote
git push -u origin feature/new-script

# 4. On GitHub, create a Pull Request (PR)

# 5. After PR is merged, switch back to main and pull
git checkout main
git pull origin main

# 6. Delete local branch if done
git branch -d feature/new-script
```

### Scenario 3: Undoing changes

```bash
# Undo changes to a file (before staging)
git restore filename.py

# Undo all unstaged changes
git restore .

# Undo staging (unstage files)
git restore --staged filename.py

# Undo last commit (keeps changes)
git reset --soft HEAD~1

# Undo last commit (discards changes) - CAREFUL!
git reset --hard HEAD~1
```

### Scenario 4: Check what others changed

```bash
# Fetch remote changes without merging
git fetch origin

# See what changed on remote
git log origin/main..main

# See differences between your branch and remote
git diff origin/main main
```

---

## Typical Workflow Order

### Before Starting Work:
```bash
git pull origin main    # Get latest changes
git status              # Verify clean working tree
```

### During Work:
```bash
git status              # Check what's changed
git diff                # Review changes
```

### After Completing Work:
```bash
git add .               # Stage changes
git status              # Verify what's staged
git commit -m "..."     # Commit with message
git pull origin main    # Get any new remote changes (merge if needed)
git push origin main    # Push your changes
```

### If Push Fails (conflicts):
```bash
git pull origin main    # This will merge or show conflicts
# Resolve conflicts in files, then:
git add .
git commit -m "Merge remote changes"
git push origin main
```

---

## Useful Commands Reference

### Status & Info
```bash
git status              # See what's changed
git log                 # See commit history
git log --oneline       # Compact commit history
git branch              # List branches
git remote -v           # Show remote repositories
```

### Branching
```bash
git branch              # List branches
git branch branch-name  # Create branch
git checkout branch-name # Switch to branch
git checkout -b branch-name # Create and switch
git merge branch-name   # Merge branch into current
```

### Stashing (save work temporarily)
```bash
git stash               # Save uncommitted changes temporarily
git stash list          # See stashed changes
git stash pop           # Restore most recent stash
git stash apply         # Apply stash but keep it
```

### Pull vs Fetch
```bash
git pull origin main    # Fetch + merge (most common)
git fetch origin        # Just download changes (doesn't merge)
git fetch origin main   # Fetch specific branch
```

---

## Quick Reference Card

**Everyday Commands (in order):**
1. `git pull origin main` - Get latest
2. Make changes
3. `git add .` - Stage
4. `git commit -m "message"` - Commit
5. `git pull origin main` - Check for conflicts
6. `git push origin main` - Push

**If push fails:**
1. `git pull origin main` - Merge remote
2. Resolve conflicts if any
3. `git add .`
4. `git commit -m "Merge"`
5. `git push origin main`

---

## Best Practices

1. **Always pull before pushing** - Prevents conflicts
2. **Commit often** - Small, logical commits are better
3. **Write clear commit messages** - Describe what and why
4. **Don't force push to main** - Use `git push --force` only if you know what you're doing
5. **Use branches for features** - Keeps main clean
6. **Pull before starting work** - Get latest code
7. **Review changes with `git diff`** - Before committing

---

## Common Issues & Solutions

### "Your branch is behind 'origin/main'"
```bash
git pull origin main
# Resolve conflicts if needed
git push origin main
```

### "Please commit your changes or stash them"
```bash
# Option 1: Commit changes
git add .
git commit -m "WIP: work in progress"

# Option 2: Stash changes
git stash
# Do your pull/merge
git stash pop
```

### Accidentally committed to wrong branch
```bash
# Undo commit but keep changes
git reset --soft HEAD~1
# Switch to correct branch
git checkout correct-branch
# Commit there
git commit -m "message"
```

