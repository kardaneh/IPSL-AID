Pre-Push Workflow
=================

This section outlines the standardized workflow to follow **before pushing
changes to the IPSL-AID repository**. Adhering to this workflow ensures that
your contributions are clean, tested, and compatible with the latest codebase,
facilitating smooth collaboration and maintaining code quality.

.. contents:: Table of Contents
   :depth: 2
   :local:

Branch Strategy Overview
------------------------

IPSL-AID uses a **two-tier branch system** to balance collaboration and
individual development. Based on your repository structure:

.. code-block:: bash

    $ git branch -a
      feature/you               # Your individual feature branch
    * main                      # Production branch (protected)
      remotes/origin/Dev        # Team integration branch (capital D!)
      remotes/origin/feature/you
      remotes/origin/gh-pages   # Documentation branch
      remotes/origin/main

Branch Types and Their Purposes
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 35 20

   * - Branch
     - Pattern
     - Purpose
     - Who Can Push
   * - **Development**
     - ``Dev`` (note: capital D)
     - Integration branch for team collaboration
     - All team members
   * - **Feature**
     - ``feature/*``
     - Individual development sandboxes
     - Only the creator
   * - **Main**
     - ``main``
     - Production-ready code
     - Admin/PR only
   * - **Documentation**
     - ``gh-pages``
     - Project documentation
     - Admin only

When to Use Each Branch
-----------------------

**Use ``Dev`` (team branch) when:**
- Multiple people are contributing to the same feature
- You need immediate integration with teammates' work
- Fixing critical bugs that affect everyone
- Doing rapid prototyping with pair programming

**Use ``feature/*`` (individual branch) when:**
- Working alone on a specific feature (like ``feature/you``)
- Experimenting with new approaches
- Developing code that might break things
- Creating a safe sandbox for exploration

**Important Notes:**
- ``main`` is **protected** - never push directly to it
- ``gh-pages`` is for documentation only
- All work flows from ``feature/*`` → ``Dev`` → ``main``
- **Note the capitalization:** The team branch is ``Dev`` (capital D), not ``dev``

Why This Workflow Matters
-------------------------

A disciplined pre-push workflow ensures:

- **Reproducibility** of experiments
- **Model versioning** and experiment tracking
- **Code quality** for both ML and data processing components
- **Smooth collaboration** between researchers with different expertise
- **Clean branch history** distinguishing team work (``Dev``) from individual
  exploration (``feature/*``)

Prerequisites
-------------

Before starting, ensure you have:

- A clean working directory (or stashed changes)
- Access to the IPSL-AID repository
- Required tools installed:

  .. code-block:: bash

      uv --version          # Fast Python package manager
      pre-commit --version  # Git hooks for code quality
      git --version         # Version control
      python --version      # Python 3.9+ recommended

Understanding Your Current Branches
-----------------------------------

Based on a typical IPSL-AID setup:

.. code-block:: bash

    # View all branches (local and remote)
    git fetch
    git branch -a

    # You should see:
      feature/you           # Your personal feature branch
    * main                      # You're currently on main (production)
      remotes/origin/Dev        # Team development branch (capital D!)
      remotes/origin/feature/you
      remotes/origin/gh-pages
      remotes/origin/main

**Important:** Note that the team branch is ``Dev`` (capital D), not ``dev``.
Always use the exact capitalization:

.. code-block:: bash

    git checkout Dev            # ✅ Correct
    git checkout dev            # ❌ Wrong - branch doesn't exist

---

Branch-Specific Workflows
-------------------------

Workflow A: Working on ``Dev`` (Team Collaboration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**⚠️ CRITICAL:** ``Dev`` is a shared branch where ALL team members can push
directly. Strict discipline is required.

Step 1: Switch to Dev and Fetch Latest
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git checkout Dev
    git fetch origin
    git status

Expected output: "Your branch is up to date with 'origin/Dev'"

Step 2: Pull with Rebase (Mandatory)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git pull --rebase origin Dev

**Why mandatory on ``Dev``:** Multiple team members push frequently. Rebase
prevents unnecessary merge commits.

Step 3: Make Your Changes
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Make focused, small commits
    git add <specific-file.py>
    git commit -m "fix: correct normalization in preprocessing"

    # For larger features, commit frequently
    git add .
    git commit -m "wip: add attention mechanism (unfinished)"

**Commit rules for ``Dev``:**
- ✅ Small, logical commits
- ✅ Use conventional commits (feat:, fix:, docs:, test:)
- ✅ Reference issue numbers when possible
- ❌ No half-broken code on Dev

Step 4: Run Pre-commit Hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pre-commit run --all-files

If hooks modify files:

.. code-block:: bash

    git add .
    pre-commit run --all-files  # Verify clean

Step 5: Run Tests
^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Full test suite (mandatory for Dev)
    python -m pytest tests/ -v

    # Or specific module if changes are isolated
    python -m pytest tests/test_preprocessing.py -v

**Success criteria for ``Dev``:**
- ✅ **All** tests must pass (100%)
- ✅ Coverage must not decrease
- ✅ No new warnings

Step 6: Push to Dev
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git push origin Dev

**If push is rejected** (someone pushed while you were working):

.. code-block:: bash

    git pull --rebase origin Dev
    # Resolve any conflicts
    pre-commit run --all-files
    python -m pytest tests/ -v
    git push origin Dev

**🚫 NEVER use ``--force`` or ``--force-with-lease`` on ``Dev``** - this will
overwrite teammates' work!

Workflow B: Working on ``feature/*`` (Individual Development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Note:** ``feature/*`` branches are your **private sandbox**. You have complete
freedom, but follow these guidelines for smooth integration later.

Step 1: Create or Switch to Feature Branch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Create new feature branch from latest Dev
    git checkout Dev
    git pull --rebase origin Dev
    git checkout -b feature/your-name-description

    # Example: feature/you-attention-unet

    # Or switch to existing feature branch (like yours)
    git checkout feature/you

**Branch naming convention:**

.. code-block:: text

    feature/{developer-name}-{brief-description}
    Examples:
    - feature/you                    # Simple name (ok for personal use)
    - feature/you-attention-unet     # Descriptive (recommended)
    - feature/sarah-data-augmentation
    - feature/kevin-loss-function-experiment

Step 2: Work Freely (But Smartly)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Make changes, experiment, break things - it's your sandbox!
    git add .
    git commit -m "experiment: try deeper U-Net with 5 downsampling layers"

    # Commit often - you can squash later
    git commit -m "wip: testing different activation functions"

    # Create checkpoint before risky changes
    git tag experiment-checkpoint-1

**Freedom on ``feature/*``:**
- ✅ Commit broken code temporarily
- ✅ Experiment with different approaches
- ✅ Rewrite history (rebase, squash, amend)
- ✅ Push whenever you want

**But remember:**
- ⚠️ Keep commits meaningful (even experiments should be documented)
- ⚠️ Don't commit large datasets or model checkpoints
- ⚠️ Clean up before creating Pull Request to ``Dev``

Step 3: Sync with ``Dev`` Regularly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Even on feature branches, stay updated with team changes:

.. code-block:: bash

    # Fetch latest Dev changes
    git fetch origin

    # Rebase your feature onto latest Dev (keeps history clean)
    git rebase origin/Dev

    # Or merge if you prefer (creates merge commit)
    git merge origin/Dev

**When to sync:**
- Daily at start of work
- Before running experiments (ensure latest preprocessing)
- Before creating Pull Request to ``Dev``

Step 4: Prepare for Integration (Before Push to Remote)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When your feature is ready to share with the team:

.. code-block:: bash

    # 1. Ensure you're on your feature branch
    git branch  # Should show * feature/you

    # 2. Rebase onto latest Dev
    git fetch origin
    git rebase origin/Dev

    # 3. Run pre-commit hooks
    pre-commit run --all-files

    # 4. Run tests (your feature should pass all tests)
    python -m pytest tests/ -v

    # 5. Squash messy commits (optional but recommended)
    git rebase -i origin/Dev
    # Change 'pick' to 'squash' for intermediate commits

    # 6. Push to remote (first time or after rebase)
    git push -u origin feature/your-name

    # After rebase (history rewritten), force-push safely:
    git push --force-with-lease origin feature/your-name

Step 5: Create Pull Request to ``Dev``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # From GitHub UI or using gh CLI
    gh pr create --base Dev --head feature/your-name \
      --title "feat: add attention mechanism to U-Net" \
      --body "## Summary\n- Implements multi-head attention\n- Improves downscaling accuracy by 15%\n\n## Testing\n- [x] Unit tests pass\n- [x] Integration tests pass\n- [x] Pre-commit hooks pass"

**PR requirements for merging to ``Dev``:**
- ✅ All tests pass
- ✅ No merge conflicts with ``Dev``
- ✅ At least one review from another team member
- ✅ Pre-commit hooks passed

---

1. Fetch Latest Changes From Remote
-----------------------------------

Always begin by updating your local knowledge of the remote repository without
modifying your working files:

.. code-block:: bash

    git fetch origin

This command:
- Downloads new data from remote branches
- Updates ``origin/*`` references
- Does **not** merge or rebase your working files

**Why this matters for IPSL-AID:** Multiple researchers may be working on
different modules, preprocessing pipelines, or evaluation metrics simultaneously.

**For ``Dev`` branch:** Run this frequently to see what teammates have pushed.

**For ``feature/*`` branch:** Run this to stay aware of changes in ``Dev``
without affecting your work.

2. Check Branch Status
----------------------

Examine your current branch status:

.. code-block:: bash

    git status

**Interpretation Guide:**

.. list-table::
   :header-rows: 1
   :widths: 30 40

   * - Status Message
     - Required Action
   * - "Your branch is up to date"
     - Proceed to step 3
   * - "Your branch is ahead"
     - Your changes are ready to push
   * - "Your branch is behind"
     - **MUST** update before pushing (see step 3)
   * - "Changes not staged"
     - Stage your changes with ``git add``
   * - "Unmerged paths"
     - You have unresolved conflicts

If you see:

.. code-block:: text

    Your branch is behind 'origin/Dev' by X commits

You must update your branch before pushing to avoid integration issues.

3. Rebase Onto Latest Remote Branch
-----------------------------------

For ``feature/*`` Branches (Individual)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rebase your feature branch onto the latest version of ``Dev``:

.. code-block:: bash

    git fetch origin
    git rebase origin/Dev

For ``Dev`` Branch (Team)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git pull --rebase origin Dev

**Why rebase instead of merge?**

.. list-table::
   :header-rows: 1
   :widths: 15 30 30

   * - Approach
     - Result
     - Use Case
   * - **Merge**
     - Creates merge commits, more complex history
     - When preserving experiment history for papers
   * - **Rebase**
     - Linear, clean history
     - For feature branches before PR to ``Dev``

For IPSL-AID development, **rebase is preferred** for feature branches to maintain
a readable project history, especially when tracking model iterations.

Conflict Resolution Guide
-------------------------

Conflicts occur when Git cannot automatically reconcile changes. This is common
in collaborative development, especially when multiple researchers modify:

- **Model architecture definitions** (``networks.py``)
- **Training Module** (``main.py``)
- **Data preprocessing pipelines** (``dataset.py``)
- **Dependency specifications** (``pyproject.toml``)

On ``Dev`` Branch (Team Collaboration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Conflicts on ``Dev`` are serious** - they block everyone. Resolve immediately:

.. code-block:: bash

    git pull --rebase origin Dev
    # If conflict appears:

**When a conflict occurs**, Git will pause and display:

.. code-block:: text

    CONFLICT (content): Merge conflict in networks.py
    error: could not apply abc1234... feat: add attention mechanism to U-Net

Step 1 — Identify Conflicted Files
----------------------------------

.. code-block:: bash

    git status

Look for files under:

.. code-block:: text

    Unmerged paths:
      both modified:   IPSL_AID/networks.py
      both modified:   IPSL_AID/dataset.py

Step 2 — Examine the Conflict
-----------------------------

Open each conflicted file. You'll see conflict markers:

.. code-block:: python

    <<<<<<< HEAD
    # Your local changes - experimenting with deeper network
    class DownscalingUNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=3, hidden_dims=[64, 128, 256, 512]):
            super().__init__()
            self.encoder = Encoder(in_channels, hidden_dims)
    =======
    # Remote changes from origin/Dev - added residual connections
    class DownscalingUNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=3, hidden_dims=[64, 128, 256],
                     use_residual=True):
            super().__init__()
            self.encoder = Encoder(in_channels, hidden_dims, use_residual)
    >>>>>>> origin/Dev

**Understanding the markers:**

- ``<<<<<<< HEAD`` → Your current branch's version
- ``=======`` → Separator between conflicting versions
- ``>>>>>>> origin/Dev`` → Remote branch's version

Step 3 — Resolve the Conflict
-----------------------------

Edit the file to create the correct version. For model code:

1. **Preserve both innovations** if they're compatible
2. **Test the combined architecture** mentally or with quick local tests
3. **Check parameter compatibility** with existing training configs
4. **Document architectural decisions** in comments

Example resolution combining both approaches:

.. code-block:: python

    # Resolved: deeper network with residual connections
    class DownscalingUNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=3, hidden_dims=[64, 128, 256, 512],
                     use_residual=True):
            super().__init__()
            self.encoder = Encoder(in_channels, hidden_dims, use_residual)

**Critical:** Remove **ALL** conflict markers:

.. code-block:: text

    <<<<<<<
    =======
    >>>>>>>

Step 4 — Mark as Resolved
-------------------------

After fixing each file, and passing the pre-commit hooks (see next section),
stage the resolved files:

.. code-block:: bash

    git add IPSL_AID/networks.py
    git add IPSL_AID/dataset.py

**Do not** use ``git add .`` blindly - ensure only resolved files are staged.

Step 5 — Continue the Rebase
----------------------------

.. code-block:: bash

    git rebase --continue

If more conflicts appear, repeat the process. Git will apply each commit one by one.

Step 6 — Notify Team (For ``Dev`` Only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After resolving conflicts on ``Dev``, immediately notify the team:

.. code-block:: bash

    # In Slack #ipsl-aid:
    "Resolved merge conflicts in networks.py - please pull latest Dev"

On ``feature/*`` Branches (Individual)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conflicts on feature branches are **your responsibility only**:

.. code-block:: bash

    git rebase origin/Dev
    # Resolve conflicts using same process above
    git add <resolved-files>
    git rebase --continue

**You have more flexibility:**
- Can abort rebase: ``git rebase --abort``
- Can squash conflicting commits
- Can start over with fresh branch

Abort Rebase (Emergency Option)
-------------------------------

If the rebase becomes too complex or you need to start over:

.. code-block:: bash

    git rebase --abort

This returns your branch to its state before starting the rebase.

**When to abort:**
- You're unsure about conflict resolutions
- You need to discuss architectural changes with the team
- You accidentally started rebase on wrong branch

4. Standardize Code with Pre-commit Hooks
-----------------------------------------

IPSL-AID uses pre-commit hooks to enforce code quality standards. After successful
rebase, run all hooks:

.. code-block:: bash

    pre-commit run --all-files

**What these hooks check (Python-focused):**

.. list-table::
   :header-rows: 1
   :widths: 20 50

   * - Hook
     - Purpose
   * - **black**
     - Consistent Python code formatting
   * - **isort**
     - Sorts imports alphabetically
   * - **flake8**
     - PEP 8 compliance and style issues
   * - **mypy**
     - Type hint checking (critical for ML code)
   * - **pylint**
     - Code quality and best practices
   * - **pydocstyle**
     - Docstring conventions for documentation
   * - **nbqa**
     - Applies tools to Jupyter notebooks (if present)
   * - **yaml validators**
     - Configuration file syntax (for model configs)
   * - **trailing-whitespace**
     - Clean diffs
   * - **check-json**
     - Validates JSON files (for experiment configs)

**Because the hooks modify files automatically:**

.. code-block:: bash

    git add .

Then run pre-commit again to confirm everything is clean:

.. code-block:: bash

    pre-commit run --all-files

**Expected output:** "All files passed" or similar success message.

Branch-Specific Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20

   * - Hook
     - Required on ``Dev``
     - Required on ``feature/*``
   * - black
     - ✅ Always
     - ✅ Before PR
   * - isort
     - ✅ Always
     - ✅ Before PR
   * - flake8
     - ✅ Always
     - ✅ Before PR
   * - mypy
     - ✅ Always
     - Recommended
   * - pylint
     - ✅ Always
     - Recommended
   * - trailing-whitespace
     - ✅ Always
     - ✅ Always

5. Run the Test Suite
---------------------

Before pushing, verify your changes don't break existing functionality:

.. code-block:: bash

    # Run the full test suite with pytest
    python -m pytest tests/ -v

    # For a specific module
    python -m pytest tests/test_networks.py -v

**Success criteria:**

- ✅ All tests pass (0 failures)
- ✅ Coverage doesn't decrease significantly
- ✅ No warnings about deprecated functions
- ✅ Tests complete in reasonable time

**If tests fail:**

- Examine error messages carefully
- Check if failures relate to your changes
- Fix issues locally
- Re-run tests until they pass

Branch-Specific Testing Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**On ``Dev`` branch (Strict):**
- Full test suite **must** pass
- Coverage must not decrease
- No skipped tests allowed

**On ``feature/*`` branch (Flexible):**
- During development: test specific modules
- Before PR to ``Dev``: full suite must pass
- Document any known failures in PR description

6. Commit Changes (If Needed)
-----------------------------

If you made additional fixes (conflict resolution, formatting, test fixes):

.. code-block:: bash

    git add .
    git commit -m "fix: resolve merge conflicts and apply formatting"

**Commit message guidelines for IPSL-AID (Conventional Commits):**

.. list-table::
   :header-rows: 1
   :widths: 15 50

   * - Type
     - Example
   * - ``feat:``
     - feat: add attention U-Net for precipitation downscaling
   * - ``fix:``
     - fix: correct normalization in data preprocessing
   * - ``docs:``
     - docs: update model card for U-Net architecture
   * - ``test:``
     - test: add validation tests for GAN discriminator
   * - ``refactor:``
     - refactor: simplify loss function computation
   * - ``perf:``
     - perf: optimize dataloading with parallel workers
   * - ``config:``
     - config: update training hyperparameters for v2
   * - ``experiment:``
     - experiment: log results of downscaling ablation

**If no changes were needed** after rebase and hooks, you may not need a new commit.

7. Push Your Changes
--------------------

Pushing to ``Dev`` Branch (Team)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git push origin Dev

**⚠️ CRITICAL:** Never use force push on ``Dev``:

.. code-block:: bash

    git push --force origin Dev              # ❌ DANGEROUS - overwrites teammates' work
    git push --force-with-lease origin Dev   # ❌ Also dangerous on shared branches
    git push origin Dev                      # ✅ Only safe option

Pushing to ``feature/*`` Branch (Individual)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git push origin feature/you

**If push is rejected** (due to history rewrite from rebase):

.. code-block:: bash

    git push --force-with-lease origin feature/you

⚠ **Critical:** Always use ``--force-with-lease`` on feature branches, never
plain ``--force``.

.. list-table::
   :header-rows: 1
   :widths: 20 50

   * - Option
     - Safety
   * - ``--force``
     - Overwrites remote branch **blindly** - DANGEROUS
   * - ``--force-with-lease``
     - Checks if remote branch has changed since your last fetch - **SAFER**

Quick Reference: Daily Workflow
-------------------------------

For ``Dev`` Branch (Team Collaboration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Start of day
    git checkout Dev
    git fetch origin
    git pull --rebase origin Dev

    # Make changes
    git add <files>
    git commit -m "type: description"

    # Before push
    pre-commit run --all-files
    python -m pytest tests/ -v

    # Push (never force!)
    git push origin Dev

For ``feature/*`` Branch (Individual Development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Start new feature
    git checkout Dev
    git pull --rebase origin Dev
    git checkout -b feature/your-name-description

    # Daily work (sync with Dev)
    git fetch origin
    git rebase origin/Dev

    # Make commits freely
    git add .
    git commit -m "experiment: try X"

    # Before sharing (PR to Dev)
    git fetch origin
    git rebase origin/Dev
    pre-commit run --all-files
    python -m pytest tests/ -v

    # Push to remote
    git push -u origin feature/your-name-description

    # After rebasing (history rewritten)
    git push --force-with-lease origin feature/your-name-description

Complete Workflow Example (Your Current Situation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on your branches (``feature/you`` and ``Dev``):

.. code-block:: bash

    # You're currently on main - switch to your feature branch
    git checkout feature/you

    # Sync with latest Dev changes
    git fetch origin
    git rebase origin/Dev

    # Make your changes
    # ... edit files ...

    # Stage and commit
    git add .
    git commit -m "feat: add attention mechanism to downscaling model"

    # Run quality checks
    pre-commit run --all-files
    python -m pytest tests/ -v

    # Push to your feature branch
    git push --force-with-lease origin feature/you

    # Create Pull Request to Dev when ready
    gh pr create --base Dev --head feature/you

Common Pitfalls to Avoid
------------------------

On ``Dev`` Branch (Team)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 40

   * - Pitfall
     - Solution
   * - Using ``--force`` on Dev
     - Wipes teammates' work - **NEVER DO THIS**
   * - Pushing without pulling
     - Rejection, wasted time
   * - Skipping tests
     - Breaking CI/CD for everyone
   * - Large commits
     - Hard to review, harder to revert
   * - Pushing broken code
     - Blocks all team members

On ``feature/*`` Branches (Individual)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 40

   * - Pitfall
     - Solution
   * - Never syncing with Dev
     - Massive conflicts at PR time
   * - Committing large models
     - Repository bloat
   * - Notebooks with outputs
     - Huge diffs, conflicts
   * - Hardcoded paths
     - Breaks on other machines
   * - Changing random seeds
     - Non-reproducible experiments

General Pitfalls
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 40

   * - Pitfall
     - Solution
   * - Committing large model checkpoints
     - Use DVC or model registry instead
   * - Notebooks with huge outputs
     - Clear outputs before commit
   * - Hardcoded paths
     - Use pathlib and relative paths
   * - Ignoring type hints
     - Add mypy to pre-commit and fix warnings
   * - Changing random seeds without documentation
     - Document or make configurable
   * - Forgetting to update dependencies
     - Run ``uv pip list`` and update ``pyproject.toml``

Important Rules Summary
-----------------------

✅ **DO:**

**On all branches:**
- Fetch before working
- Run pre-commit hooks
- Test thoroughly
- Use conventional commits
- Document experiments
- Version control configurations

**On ``Dev`` specifically:**
- Pull with rebase before push
- Keep Dev always green
- Communicate large changes
- **Never force push**

**On ``feature/*`` specifically:**
- Sync with Dev daily
- Use descriptive branch names
- Clean up before PR
- Use ``--force-with-lease`` safely

❌ **DON'T:**

**On any branch:**
- Ignore conflicts
- Leave conflict markers
- Skip tests after changes
- Commit large data files
- Commit notebooks with outputs
- Hardcode model paths or seeds

**On ``Dev`` specifically:**
- **NEVER use ``--force`` or ``--force-with-lease``**
- Push without running tests
- Make massive commits
- Bypass pre-commit hooks

**On ``feature/*`` specifically:**
- Let branch get too stale (>1 week without rebase)
- Create PR without testing
- Forget to squash messy commits

Branch Workflow Diagram
-----------------------

.. code-block:: text

    Team Collaboration Flow:
    ┌─────────────────┐
    │  feature/X      │──┐
    └─────────────────┘  │
                         │
    ┌─────────────────┐  │    ┌─────┐    ┌──────┐
    │  feature/xox    │──┼───→│ Dev │───→│ main │
    └─────────────────┘  │    └─────┘    └──────┘
                         │
    ┌─────────────────┐  │
    │ feature/you     │──┘
    └─────────────────┘

    Individual Work (Your Current Setup):
    feature/you ──PR──→ Dev ──Release──→ main
          ↑                        ↑
      Your sandbox            Team integration
      - Experiment freely      - Stable
      - Can break             - Always tested
      - Push anytime          - Protected

Following this workflow ensures that your contributions to IPSL-AID integrate
smoothly with the work of other researchers while maintaining the flexibility
needed for individual exploration. Remember: ``Dev`` is for team collaboration,
``feature/*`` is your personal sandbox!
