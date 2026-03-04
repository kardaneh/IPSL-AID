Pre-Push Workflow
=================

This section outlines the standardized workflow to follow **before pushing
changes to the IPSL-AID repository**. Adhering to this workflow ensures that your contributions are clean, tested, and compatible with the latest codebase, facilitating smooth collaboration and maintaining code quality.

.. contents:: Table of Contents
   :depth: 2
   :local:

Why This Workflow Matters
-------------------------

A disciplined pre-push workflow ensures:

- **Reproducibility** of experiments
- **Model versioning** and experiment tracking
- **Code quality** for both ML and data processing components
- **Smooth collaboration** between researchers with different expertise

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

    Your branch is behind 'origin/main' by X commits

You must update your branch before pushing to avoid integration issues.

3. Rebase Onto Latest Remote Branch
-----------------------------------

Rebase your feature branch onto the latest version of the base branch:

.. code-block:: bash

    git pull --rebase origin main

(Replace ``main`` with your base branch, e.g., ``develop`` or ``experimental``.)

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
     - For feature branches before PR

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
    # Remote changes from origin/main - added residual connections
    class DownscalingUNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=3, hidden_dims=[64, 128, 256],
                     use_residual=True):
            super().__init__()
            self.encoder = Encoder(in_channels, hidden_dims, use_residual)
    >>>>>>> origin/main

**Understanding the markers:**

- ``<<<<<<< HEAD`` → Your current branch's version
- ``=======`` → Separator between conflicting versions
- ``>>>>>>> origin/main`` → Remote branch's version

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

After fixing each file, and passing the pre-commit hooks (see next section), stage the resolved files:

.. code-block:: bash

    git add IPSL_AID/networks.py
    git add IPSL_AID/dataset.py

**Do not** use ``git add .`` blindly - ensure only resolved files are staged.

Step 5 — Continue the Rebase
----------------------------

.. code-block:: bash

    git rebase --continue

If more conflicts appear, repeat the process. Git will apply each commit one by one.

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

5. Run the Test Suite
---------------------

Before pushing, verify your changes don't break existing functionality:

.. code-block:: bash

    # Run the full test suite with pytest
    python -m tests.test_all

    # For a specific module
    python -m tests.test_all networks

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

Finally, push your branch to the remote repository:

.. code-block:: bash

    git push

**If push is rejected** (due to history rewrite from rebase):

.. code-block:: bash

    git push --force-with-lease

⚠ **Critical:** Always use ``--force-with-lease``, never plain ``--force``.

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

For quick daily use, here's the complete workflow in one block:

.. code-block:: bash

    # Step 1-2: Update and check status
    git fetch origin
    git status

    # Step 3: Rebase onto latest main
    git pull --rebase origin main
    # (Resolve conflicts if needed)

    # Step 4: Run pre-commit hooks
    pre-commit run --all-files

    # Step 5: Run tests
    python -m tests.test_all

    # Step 6: Commit if needed
    git add .
    git commit -m "type: your message here"

    # Step 7: Push safely
    git push origin your-branch

Common Pitfalls to Avoid
------------------------

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
   * - Changing random seeds
     - Document or make configurable
   * - Forgetting to update requirements
     - Run ``uv pip list`` and update ``pyproject.toml``

Important Rules Summary
-----------------------

✅ **DO:**

- Fetch before working
- Rebase feature branches
- Resolve conflicts carefully
- Run pre-commit hooks
- Test thoroughly
- Use ``--force-with-lease``
- Document experiments
- Version control configurations

❌ **DON'T:**

- Ignore conflicts
- Leave conflict markers
- Skip tests after changes
- Push without rebasing
- Use plain ``--force``
- Commit large data files
- Commit notebooks with outputs
- Hardcode model paths or seeds


Following this workflow ensures that your contributions to IPSL-AID integrate
smoothly with the work of other researchers and maintain the high standards
required for AI-based climate downscaling research.
