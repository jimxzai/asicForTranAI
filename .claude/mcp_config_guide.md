# MCP Server Setup for ASIC-Fortran-AI Project

**Purpose**: Automate GitHub, LinkedIn updates, and job applications

---

## Quick Setup (3 MCP Servers - 15 minutes)

### 1. Git MCP Server (Priority 1)

**What it does**: Enhanced Git operations, automatic repo creation, PR management

**Install**:
```bash
# Install uv (Python package manager) if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# The MCP server will be installed automatically when configured
```

**Config** - Add to `~/.claude/mcp_settings.json`:
```json
{
  "mcpServers": {
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git", "--repository", "/Users/jimxiao/ai/asicForTranAI"]
    }
  }
}
```

**Benefits for our project**:
- ✅ Auto-create `spark-llama-safety` repo
- ✅ Push career materials in one command
- ✅ Manage branches for Fortran experiments
- ✅ Create PRs if you want to contribute to LFortran, Groq, etc.

---

### 2. Filesystem MCP Server (Priority 2)

**What it does**: Safe access to specific directories, useful for organizing outputs

**Install**:
```bash
npm install -g @modelcontextprotocol/server-filesystem
```

**Config** - Add to `~/.claude/mcp_settings.json`:
```json
{
  "mcpServers": {
    "git": { ... },  // from above
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/jimxiao/ai/asicForTranAI",
        "/Users/jimxiao/Documents/JobApplications"  // optional: separate folder for tracking
      ]
    }
  }
}
```

**Benefits**:
- ✅ Organize job application materials separately
- ✅ Safe access to multiple project directories
- ✅ Prevent accidental modifications outside project

---

### 3. Desktop Commander (Priority 3 - Optional)

**What it does**: GUI automation - screenshots, clicking, form filling

**Install**:
```bash
# No install needed, uses npx
```

**Config** - Add to `~/.claude/mcp_settings.json`:
```json
{
  "mcpServers": {
    "git": { ... },
    "filesystem": { ... },
    "desktop": {
      "command": "npx",
      "args": ["-y", "@anthropic/desktop-commander"]
    }
  }
}
```

**Benefits**:
- ✅ I can take screenshots to verify LinkedIn changes
- ✅ Help fill out job application forms (if tedious)
- ✅ Capture benchmark results visually
- ⚠️ Requires accessibility permissions on macOS

---

## Complete mcp_settings.json (Copy This)

**Location**: `~/.claude/mcp_settings.json`

```json
{
  "mcpServers": {
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git", "--repository", "/Users/jimxiao/ai/asicForTranAI"]
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/jimxiao/ai/asicForTranAI"
      ]
    }
  }
}
```

**Note**: Desktop Commander commented out by default (requires macOS permissions setup)

---

## Setup Steps (Run These Now)

```bash
# 1. Install uv (for Git MCP)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc  # or ~/.bashrc

# 2. Create MCP config directory
mkdir -p ~/.claude

# 3. Create config file (paste the JSON above)
cat > ~/.claude/mcp_settings.json << 'EOF'
{
  "mcpServers": {
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git", "--repository", "/Users/jimxiao/ai/asicForTranAI"]
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/jimxiao/ai/asicForTranAI"
      ]
    }
  }
}
EOF

# 4. Restart Claude Code (close and reopen the app)

# 5. Verify MCP servers loaded
# (In next Claude Code chat, I'll have new Git tools available)
```

---

## What This Enables (For Our Project)

### With Git MCP:
```
You: "Create spark-llama-safety repo on GitHub"
Me: [Uses Git MCP to create repo, push code, set description]
```

### With Filesystem MCP:
```
You: "Organize all career PDFs in a separate folder"
Me: [Creates JobApplications folder, copies materials with proper naming]
```

### With Desktop Commander (if enabled):
```
You: "Screenshot my LinkedIn profile after I update it"
Me: [Takes screenshot, saves to project folder]
```

---

## Testing (After Setup)

**Restart Claude Code**, then in a new chat:

```
Me: "List available MCP tools"
```

You should see:
- `git_*` tools (create repo, commit, push, etc.)
- `fs_*` tools (read/write specific directories)

---

## Next: Immediate Actions (With MCP Enabled)

Once MCP is set up, we can:

1. **Git MCP**:
   - Create GitHub repo for spark-llama-safety (1 command)
   - Push all career materials (1 command)
   - Initialize proper .gitignore, LICENSE, etc.

2. **Filesystem MCP**:
   - Create organized folder structure for job applications
   - Auto-generate PDFs from markdown
   - Track application status in CSV

3. **Continue Fortran Work**:
   - Deploy to Groq
   - Run benchmarks
   - Commit results to Git automatically

---

## Troubleshooting

**"uv: command not found"**:
```bash
# Install uv manually
brew install uv  # macOS with Homebrew
# OR
pip install uv
```

**"npx: command not found"**:
```bash
# Install Node.js
brew install node  # macOS
# OR download from nodejs.org
```

**"MCP server not showing up"**:
- Restart Claude Code (fully quit, reopen)
- Check `~/.claude/mcp_settings.json` syntax (must be valid JSON)
- Run `uvx mcp-server-git --version` manually to test

---

## Advanced (Later, If Needed)

**Database MCP** (track applications):
```json
"sqlite": {
  "command": "uvx",
  "args": ["mcp-server-sqlite", "--db-path", "/Users/jimxiao/ai/job_tracker.db"]
}
```

**GitHub API MCP** (automated PR creation to LFortran, Groq projects):
```json
"github": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-github"],
  "env": {
    "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
  }
}
```

---

**Status**: Ready to set up. Copy the commands above and run them in your terminal.

**Next**: After setup, I'll have Git tools to auto-create repos, push code, etc.
