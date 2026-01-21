#!/usr/bin/env python3
"""
Status line generator for Claude Code integration.
Reads JSON input from stdin and outputs formatted status line.

Focuses on context window usage - the key metric for predicting
when Claude's response quality may start to degrade.
"""

import json
import sys
import os
import select
from pathlib import Path
from git_info import GitInfo

def read_stdin_data() -> dict:
    """Read JSON data passed via stdin from Claude Code."""
    stdin_data = {}

    try:
        # Check if there's data available on stdin (non-blocking)
        if select.select([sys.stdin], [], [], 0.1)[0]:
            raw_input = sys.stdin.read()
            if raw_input.strip():
                try:
                    stdin_data = json.loads(raw_input)
                except json.JSONDecodeError:
                    pass
    except:
        pass

    return stdin_data

def get_context_color(percentage: float) -> tuple:
    """Get RGB color based on context usage percentage."""
    if percentage < 40:
        return (0, 255, 0)      # Green - safe zone
    elif percentage < 65:
        return (255, 255, 0)    # Yellow - getting warm
    elif percentage < 80:
        return (255, 165, 0)    # Orange - caution, consider wrapping up
    else:
        return (255, 100, 100)  # Red - danger zone, quality may degrade

def generate_status_line():
    """Generate status line output for Claude Code."""

    # Read data from Claude Code via stdin
    stdin_data = read_stdin_data()

    # Get current project name from working directory
    try:
        project_path = os.getcwd()
        project_name = Path(project_path).name
    except:
        project_name = 'unknown'

    # Initialize git info
    git_info = GitInfo(cache_duration=5)

    # Format parts
    parts = []

    # Project name
    parts.append(f"📁 {project_name}")

    # Add git information
    git_status = git_info.get_git_status(project_path)
    git_display = git_info.format_git_info(git_status)
    if git_display:
        parts.append(git_display)

    # Current model - use stdin data if available (most accurate)
    current_model = "Sonnet 4"  # Default
    if stdin_data.get('model', {}).get('display_name'):
        model_name = stdin_data['model']['display_name']
        model_id = stdin_data['model'].get('id', '')
        if 'Opus' in model_name:
            current_model = "Opus 4.5" if "4.5" in model_name or "4-5" in model_id else "Opus 4"
        elif 'Sonnet' in model_name:
            current_model = "Sonnet 4"
    else:
        # Fallback to environment variable
        claude_model = os.environ.get('CLAUDE_MODEL', '').lower()
        if 'opus' in claude_model:
            current_model = "Opus 4"

    parts.append(f"🤖 {current_model}")

    # Context window usage - the key metric
    ctx_window = stdin_data.get('context_window', {})
    if ctx_window:
        ctx_used_pct = ctx_window.get('used_percentage', 0)
        ctx_window_size = ctx_window.get('context_window_size', 200000)

        # Calculate actual tokens in context (includes cached conversation)
        current_usage = ctx_window.get('current_usage', {})
        actual_context_tokens = (
            current_usage.get('cache_read_input_tokens', 0) +
            current_usage.get('cache_creation_input_tokens', 0) +
            current_usage.get('input_tokens', 0) +
            current_usage.get('output_tokens', 0)
        )

        # If we couldn't get current_usage, estimate from percentage
        if actual_context_tokens == 0 and ctx_used_pct > 0:
            actual_context_tokens = int(ctx_window_size * ctx_used_pct / 100)

        tokens_k = actual_context_tokens // 1000
        window_k = ctx_window_size // 1000

        # Color based on usage
        ctx_color = get_context_color(ctx_used_pct)

        # Context display with warning at 65%+
        if ctx_used_pct >= 65:
            parts.append(f"\033[38;2;{ctx_color[0]};{ctx_color[1]};{ctx_color[2]}m⚠️ CTX:{ctx_used_pct}% ({tokens_k}k/{window_k}k)\033[0m")
        else:
            parts.append(f"\033[38;2;{ctx_color[0]};{ctx_color[1]};{ctx_color[2]}m📊 CTX:{ctx_used_pct}% ({tokens_k}k/{window_k}k)\033[0m")

    # Output the status line
    status_line = " | ".join(parts)
    print(status_line)

if __name__ == "__main__":
    generate_status_line()