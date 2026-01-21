#!/usr/bin/env python3
"""
Main tracker module for Claude Code usage tracking.
Optimized with numpy for fast processing of conversation data.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, asdict
import time

@dataclass
class SessionData:
    """Data class for session information."""
    session_id: str
    start_time: float
    end_time: float
    duration_hours: float
    prompt_count: int
    sonnet_responses: int
    opus_responses: int
    project: str

@dataclass
class ContextData:
    """Context window usage data."""
    estimated_tokens: int
    context_limit: int
    percentage: float
    warning_threshold: float = 45.0

@dataclass
class UsageData:
    """Complete usage data structure."""
    current_5h_prompts: int
    current_5h_start: float
    weekly_sonnet_hours: float
    weekly_opus_hours: float
    weekly_prompts: int
    weekly_start: float
    last_updated: float
    sessions: List[SessionData]
    context: Optional[ContextData] = None

class UsageTracker:
    """Main usage tracker with optimized performance."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the tracker with configuration."""
        self.home = Path.home()
        self.claude_projects = self.home / ".claude" / "projects"
        self.config_path = config_path or Path(__file__).parent.parent / "config"
        self.data_path = Path(__file__).parent.parent / "data"
        
        # Ensure data directory exists
        self.data_path.mkdir(exist_ok=True)
        
        # Time constants
        self.week_start = self._get_week_start()
        self.cycle_5h_start = self._get_5h_cycle_start()
        
        # Cache for parsed data
        self._cache = {}
        self._cache_time = 0
        self.cache_duration = 5  # Cache for 5 seconds
    
    def _get_week_start(self) -> float:
        """Get Monday midnight timestamp in seconds."""
        now = datetime.now()
        days_since_monday = now.weekday()
        monday = now - timedelta(days=days_since_monday)
        monday_midnight = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        return monday_midnight.timestamp()
    
    def _get_5h_cycle_start(self) -> float:
        """Get current 5-hour cycle start in seconds."""
        now = time.time()
        hours_since_epoch = now / 3600
        cycle_number = int(hours_since_epoch / 5)
        return cycle_number * 5 * 3600

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text (~4 chars per token)."""
        if not text:
            return 0
        return len(text) // 4

    def estimate_context_usage(self, project_path: Optional[str] = None) -> Optional[ContextData]:
        """
        Estimate context window usage for the current session.
        Uses ~4 chars per token heuristic against 200k context limit.
        """
        CONTEXT_LIMIT = 200000  # Opus 4.5 context window
        WARNING_THRESHOLD = 45.0

        # Find current session file
        if project_path is None:
            project_path = os.getcwd()

        # Claude stores sessions in ~/.claude/projects/<encoded-path>/
        # The folder name is the path with slashes replaced by dashes
        encoded_path = project_path.replace('/', '-')
        project_dir = self.claude_projects / encoded_path

        # Fallback: find most recently modified project if exact match not found
        if not project_dir.exists() and self.claude_projects.exists():
            # Find all project dirs with recent activity
            recent_dirs = []
            for pdir in self.claude_projects.iterdir():
                if pdir.is_dir() and not pdir.name.startswith('.'):
                    jsonl_files = list(pdir.glob('*.jsonl'))
                    if jsonl_files:
                        most_recent = max(jsonl_files, key=lambda f: f.stat().st_mtime)
                        mtime = most_recent.stat().st_mtime
                        # Only consider files modified in last 5 minutes (active session)
                        if (time.time() - mtime) < 300:
                            recent_dirs.append((pdir, mtime))

            if recent_dirs:
                # Use the most recently modified project
                project_dir = max(recent_dirs, key=lambda x: x[1])[0]

        if not project_dir or not project_dir.exists():
            return None

        # Find most recent session file
        jsonl_files = list(project_dir.glob('*.jsonl'))
        if not jsonl_files:
            return None

        current_session = max(jsonl_files, key=lambda f: f.stat().st_mtime)

        # Estimate tokens from session file
        # Use file size as proxy, with ~6 chars per token (accounting for JSON overhead)
        # This gives a reasonable estimate of actual context window usage
        try:
            file_size = current_session.stat().st_size

            # Also count actual content for more accuracy
            total_content_chars = 0
            with open(current_session, 'r') as f:
                for line in f:
                    try:
                        msg = json.loads(line)
                        # Skip non-message entries
                        if msg.get('type') not in ('user', 'assistant'):
                            continue

                        content = msg.get('message', {}).get('content', '')

                        if isinstance(content, str):
                            total_content_chars += len(content)
                        elif isinstance(content, list):
                            # Handle array content (tool_use, tool_result, text blocks)
                            for item in content:
                                if isinstance(item, dict):
                                    # Count text content
                                    if text := item.get('text', ''):
                                        total_content_chars += len(text)
                                    # Count tool input/output
                                    if inp := item.get('input'):
                                        total_content_chars += len(json.dumps(inp))
                                    if content_blocks := item.get('content'):
                                        if isinstance(content_blocks, list):
                                            for cb in content_blocks:
                                                if isinstance(cb, dict) and (t := cb.get('text')):
                                                    total_content_chars += len(t)
                    except:
                        continue

            # Use the larger of: file-based estimate or content-based estimate
            # File-based: ~6 chars/token (accounts for JSON overhead)
            # Content-based: ~4 chars/token (raw text)
            file_based_tokens = file_size // 6
            content_based_tokens = total_content_chars // 4
            estimated_tokens = max(file_based_tokens, content_based_tokens)
        except:
            return None
        percentage = (estimated_tokens / CONTEXT_LIMIT) * 100

        return ContextData(
            estimated_tokens=estimated_tokens,
            context_limit=CONTEXT_LIMIT,
            percentage=round(percentage, 1),
            warning_threshold=WARNING_THRESHOLD
        )
    
    def _parse_timestamp(self, ts: str) -> Optional[float]:
        """Parse ISO timestamp to epoch seconds efficiently."""
        if not ts or ts == 'null':
            return None
        try:
            # Remove milliseconds for faster parsing
            clean_ts = ts.split('.')[0] + 'Z' if '.' in ts else ts
            dt = datetime.fromisoformat(clean_ts.replace('Z', '+00:00'))
            return dt.timestamp()
        except:
            return None
    
    def _is_command_message(self, content) -> bool:
        """Check if message is a local command."""
        if isinstance(content, str):
            return '<command-name>' in content or '<local-command-stdout>' in content
        elif isinstance(content, list):
            # Handle array content like [{"type":"text","text":"..."}]
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text = item.get('text', '')
                    if '<command-name>' in text or '<local-command-stdout>' in text:
                        return True
        return False
    
    def _analyze_jsonl_file(self, jsonl_path: Path) -> SessionData:
        """Analyze a single JSONL file (session) with caching."""
        # Check cache
        cache_key = str(jsonl_path)
        if cache_key in self._cache and (time.time() - self._cache_time) < self.cache_duration:
            return self._cache[cache_key]
        
        timestamps = []
        prompts = 0
        sonnet_responses = 0
        opus_responses = 0
        
        try:
            with open(jsonl_path, 'r') as f:
                for line in f:
                    try:
                        msg = json.loads(line)
                        
                        # Collect timestamp
                        if ts := msg.get('timestamp'):
                            if epoch := self._parse_timestamp(ts):
                                timestamps.append(epoch)
                        
                        # Count user prompts (excluding commands and meta messages)
                        if (msg.get('type') == 'user' and 
                            msg.get('message', {}).get('role') == 'user' and
                            not msg.get('isMeta', False) and
                            msg.get('userType') == 'external'):  # Only external user messages
                            
                            content = msg.get('message', {}).get('content', '')
                            # Skip empty content and command messages
                            if content and not self._is_command_message(content):
                                prompts += 1
                        
                        # Count model responses
                        elif msg.get('type') == 'assistant':
                            model = msg.get('message', {}).get('model', '').lower()
                            if 'opus' in model:
                                opus_responses += 1
                            elif 'sonnet' in model:
                                sonnet_responses += 1
                    except:
                        continue
        except:
            pass
        
        # Calculate session duration
        duration_hours = 0.0
        start_time = 0.0
        end_time = 0.0
        
        if timestamps:
            timestamps = np.array(timestamps)
            start_time = float(timestamps.min())
            end_time = float(timestamps.max())
            duration_hours = (end_time - start_time) / 3600
        
        session = SessionData(
            session_id=jsonl_path.stem,
            start_time=start_time,
            end_time=end_time,
            duration_hours=duration_hours,
            prompt_count=prompts,
            sonnet_responses=sonnet_responses,
            opus_responses=opus_responses,
            project=jsonl_path.parent.name
        )
        
        # Update cache
        self._cache[cache_key] = session
        self._cache_time = time.time()
        
        return session
    
    def get_all_sessions(self) -> List[SessionData]:
        """Get all sessions across all projects."""
        sessions = []
        
        if not self.claude_projects.exists():
            return sessions
        
        # Process all projects in parallel-friendly way
        for project_dir in self.claude_projects.iterdir():
            if project_dir.is_dir():
                for jsonl_file in project_dir.glob('*.jsonl'):
                    session = self._analyze_jsonl_file(jsonl_file)
                    if session.duration_hours > 0:  # Only include real sessions
                        sessions.append(session)
        
        return sessions
    
    def calculate_usage(self) -> UsageData:
        """Calculate complete usage statistics."""
        sessions = self.get_all_sessions()
        
        # Filter sessions by time
        week_sessions = [s for s in sessions if s.start_time >= self.week_start]
        cycle_sessions = [s for s in sessions if s.start_time >= self.cycle_5h_start]
        
        # Calculate 5-hour cycle stats
        cycle_prompts = sum(s.prompt_count for s in cycle_sessions)
        
        # Calculate weekly stats using numpy for efficiency
        weekly_prompts = sum(s.prompt_count for s in week_sessions)
        
        # Calculate model-specific hours
        sonnet_hours = 0.0
        opus_hours = 0.0
        
        for session in week_sessions:
            total_responses = session.sonnet_responses + session.opus_responses
            if total_responses > 0:
                sonnet_ratio = session.sonnet_responses / total_responses
                opus_ratio = session.opus_responses / total_responses
                sonnet_hours += session.duration_hours * sonnet_ratio
                opus_hours += session.duration_hours * opus_ratio
        
        # Get context usage for current session
        context_data = self.estimate_context_usage()

        return UsageData(
            current_5h_prompts=cycle_prompts,
            current_5h_start=self.cycle_5h_start,
            weekly_sonnet_hours=round(sonnet_hours, 2),
            weekly_opus_hours=round(opus_hours, 2),
            weekly_prompts=weekly_prompts,
            weekly_start=self.week_start,
            last_updated=time.time(),
            sessions=week_sessions,
            context=context_data
        )
    
    def save_usage_data(self, usage_data: UsageData):
        """Save usage data to JSON file."""
        data = {
            "current_5h_cycle": {
                "start_time": int(usage_data.current_5h_start * 1000),
                "total_prompts": usage_data.current_5h_prompts,
                "total_hours": round(usage_data.current_5h_prompts / 10, 2)  # Legacy field
            },
            "current_week": {
                "start_time": int(usage_data.weekly_start * 1000),
                "sonnet4_hours": usage_data.weekly_sonnet_hours,
                "opus4_hours": usage_data.weekly_opus_hours,
                "total_sessions": len(usage_data.sessions)
            },
            "last_updated": int(usage_data.last_updated * 1000)
        }
        
        with open(self.data_path / "usage_data.json", 'w') as f:
            json.dump(data, f, indent=2)
    
    def update(self) -> UsageData:
        """Update and return current usage data."""
        usage_data = self.calculate_usage()
        self.save_usage_data(usage_data)
        return usage_data