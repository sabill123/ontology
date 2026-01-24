"""
Logging utilities for the Ontology Platform
Based on multiagent/logger.py

Enhanced with:
- Session-specific log files
- Structured event logging
- Conversation tracking
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler
from rich.theme import Theme
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Custom theme for the console
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "error": "bold red",
    "success": "bold green",
    "thought": "italic cyan",
    "code": "bold yellow",
    "result": "white",
    # Agent colors
    "data_analyst": "blue",
    "semantic_architect": "green",
    "link_resolver": "magenta",
    # Status colors
    "pending": "yellow",
    "approved": "bold green",
    "rejected": "bold red",
})

console = Console(theme=custom_theme)


class SessionLogger:
    """
    Enhanced logger with session-specific file logging.
    Maintains separate log files for each session.
    """

    def __init__(self, base_log_dir: str = "./data/logs"):
        self.base_log_dir = Path(base_log_dir)
        self.conversations_dir = self.base_log_dir / "conversations"
        self.events_dir = self.base_log_dir / "events"
        self._ensure_directories()

        self.current_session_id: Optional[str] = None
        self.session_logger: Optional[logging.Logger] = None
        self.event_file: Optional[Path] = None

        # Setup global logger
        self._setup_global_logger()

    def _ensure_directories(self):
        """Create log directories if they don't exist."""
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)

    def _setup_global_logger(self):
        """Setup global file logger."""
        global_log_file = self.base_log_dir / "ontology.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(global_log_file),
            ]
        )
        self.global_logger = logging.getLogger("ontology")
        self.global_logger.setLevel(logging.DEBUG)

    def start_session(self, session_id: str) -> None:
        """
        Start logging for a new session.

        Args:
            session_id: Session identifier
        """
        self.current_session_id = session_id

        # Create session-specific logger
        session_log_file = self.conversations_dir / f"{session_id}.log"
        self.session_logger = logging.getLogger(f"ontology.session.{session_id}")
        self.session_logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        self.session_logger.handlers = []

        # Add file handler for session
        file_handler = logging.FileHandler(session_log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        self.session_logger.addHandler(file_handler)

        # Initialize event file
        self.event_file = self.events_dir / f"{session_id}_events.jsonl"

        self.info(f"Session started: {session_id}")
        self.log_event("session_start", {"session_id": session_id})

    def end_session(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """
        End the current session.

        Args:
            summary: Optional session summary
        """
        if self.current_session_id:
            self.log_event("session_end", {
                "session_id": self.current_session_id,
                "summary": summary or {}
            })
            self.info(f"Session ended: {self.current_session_id}")
            self.current_session_id = None
            self.session_logger = None
            self.event_file = None

    def info(self, message: str) -> None:
        """Log info message to both global and session logs."""
        self.global_logger.info(message)
        if self.session_logger:
            self.session_logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.global_logger.debug(message)
        if self.session_logger:
            self.session_logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.global_logger.warning(message)
        if self.session_logger:
            self.session_logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.global_logger.error(message)
        if self.session_logger:
            self.session_logger.error(message)

    def log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        level: str = "info"
    ) -> None:
        """
        Log a structured event for traceability.

        Args:
            event_type: Type of event (e.g., "concept_approved", "agent_vote")
            data: Event data
            level: Log level
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.current_session_id,
            "event_type": event_type,
            "level": level,
            "data": data,
        }

        # Write to event file (JSONL format)
        if self.event_file:
            with open(self.event_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

        # Also log to regular logger
        msg = f"[{event_type}] {json.dumps(data, ensure_ascii=False, default=str)[:200]}"
        getattr(self.global_logger, level)(msg)

    def log_agent_message(
        self,
        agent_name: str,
        message: str,
        vote: Optional[str] = None,
        concept_id: Optional[str] = None,
    ) -> None:
        """Log an agent message/vote."""
        self.log_event("agent_message", {
            "agent": agent_name,
            "message": message[:500],
            "vote": vote,
            "concept_id": concept_id,
        })

    def log_concept_status(
        self,
        concept_id: str,
        concept_name: str,
        status: str,
        confidence: Optional[float] = None,
    ) -> None:
        """Log a concept status change."""
        self.log_event("concept_status", {
            "concept_id": concept_id,
            "concept_name": concept_name,
            "status": status,
            "confidence": confidence,
        })

    def log_data_source(
        self,
        source_id: str,
        source_name: str,
        tables_count: int,
    ) -> None:
        """Log data source registration."""
        self.log_event("data_source_registered", {
            "source_id": source_id,
            "source_name": source_name,
            "tables_count": tables_count,
        })

    def get_session_log_path(self, session_id: str) -> Path:
        """Get the log file path for a session."""
        return self.conversations_dir / f"{session_id}.log"

    def get_session_events_path(self, session_id: str) -> Path:
        """Get the events file path for a session."""
        return self.events_dir / f"{session_id}_events.jsonl"

    def list_session_logs(self) -> list:
        """List all session log files."""
        return sorted(
            [f.stem for f in self.conversations_dir.glob("*.log")],
            reverse=True
        )


# Global session logger instance
session_logger = SessionLogger()


def setup_logging(log_file: str = "ontology.log"):
    """Legacy setup_logging for backwards compatibility."""
    return session_logger.global_logger


# Global file logger instance (backwards compatible)
logger = session_logger.global_logger


def log_step(step_name: str, status: str = "INFO"):
    """Logs a step to the file."""
    logger.info(f"[{step_name}] {status}")


def print_panel(content: str, title: str, style: str = "info"):
    """Prints a rich panel to the console."""
    console.print(Panel(content, title=title, border_style=style, expand=False))


def print_status(message: str, style: str = "info"):
    """Prints a status message."""
    console.print(f"[{style}]{message}[/{style}]")


def print_agent_message(agent_name: str, content: str, vote: str = None):
    """Prints an agent message with appropriate styling."""
    agent_styles = {
        "Data Analyst": "data_analyst",
        "Semantic Architect": "semantic_architect",
        "Link Resolver": "link_resolver",
    }
    style = agent_styles.get(agent_name, "info")
    title = f"{agent_name}"
    if vote:
        title += f" [{vote}]"
    print_panel(content, title, style)


def print_concept_table(concepts: list):
    """Prints a table of concepts."""
    table = Table(title="Concepts in Mempool")
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Priority", style="blue")

    for concept in concepts:
        table.add_row(
            concept.get("id", ""),
            concept.get("type", ""),
            concept.get("name", ""),
            concept.get("status", ""),
            str(concept.get("priority", 0))
        )

    console.print(table)


def print_chain_table(blocks: list):
    """Prints a table of ontology blocks."""
    table = Table(title="Ontology Chain")
    table.add_column("Block", style="cyan")
    table.add_column("Concept", style="green")
    table.add_column("Type", style="magenta")
    table.add_column("Votes", style="yellow")
    table.add_column("Hash", style="dim")

    for block in blocks:
        votes_str = ", ".join([
            f"{v.get('agent', '?')[:1]}:{v.get('vote', '?')[:1]}"
            for v in block.get("votes", [])
        ])
        table.add_row(
            str(block.get("id", "")),
            block.get("concept", {}).get("name", ""),
            block.get("concept", {}).get("type", ""),
            votes_str,
            block.get("hash", "")[:12] + "..."
        )

    console.print(table)


def create_progress():
    """Creates a progress bar for long operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    )
