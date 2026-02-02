"""
Database layer for Semantic Scout.

Handles job persistence, duplicate prevention, and application status tracking
using SQLite.
"""

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)


class ApplicationStatus(Enum):
    """Status of job application."""
    NEW = "new"
    REVIEWED = "reviewed"
    APPLIED = "applied"
    INTERVIEWING = "interviewing"
    REJECTED = "rejected"
    OFFER = "offer"
    DECLINED = "declined"
    ACCEPTED = "accepted"


@dataclass
class Job:
    """Represents a job listing."""
    id: Optional[int]
    job_hash: str
    title: str
    company: str
    location: str
    description: str
    url: str
    source: str
    latitude: Optional[float]
    longitude: Optional[float]
    is_remote: bool
    distance_miles: Optional[float]
    matched_city: Optional[str]  # Name of the city stanza that matched, or "Remote" if remote-matched
    semantic_score: Optional[float]
    score_reasoning: Optional[str]
    application_status: ApplicationStatus
    created_at: datetime
    updated_at: datetime
    raw_data: Optional[str]
    processing_complete: bool = False  # True when all LLM-dependent processing has succeeded
    
    @classmethod
    def create_hash(cls, title: str, company: str, url: str) -> str:
        """Create a unique hash for job deduplication."""
        content = f"{title.lower().strip()}|{company.lower().strip()}|{url.strip()}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]


class Database:
    """SQLite database manager for job listings."""
    
    def __init__(self, db_path: str = "jobs.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._init_database()
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_database(self) -> None:
        """Initialize the database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_hash TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    company TEXT NOT NULL,
                    location TEXT NOT NULL,
                    description TEXT,
                    url TEXT NOT NULL,
                    source TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    is_remote BOOLEAN DEFAULT FALSE,
                    distance_miles REAL,
                    matched_city TEXT,
                    semantic_score REAL,
                    score_reasoning TEXT,
                    application_status TEXT DEFAULT 'new',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    raw_data TEXT,
                    processing_complete BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_hash ON jobs(job_hash)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_score ON jobs(semantic_score)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(application_status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)
            """)
            
            # Migrate existing databases: add matched_city column if missing
            cursor.execute("PRAGMA table_info(jobs)")
            columns = [row[1] for row in cursor.fetchall()]
            if "matched_city" not in columns:
                cursor.execute("ALTER TABLE jobs ADD COLUMN matched_city TEXT")
                logger.info("Migrated database: added matched_city column")
            
            # Migrate existing databases: add processing_complete column if missing
            if "processing_complete" not in columns:
                cursor.execute("ALTER TABLE jobs ADD COLUMN processing_complete BOOLEAN DEFAULT FALSE")
                # Mark already-scored jobs as complete (they succeeded before this feature)
                cursor.execute("UPDATE jobs SET processing_complete = TRUE WHERE semantic_score IS NOT NULL")
                logger.info("Migrated database: added processing_complete column")
            
            # Create search history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    keyword TEXT NOT NULL,
                    source TEXT NOT NULL,
                    jobs_found INTEGER DEFAULT 0,
                    jobs_new INTEGER DEFAULT 0,
                    searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create scoring history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scoring_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    score REAL NOT NULL,
                    reasoning TEXT,
                    model TEXT NOT NULL,
                    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES jobs(id)
                )
            """)
            
            conn.commit()
    
    def job_exists(self, job_hash: str) -> bool:
        """
        Check if a job already exists in the database.
        
        Args:
            job_hash: The unique hash of the job.
        
        Returns:
            True if the job exists, False otherwise.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM jobs WHERE job_hash = ?", (job_hash,))
            return cursor.fetchone() is not None
    
    def insert_job(self, job: Job) -> Optional[int]:
        """
        Insert a new job into the database.
        
        Args:
            job: The job to insert.
        
        Returns:
            The ID of the inserted job, or None if it already exists.
        """
        if self.job_exists(job.job_hash):
            return None
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO jobs (
                    job_hash, title, company, location, description, url, source,
                    latitude, longitude, is_remote, distance_miles, matched_city,
                    semantic_score, score_reasoning, application_status, raw_data,
                    processing_complete
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_hash, job.title, job.company, job.location, job.description,
                job.url, job.source, job.latitude, job.longitude, job.is_remote,
                job.distance_miles, job.matched_city, job.semantic_score, job.score_reasoning,
                job.application_status.value, job.raw_data, job.processing_complete
            ))
            conn.commit()
            return cursor.lastrowid
    
    def update_job_description(self, job_id: int, description: str) -> None:
        """
        Update a job's description.
        
        Used when fetching full descriptions after initial fast collection.
        
        Args:
            job_id: The ID of the job to update.
            description: The full job description.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE jobs 
                SET description = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (description, job_id))
            conn.commit()
    
    def update_job_geo_info(
        self,
        job_id: int,
        is_remote: bool,
        distance_miles: Optional[float],
        matched_city: Optional[str]
    ) -> None:
        """
        Update a job's geographic information after geo-filtering.
        
        Args:
            job_id: The ID of the job to update.
            is_remote: Whether the job is remote.
            distance_miles: Distance from matched city (or None if remote).
            matched_city: Name of the city stanza that matched, or "Remote".
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE jobs 
                SET is_remote = ?, distance_miles = ?, matched_city = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (is_remote, distance_miles, matched_city, job_id))
            conn.commit()
    
    def update_job_score(
        self, 
        job_id: int, 
        score: float, 
        reasoning: str, 
        model: str
    ) -> None:
        """
        Update a job's semantic score.
        
        Args:
            job_id: The ID of the job to update.
            score: The semantic score (1-10).
            reasoning: The LLM's reasoning for the score.
            model: The LLM model used for scoring.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Update the job
            cursor.execute("""
                UPDATE jobs 
                SET semantic_score = ?, score_reasoning = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (score, reasoning, job_id))
            
            # Record in scoring history
            cursor.execute("""
                INSERT INTO scoring_history (job_id, score, reasoning, model)
                VALUES (?, ?, ?, ?)
            """, (job_id, score, reasoning, model))
            
            conn.commit()
    
    def mark_processing_complete(self, job_id: int) -> None:
        """
        Mark a job as having completed all LLM-dependent processing.
        
        This should only be called after successful scoring.
        Jobs that fail LLM processing remain incomplete for retry.
        
        Args:
            job_id: The ID of the job to mark complete.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE jobs 
                SET processing_complete = TRUE, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (job_id,))
            conn.commit()
    
    def get_incomplete_jobs(self) -> list[Job]:
        """
        Get all jobs where LLM-dependent processing has not completed.
        
        This includes jobs that:
        - Have not been scored yet
        - Failed during LLM geocoding fallback
        - Failed during scoring
        
        Returns:
            List of incomplete Job objects.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM jobs 
                WHERE processing_complete = FALSE OR processing_complete IS NULL
                ORDER BY created_at DESC
            """)
            return [self._row_to_job(row) for row in cursor.fetchall()]
    
    def update_application_status(
        self, 
        job_id: int, 
        status: ApplicationStatus
    ) -> None:
        """
        Update a job's application status.
        
        Args:
            job_id: The ID of the job to update.
            status: The new application status.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE jobs 
                SET application_status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status.value, job_id))
            conn.commit()
    
    def get_job_by_id(self, job_id: int) -> Optional[Job]:
        """
        Get a job by its ID.
        
        Args:
            job_id: The ID of the job.
        
        Returns:
            The Job object, or None if not found.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            return self._row_to_job(row) if row else None
    
    def get_jobs_by_score_threshold(
        self, 
        min_score: float,
        since: Optional[datetime] = None
    ) -> list[Job]:
        """
        Get jobs with semantic scores at or above the threshold.
        
        Args:
            min_score: Minimum semantic score.
            since: Only include jobs created after this datetime.
        
        Returns:
            List of matching Job objects.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if since:
                cursor.execute("""
                    SELECT * FROM jobs 
                    WHERE semantic_score >= ? AND created_at >= ?
                    ORDER BY semantic_score DESC
                """, (min_score, since))
            else:
                cursor.execute("""
                    SELECT * FROM jobs 
                    WHERE semantic_score >= ?
                    ORDER BY semantic_score DESC
                """, (min_score,))
            
            return [self._row_to_job(row) for row in cursor.fetchall()]
    
    def get_unscored_jobs(self) -> list[Job]:
        """
        Get all jobs that haven't been scored yet.
        
        Returns:
            List of unscored Job objects.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM jobs 
                WHERE semantic_score IS NULL
                ORDER BY created_at DESC
            """)
            return [self._row_to_job(row) for row in cursor.fetchall()]
    
    def get_all_jobs(
        self, 
        status: Optional[ApplicationStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[Job]:
        """
        Get all jobs with optional filtering.
        
        Args:
            status: Filter by application status.
            limit: Maximum number of jobs to return.
            offset: Number of jobs to skip.
        
        Returns:
            List of Job objects.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute("""
                    SELECT * FROM jobs 
                    WHERE application_status = ?
                    ORDER BY semantic_score DESC NULLS LAST, created_at DESC
                    LIMIT ? OFFSET ?
                """, (status.value, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM jobs 
                    ORDER BY semantic_score DESC NULLS LAST, created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            
            return [self._row_to_job(row) for row in cursor.fetchall()]
    
    def record_search(
        self, 
        keyword: str, 
        source: str, 
        jobs_found: int, 
        jobs_new: int
    ) -> None:
        """
        Record a search operation in history.
        
        Args:
            keyword: The search keyword used.
            source: The job board source.
            jobs_found: Total jobs found.
            jobs_new: Number of new (not duplicate) jobs.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO search_history (keyword, source, jobs_found, jobs_new)
                VALUES (?, ?, ?, ?)
            """, (keyword, source, jobs_found, jobs_new))
            conn.commit()
    
    def get_stats(self) -> dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with various statistics.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total jobs
            cursor.execute("SELECT COUNT(*) FROM jobs")
            stats["total_jobs"] = cursor.fetchone()[0]
            
            # Jobs by status
            cursor.execute("""
                SELECT application_status, COUNT(*) 
                FROM jobs 
                GROUP BY application_status
            """)
            stats["by_status"] = dict(cursor.fetchall())
            
            # Scored jobs
            cursor.execute("SELECT COUNT(*) FROM jobs WHERE semantic_score IS NOT NULL")
            stats["scored_jobs"] = cursor.fetchone()[0]
            
            # Average score
            cursor.execute("SELECT AVG(semantic_score) FROM jobs WHERE semantic_score IS NOT NULL")
            avg_score = cursor.fetchone()[0]
            stats["average_score"] = round(avg_score, 2) if avg_score else None
            
            # High-scoring jobs (>= 7)
            cursor.execute("SELECT COUNT(*) FROM jobs WHERE semantic_score >= 7")
            stats["high_scoring_jobs"] = cursor.fetchone()[0]
            
            # Incomplete jobs (pending retry)
            cursor.execute("SELECT COUNT(*) FROM jobs WHERE processing_complete = FALSE OR processing_complete IS NULL")
            stats["incomplete_jobs"] = cursor.fetchone()[0]
            
            # Jobs by source
            cursor.execute("""
                SELECT source, COUNT(*) 
                FROM jobs 
                GROUP BY source
            """)
            stats["by_source"] = dict(cursor.fetchall())
            
            return stats
    
    def _row_to_job(self, row: sqlite3.Row) -> Job:
        """Convert a database row to a Job object."""
        return Job(
            id=row["id"],
            job_hash=row["job_hash"],
            title=row["title"],
            company=row["company"],
            location=row["location"],
            description=row["description"],
            url=row["url"],
            source=row["source"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            is_remote=bool(row["is_remote"]),
            distance_miles=row["distance_miles"],
            matched_city=row["matched_city"] if "matched_city" in row.keys() else None,
            semantic_score=row["semantic_score"],
            score_reasoning=row["score_reasoning"],
            application_status=ApplicationStatus(row["application_status"]),
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.now(),
            raw_data=row["raw_data"],
            processing_complete=bool(row["processing_complete"]) if "processing_complete" in row.keys() else False,
        )
