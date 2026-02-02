#!/usr/bin/env python3
"""
Semantic Scout - Local AI Job Intelligence Engine

Main entry point and pipeline orchestrator. This module coordinates the
job collection, geographic filtering, semantic scoring, and reporting workflow.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from semanticscout.config import Config, load_config, generate_example_config
from semanticscout.database import Database, Job, ApplicationStatus
from semanticscout.collector import CollectorManager, RawJobListing, raw_to_job
from semanticscout.geofence import Geofencer
from semanticscout.scorer import ScorerManager
from semanticscout.reporting import ReportGenerator
from semanticscout.deduplicator import JobDeduplicator


# Configure logging
def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Reduce noise from external libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('geopy').setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


class SemanticScoutPipeline:
    """
    Main pipeline orchestrator for Semantic Scout.
    
    Coordinates the job collection, filtering, scoring, and reporting workflow.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the pipeline.
        
        Args:
            config: Application configuration.
        """
        self.config = config
        
        # Initialize components
        self.db = Database(config.database.db_path)
        self.geofencer = Geofencer(config.location, config.llm)
        self.collector_manager = CollectorManager(config.scraper)
        self.scorer_manager = ScorerManager(config.llm, config.search)
        self.reporter = ReportGenerator(config.reporting, self.db)
        
        # Initialize deduplicator if enabled
        self.deduplicator = None
        if config.deduplication.enabled:
            self.deduplicator = JobDeduplicator(
                similarity_threshold=config.deduplication.similarity_threshold
            )
        
        logger.info("Semantic Scout pipeline initialized")
    
    async def run_full_pipeline(self) -> dict:
        """
        Run the complete job search and scoring pipeline.
        
        Pipeline flow:
        1. Fast collect (titles, companies, locations, URLs only)
        2. Deduplicate and store in DB
        3. Apply geographic filter
        4. Fetch full descriptions ONLY for geo-filtered jobs
        5. Score with full context
        6. Generate reports
        
        Returns:
            Dictionary with pipeline statistics.
        """
        stats = {
            "start_time": datetime.now(),
            "jobs_collected": 0,
            "jobs_new": 0,
            "jobs_passed_geofence": 0,
            "descriptions_fetched": 0,
            "jobs_scored": 0,
            "high_scoring_jobs": 0,
            "errors": [],
        }
        
        try:
            # Step 1: Fast collect jobs from all sources (no full descriptions)
            logger.info("=" * 60)
            logger.info("STEP 1: Fast collecting jobs from configured sources")
            logger.info("=" * 60)
            
            await self.collector_manager.initialize()
            
            raw_jobs = await self.collector_manager.search_all(
                keywords=self.config.search.keywords,
                job_boards=self.config.search.job_boards,
                max_results_per_keyword=self.config.search.max_results_per_keyword,
                parallel=self.config.scraper.parallel_boards
            )
            
            stats["jobs_collected"] = len(raw_jobs)
            logger.info(f"Collected {len(raw_jobs)} total job listings (fast mode)")
            
            # Step 2: Deduplicate and convert to Job objects
            logger.info("=" * 60)
            logger.info("STEP 2: Deduplicating and storing new jobs")
            logger.info("=" * 60)
            
            new_jobs = []
            duplicate_count = 0
            cross_source_dupe_count = 0
            
            # Load existing jobs once for deduplication
            # Tuple format: (id, title, company, location, description, url)
            existing_jobs_list = self.db.get_all_jobs(limit=10000)
            existing_tuples = [
                (j.id, j.title, j.company, j.location, j.description or "", j.url)
                for j in existing_jobs_list if j.id
            ]
            
            # Warm up deduplicator cache with existing jobs if enabled
            if self.deduplicator and existing_tuples:
                logger.info(f"Pre-loading {len(existing_tuples)} existing jobs into deduplicator cache")
                self.deduplicator.batch_compute_embeddings([
                    (j[0], j[1], j[2], j[3], j[4])  # id, title, company, location, description
                    for j in existing_tuples
                ])
            
            for raw in raw_jobs:
                job = raw_to_job(raw)
                
                # First: Check exact hash (same URL from same source)
                if self.db.job_exists(job.job_hash):
                    duplicate_count += 1
                    continue
                
                # Second: Check embedding similarity (cross-source duplicates)
                if self.deduplicator and existing_tuples:
                    dup_result = self.deduplicator.check_duplicate(
                        job.title, job.company, job.location, job.description or "", job.url,
                        existing_tuples
                    )
                    
                    if dup_result.is_duplicate:
                        cross_source_dupe_count += 1
                        # Log is handled inside check_duplicate()
                        continue
                
                # Not a duplicate - enrich and store
                job = await self.geofencer.enrich_job_location(job)
                job_id = self.db.insert_job(job)
                
                if job_id:
                    job.id = job_id
                    new_jobs.append(job)
                    
                    # Add new job to existing_tuples for dedup against remaining raw jobs
                    existing_tuples.append(
                        (job_id, job.title, job.company, job.location, job.description or "", job.url)
                    )
            
            stats["jobs_new"] = len(new_jobs)
            stats["jobs_duplicate_exact"] = duplicate_count
            stats["jobs_duplicate_cross_source"] = cross_source_dupe_count
            logger.info(
                f"Found {len(new_jobs)} new unique jobs "
                f"(skipped: {duplicate_count} exact duplicates, "
                f"{cross_source_dupe_count} cross-source duplicates)"
            )
            
            # Step 3: Apply geographic filter
            logger.info("=" * 60)
            logger.info("STEP 3: Applying geographic filter")
            logger.info("=" * 60)
            
            # Get all incomplete jobs (not just new ones, includes previously failed)
            incomplete_jobs = self.db.get_incomplete_jobs()
            
            # Log breakdown of incomplete jobs
            previously_failed = [j for j in incomplete_jobs if j.id not in [nj.id for nj in new_jobs]]
            if previously_failed:
                logger.info(
                    f"Found {len(incomplete_jobs)} incomplete jobs to process "
                    f"({len(new_jobs)} new, {len(previously_failed)} retrying from previous failures)"
                )
            else:
                logger.info(f"Found {len(incomplete_jobs)} incomplete jobs to process")
            
            jobs_to_score = []
            for job in incomplete_jobs:
                result = await self.geofencer.filter_job(job)
                
                if result.passes:
                    job.is_remote = result.is_remote
                    job.distance_miles = result.distance_miles
                    job.matched_city = result.matched_city
                    
                    # Update geo info in database
                    if job.id:
                        self.db.update_job_geo_info(
                            job.id,
                            result.is_remote,
                            result.distance_miles,
                            result.matched_city
                        )
                    
                    jobs_to_score.append(job)
            
            stats["jobs_passed_geofence"] = len(jobs_to_score)
            logger.info(f"{len(jobs_to_score)} jobs passed geographic filter")
            
            # Step 4: Fetch full descriptions for filtered jobs
            if jobs_to_score:
                logger.info("=" * 60)
                logger.info("STEP 4: Fetching full descriptions for filtered jobs")
                logger.info("=" * 60)
                
                descriptions = await self.collector_manager.fetch_descriptions(jobs_to_score)
                stats["descriptions_fetched"] = len(descriptions)
                
                # Update jobs with full descriptions
                for job in jobs_to_score:
                    if job.id in descriptions:
                        job.description = descriptions[job.id]
                        # Update in database too
                        self.db.update_job_description(job.id, descriptions[job.id])
                
                logger.info(f"Updated {len(descriptions)} jobs with full descriptions")
            
            # Step 5: Score jobs with LLM
            if jobs_to_score:
                logger.info("=" * 60)
                logger.info("STEP 5: Scoring jobs with local LLM")
                logger.info("=" * 60)
                
                if await self.scorer_manager.initialize():
                    results = await self.scorer_manager.score_jobs(
                        jobs_to_score,
                        max_concurrent=2
                    )
                    
                    for job, result in results:
                        if result.success and job.id:
                            self.db.update_job_score(
                                job.id,
                                result.score,
                                result.reasoning,
                                self.config.llm.model
                            )
                            # Mark job as fully processed after successful scoring
                            self.db.mark_processing_complete(job.id)
                            stats["jobs_scored"] += 1
                            
                            if result.score >= self.config.reporting.min_score_threshold:
                                stats["high_scoring_jobs"] += 1
                        elif not result.success:
                            # Job remains with processing_complete=False for retry next run
                            stats["errors"].append(
                                f"Failed to score '{job.title}': {result.error}"
                            )
                else:
                    stats["errors"].append("Could not initialize LLM scorer - jobs will be retried next run")
            
            # Step 6: Generate reports
            logger.info("=" * 60)
            logger.info("STEP 6: Generating reports")
            logger.info("=" * 60)
            
            self.reporter.generate_daily_report()
            
            # Get high-scoring jobs for console summary
            high_scoring = self.db.get_jobs_by_score_threshold(
                self.config.reporting.min_score_threshold
            )
            self.reporter.print_summary(high_scoring)
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            stats["errors"].append(str(e))
            raise
        
        finally:
            # Cleanup
            await self.collector_manager.close()
            await self.scorer_manager.close()
        
        stats["end_time"] = datetime.now()
        stats["duration"] = (stats["end_time"] - stats["start_time"]).total_seconds()
        
        return stats
    
    async def score_only(self) -> dict:
        """
        Run only the scoring step for unscored or incomplete jobs.
        
        This mode is useful for retrying LLM scoring after failures.
        It processes jobs that either:
        - Have never been scored (semantic_score IS NULL)
        - Have processing_complete = FALSE (failed in a previous run)
        
        Returns:
            Dictionary with scoring statistics.
        """
        stats = {
            "jobs_scored": 0,
            "high_scoring_jobs": 0,
            "errors": [],
        }
        
        try:
            # Get all incomplete jobs (includes unscored and previously failed)
            incomplete_jobs = self.db.get_incomplete_jobs()
            logger.info(f"Found {len(incomplete_jobs)} incomplete jobs to process")
            
            if not incomplete_jobs:
                logger.info("No incomplete jobs to process")
                return stats
            
            # Apply geofence filter
            jobs_to_score = await self.geofencer.filter_jobs(incomplete_jobs)
            
            if not jobs_to_score:
                logger.info("No jobs passed geographic filter")
                return stats
            
            if await self.scorer_manager.initialize():
                results = await self.scorer_manager.score_jobs(jobs_to_score)
                
                for job, result in results:
                    if result.success and job.id:
                        self.db.update_job_score(
                            job.id,
                            result.score,
                            result.reasoning,
                            self.config.llm.model
                        )
                        # Mark job as fully processed after successful scoring
                        self.db.mark_processing_complete(job.id)
                        stats["jobs_scored"] += 1
                        
                        if result.score >= self.config.reporting.min_score_threshold:
                            stats["high_scoring_jobs"] += 1
                    elif not result.success:
                        # Job remains incomplete for retry
                        stats["errors"].append(
                            f"Failed to score '{job.title}': {result.error}"
                        )
            else:
                stats["errors"].append("Could not initialize LLM scorer - jobs will be retried next run")
            
        finally:
            await self.scorer_manager.close()
        
        return stats
    
    def generate_reports(self, report_type: str = "daily") -> str:
        """
        Generate reports without running the full pipeline.
        
        Args:
            report_type: Type of report ("daily", "weekly", "tracker").
        
        Returns:
            Path to the generated report.
        """
        if report_type == "daily":
            self.reporter.generate_daily_report()
        elif report_type == "weekly":
            self.reporter.generate_weekly_summary()
        elif report_type == "tracker":
            self.reporter.generate_application_tracker()
        else:
            raise ValueError(f"Unknown report type: {report_type}")
        
        return str(self.reporter.output_dir)
    
    def print_stats(self) -> None:
        """Print database statistics."""
        stats = self.db.get_stats()
        
        print("\n" + "=" * 40)
        print("SEMANTIC SCOUT DATABASE STATISTICS")
        print("=" * 40)
        print(f"Total jobs tracked: {stats['total_jobs']}")
        print(f"Jobs scored: {stats['scored_jobs']}")
        print(f"High-scoring jobs (>= 7): {stats['high_scoring_jobs']}")
        print(f"Incomplete jobs (pending retry): {stats.get('incomplete_jobs', 0)}")
        
        if stats.get('average_score'):
            print(f"Average score: {stats['average_score']}")
        
        print("\nJobs by status:")
        for status, count in stats.get('by_status', {}).items():
            print(f"  {status}: {count}")
        
        print("\nJobs by source:")
        for source, count in stats.get('by_source', {}).items():
            print(f"  {source}: {count}")
        
        print("=" * 40 + "\n")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic Scout - Local AI Job Intelligence Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    Run the full pipeline
  %(prog)s --fast             Run in fast mode (skip full descriptions)
  %(prog)s --score-only       Score unscored jobs only
  %(prog)s --report daily     Generate a daily report
  %(prog)s --stats            Show database statistics
  %(prog)s --init-config      Generate example config file
        """
    )
    
    parser.add_argument(
        '-c', '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--log-file',
        help='Write logs to file'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode: skip fetching full job descriptions (much faster but less data)'
    )
    
    parser.add_argument(
        '--score-only',
        action='store_true',
        help='Only score unscored jobs (skip collection)'
    )
    
    parser.add_argument(
        '--report',
        choices=['daily', 'weekly', 'tracker'],
        help='Generate a specific report type'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show database statistics'
    )
    
    parser.add_argument(
        '--init-config',
        action='store_true',
        help='Generate example configuration file'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(verbose=args.verbose, log_file=args.log_file)
    
    # Handle init-config separately
    if args.init_config:
        generate_example_config()
        return
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("Run with --init-config to generate an example configuration file")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Apply CLI overrides
    if args.fast:
        config.scraper.fast_mode = True
        logger.info("Fast mode enabled via CLI flag")
    
    # Initialize pipeline
    pipeline = SemanticScoutPipeline(config)
    
    # Handle different modes
    if args.stats:
        pipeline.print_stats()
        return
    
    if args.report:
        output_dir = pipeline.generate_reports(args.report)
        print(f"Report generated in: {output_dir}")
        return
    
    if args.score_only:
        logger.info("Running in score-only mode")
        stats = await pipeline.score_only()
        logger.info(f"Scoring complete: {stats['jobs_scored']} jobs scored")
        return
    
    # Run full pipeline
    logger.info("Starting Semantic Scout full pipeline")
    stats = await pipeline.run_full_pipeline()
    
    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Duration: {stats['duration']:.1f} seconds")
    print(f"Jobs collected: {stats['jobs_collected']}")
    print(f"New unique jobs: {stats['jobs_new']}")
    print(f"Jobs passed geofence: {stats['jobs_passed_geofence']}")
    print(f"Jobs scored: {stats['jobs_scored']}")
    print(f"High-scoring jobs: {stats['high_scoring_jobs']}")
    
    if stats['errors']:
        print(f"\nErrors encountered: {len(stats['errors'])}")
        for error in stats['errors'][:5]:
            print(f"  - {error}")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
