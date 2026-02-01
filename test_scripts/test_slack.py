#!/usr/bin/env python3
"""
Test script for Slack threaded message integration.

This script posts a test summary message with 2 threaded replies
containing sample job data to validate the Slack UX.

Usage:
    python test_slack.py --bot-token "xoxb-your-token" --channel-id "C0123456789"
    
Or set environment variables:
    export SLACK_BOT_TOKEN="xoxb-your-token"
    export SLACK_CHANNEL_ID="C0123456789"
    python test_slack.py
"""

import argparse
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class MockJob:
    """Mock job for testing."""
    title: str
    company: str
    location: str
    url: str
    semantic_score: float
    score_reasoning: str
    is_remote: bool
    distance_miles: Optional[float]
    matched_city: Optional[str]  # City stanza that matched, or "Remote"
    source: str


# Sample jobs with real-looking URLs for testing
SAMPLE_JOBS = [
    MockJob(
        title="Senior DevOps Engineer",
        company="Acme Corporation",
        location="New York, NY",
        url="https://www.linkedin.com/jobs/view/3812345678",
        semantic_score=9.2,
        score_reasoning="Excellent match for Kubernetes and cloud infrastructure expertise. "
                       "Role emphasizes GitOps workflows and team leadership.",
        is_remote=False,
        distance_miles=5.2,
        matched_city="nyc",
        source="linkedin"
    ),
    MockJob(
        title="Platform Engineering Manager",
        company="TechStartup Inc",
        location="Remote - United States",
        url="https://www.indeed.com/viewjob?jk=abc123def456",
        semantic_score=8.8,
        score_reasoning="Strong alignment with platform engineering goals. Includes CI/CD "
                       "pipeline design and infrastructure automation.",
        is_remote=True,
        distance_miles=None,
        matched_city="Remote",
        source="indeed"
    ),
    MockJob(
        title="Site Reliability Engineer",
        company="Global Finance Corp",
        location="Chicago, IL",
        url="https://www.linkedin.com/jobs/view/3898765432",
        semantic_score=8.5,
        score_reasoning="Hands-on SRE role with Kubernetes and observability focus. "
                       "Some mentorship opportunities.",
        is_remote=False,
        distance_miles=12.3,
        matched_city="chicago",
        source="linkedin"
    ),
    MockJob(
        title="Cloud Infrastructure Architect",
        company="Healthcare Systems LLC",
        location="Hybrid - Boston, MA",
        url="https://www.indeed.com/viewjob?jk=xyz789abc123",
        semantic_score=8.2,
        score_reasoning="Architecture role focusing on AWS infrastructure design. "
                       "Hybrid work arrangement available.",
        is_remote=False,
        distance_miles=8.7,
        matched_city="boston",
        source="indeed"
    ),
    MockJob(
        title="DevOps Technical Lead",
        company="Retail Giants Inc",
        location="Remote",
        url="https://www.linkedin.com/jobs/view/3856789012",
        semantic_score=7.9,
        score_reasoning="Technical leadership position. Strong CI/CD focus "
                       "with team management responsibilities.",
        is_remote=True,
        distance_miles=None,
        matched_city="Remote",
        source="linkedin"
    ),
    MockJob(
        title="Kubernetes Platform Engineer",
        company="Manufacturing Co",
        location="Dallas, TX",
        url="https://www.indeed.com/viewjob?jk=pec123spec456",
        semantic_score=7.6,
        score_reasoning="Specialized K8s role. Strong technical focus but limited leadership scope.",
        is_remote=False,
        distance_miles=15.1,
        matched_city="austin",
        source="indeed"
    ),
    MockJob(
        title="Senior Infrastructure Engineer",
        company="University Health",
        location="Philadelphia, PA",
        url="https://www.linkedin.com/jobs/view/3823456789",
        semantic_score=7.3,
        score_reasoning="Senior infrastructure role with cloud focus. More operational than strategic.",
        is_remote=False,
        distance_miles=22.4,
        matched_city="nyc",
        source="linkedin"
    ),
]


def build_summary_blocks(jobs: list[MockJob], date: datetime) -> list[dict]:
    """Build the summary message blocks."""
    avg_score = sum(j.semantic_score for j in jobs) / len(jobs)
    remote_count = sum(1 for j in jobs if j.is_remote)
    top_score = max(j.semantic_score for j in jobs)
    
    by_source = {}
    for job in jobs:
        by_source[job.source] = by_source.get(job.source, 0) + 1
    source_text = " | ".join(
        f"{src.title()}: {cnt}" for src, cnt in sorted(by_source.items())
    )
    
    thread_count = (len(jobs) + 4) // 5
    
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"🎯 Semantic Scout Report - {date.strftime('%B %d, %Y')}",
                "emoji": True
            }
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*🎯 Jobs Found:*\n{len(jobs)}"},
                {"type": "mrkdwn", "text": f"*🏆 Top Score:*\n{top_score:.1f}/10"},
                {"type": "mrkdwn", "text": f"*📊 Avg Score:*\n{avg_score:.1f}/10"},
                {"type": "mrkdwn", "text": f"*🏠 Remote:*\n{remote_count}"}
            ]
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"📍 *Sources:* {source_text}"}
            ]
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*📋 {len(jobs)} opportunities ranked by score*\n"
                    f"_See {thread_count} threaded replies below for full details_"
                )
            }
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": "Generated by *Semantic Scout* 🔍 (TEST MESSAGE)"}
            ]
        }
    ]
    
    return blocks


def build_job_batch_blocks(jobs: list[MockJob], start_rank: int) -> list[dict]:
    """Build blocks for a batch of jobs (threaded reply)."""
    blocks = []
    
    for i, job in enumerate(jobs):
        rank = start_rank + i
        remote_badge = " 🏠" if job.is_remote else ""
        
        # Score emoji
        if job.semantic_score >= 9:
            score_emoji = "🔥"
        elif job.semantic_score >= 8:
            score_emoji = "⭐"
        elif job.semantic_score >= 7:
            score_emoji = "✨"
        else:
            score_emoji = "📋"
        
        job_header = f"*#{rank}* {score_emoji} *{job.title}*{remote_badge}"
        
        details_parts = [f"🏢 _{job.company}_"]
        if job.location:
            details_parts.append(f"📍 {job.location}")
        details_parts.append(f"Score: *{job.semantic_score:.1f}/10*")
        
        # Add matched city info
        if job.matched_city:
            if job.matched_city == "Remote":
                pass  # Already shown via remote_badge
            elif job.matched_city == "Anywhere":
                details_parts.append("🌍 Anywhere")
            else:
                distance_str = f" {job.distance_miles:.1f}mi" if job.distance_miles else ""
                details_parts.append(f"📌 {job.matched_city}{distance_str}")
        elif job.distance_miles is not None and not job.is_remote:
            details_parts.append(f"({job.distance_miles:.1f} mi)")
        
        details_line = " • ".join(details_parts)
        job_text = f"{job_header}\n{details_line}"
        
        block = {
            "type": "section",
            "text": {"type": "mrkdwn", "text": job_text}
        }
        
        if job.url:
            block["accessory"] = {
                "type": "button",
                "text": {"type": "plain_text", "text": "View Job →", "emoji": True},
                "url": job.url,
                "action_id": f"view_job_{rank}"
            }
        
        blocks.append(block)
        
        # Add reasoning
        if job.score_reasoning:
            reasoning = job.score_reasoning[:200]
            if len(job.score_reasoning) > 200:
                reasoning += "..."
            blocks.append({
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"💡 _{reasoning}_"}]
            })
        
        # Divider between jobs (except last)
        if i < len(jobs) - 1:
            blocks.append({"type": "divider"})
    
    return blocks


def post_test_message(bot_token: str, channel_id: str) -> bool:
    """Post test message with threaded replies."""
    headers = {
        "Authorization": f"Bearer {bot_token}",
        "Content-Type": "application/json",
    }
    
    date = datetime.now()
    jobs = SAMPLE_JOBS
    
    print(f"\n🚀 Posting test message to channel {channel_id}...")
    print(f"   Using {len(jobs)} sample jobs")
    
    with httpx.Client(timeout=30.0) as client:
        # Post summary message
        summary_blocks = build_summary_blocks(jobs, date)
        text = f"🎯 Semantic Scout Test - {len(jobs)} opportunities found"
        
        response = client.post(
            "https://slack.com/api/chat.postMessage",
            headers=headers,
            json={
                "channel": channel_id,
                "text": text,
                "blocks": summary_blocks,
            }
        )
        
        if response.status_code != 200:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        result = response.json()
        if not result.get("ok"):
            print(f"❌ Slack API error: {result.get('error')}")
            if result.get("error") == "channel_not_found":
                print("   Make sure the bot is added to the channel!")
            return False
        
        thread_ts = result.get("ts")
        print(f"✅ Summary message posted (ts: {thread_ts})")
        
        # Post threaded replies (2 batches for test)
        batch_size = 5
        num_batches = 2  # Just 2 for testing
        
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            batch = jobs[start_idx:start_idx + batch_size]
            
            if not batch:
                break
            
            start_rank = start_idx + 1
            thread_blocks = build_job_batch_blocks(batch, start_rank)
            
            thread_response = client.post(
                "https://slack.com/api/chat.postMessage",
                headers=headers,
                json={
                    "channel": channel_id,
                    "thread_ts": thread_ts,
                    "text": f"Jobs {start_rank}-{start_rank + len(batch) - 1}",
                    "blocks": thread_blocks,
                }
            )
            
            if thread_response.status_code == 200:
                thread_result = thread_response.json()
                if thread_result.get("ok"):
                    print(f"✅ Thread reply {batch_num + 1} posted (jobs {start_rank}-{start_rank + len(batch) - 1})")
                else:
                    print(f"⚠️  Thread {batch_num + 1} error: {thread_result.get('error')}")
            else:
                print(f"⚠️  Thread {batch_num + 1} HTTP error: {thread_response.status_code}")
        
        print("\n🎉 Test complete! Check your Slack channel.")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Test Slack threaded message integration"
    )
    parser.add_argument(
        "--bot-token",
        default=os.environ.get("SLACK_BOT_TOKEN"),
        help="Slack bot token (or set SLACK_BOT_TOKEN env var)"
    )
    parser.add_argument(
        "--channel-id",
        default=os.environ.get("SLACK_CHANNEL_ID"),
        help="Slack channel ID (or set SLACK_CHANNEL_ID env var)"
    )
    
    args = parser.parse_args()
    
    if not args.bot_token:
        print("❌ Error: Bot token required")
        print("   Use --bot-token or set SLACK_BOT_TOKEN environment variable")
        sys.exit(1)
    
    if not args.channel_id:
        print("❌ Error: Channel ID required")
        print("   Use --channel-id or set SLACK_CHANNEL_ID environment variable")
        print("   (Right-click channel -> View channel details -> Copy ID)")
        sys.exit(1)
    
    success = post_test_message(args.bot_token, args.channel_id)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
