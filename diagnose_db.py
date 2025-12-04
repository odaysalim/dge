#!/usr/bin/env python3
"""
Database Connection Diagnostic Tool
Helps identify why ingestion data isn't appearing in Docker PostgreSQL
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import subprocess

# Colors for terminal output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color

def print_header(msg):
    print(f"\n{'='*70}")
    print(f"{msg}")
    print(f"{'='*70}")

def print_success(msg):
    print(f"{GREEN}✅ {msg}{NC}")

def print_error(msg):
    print(f"{RED}❌ {msg}{NC}")

def print_warning(msg):
    print(f"{YELLOW}⚠️  {msg}{NC}")

def check_docker_running():
    """Check if Docker containers are running"""
    print_header("Step 1: Checking Docker Services")

    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True,
            text=True,
            check=True
        )

        if result.stdout:
            print_success("Docker Compose is running")
            print(result.stdout)
            return True
        else:
            print_error("No Docker Compose services found")
            return False

    except subprocess.CalledProcessError:
        print_error("Docker Compose not running or not available")
        return False
    except FileNotFoundError:
        print_error("Docker command not found")
        return False

def test_database_connection(db_url, description):
    """Test connection to a PostgreSQL database"""
    try:
        print(f"\nTesting {description}:")
        print(f"  URL: {db_url}")

        engine = create_engine(db_url)

        with engine.connect() as conn:
            # Get PostgreSQL version
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print_success(f"Connected to PostgreSQL")
            print(f"  Version: {version[:60]}...")

            # Check pgvector extension
            result = conn.execute(text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"))
            has_pgvector = result.fetchone()[0]

            if has_pgvector:
                print_success("pgvector extension installed")
            else:
                print_warning("pgvector extension NOT installed")

            # List all tables
            result = conn.execute(text("""
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                AND tablename LIKE '%rag%'
                ORDER BY tablename
            """))
            tables = result.fetchall()

            if tables:
                print_success(f"Found {len(tables)} RAG-related tables:")
                for table in tables:
                    table_name = table[0]
                    # Count rows
                    count_result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                    count = count_result.fetchone()[0]
                    print(f"  - {table_name}: {count} rows")
            else:
                print_warning("No RAG-related tables found")

            return True

    except Exception as e:
        print_error(f"Connection failed: {e}")
        return False

def main():
    """Main diagnostic routine"""
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║          Database Connection Diagnostic Tool                     ║
║  Identifies why ingestion data isn't in Docker PostgreSQL        ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # Load environment variables
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
        print_success(f"Loaded environment from {env_file}")
    else:
        print_warning(f".env file not found, using defaults")

    # Check Docker
    docker_running = check_docker_running()

    # Test connections
    print_header("Step 2: Testing Database Connections")

    # 1. Test Docker PostgreSQL connection
    docker_db_url = "postgresql://postgres:postgres_password@localhost:5432/rag_db"
    print(f"\n[1] Docker PostgreSQL (expected location):")
    docker_connected = test_database_connection(docker_db_url, "Docker PostgreSQL")

    # 2. Test .env DATABASE_URL
    env_db_url = os.getenv("DATABASE_URL", "NOT SET")
    if env_db_url != "NOT SET" and env_db_url != docker_db_url:
        print(f"\n[2] DATABASE_URL from .env:")
        env_connected = test_database_connection(env_db_url, ".env DATABASE_URL")

    # Summary and recommendations
    print_header("Step 3: Diagnosis & Recommendations")

    if not docker_running:
        print_error("Docker is not running!")
        print("\nFix: Start Docker services:")
        print("  docker compose up -d postgres phoenix")
        return

    if not docker_connected:
        print_error("Cannot connect to Docker PostgreSQL!")
        print("\nPossible issues:")
        print("  1. PostgreSQL container not started")
        print("  2. Port 5432 not exposed or in use")
        print("  3. Wrong credentials")
        print("\nFix: Restart PostgreSQL container:")
        print("  docker compose down")
        print("  docker compose up -d postgres phoenix")
        return

    # Check if .env file exists and has correct DATABASE_URL
    if not env_file.exists():
        print_error(".env file missing!")
        print("\nFix: Create .env from template:")
        print("  cp .env.example .env")
        print("  # Then edit .env and set OPENAI_API_KEY")
        return

    if env_db_url != docker_db_url:
        print_warning("DATABASE_URL in .env doesn't match Docker PostgreSQL!")
        print(f"\n  Current: {env_db_url}")
        print(f"  Should be: {docker_db_url}")
        print("\nFix: Update DATABASE_URL in .env:")
        print(f"  DATABASE_URL={docker_db_url}")
        print("\nThen re-run ingestion:")
        print("  python src/data_ingestion/ingest_openai.py")
        return

    print_success("Database connection looks good!")
    print("\nIf you still have empty tables, re-run ingestion:")
    print("  python src/data_ingestion/ingest_openai.py")

if __name__ == "__main__":
    main()
