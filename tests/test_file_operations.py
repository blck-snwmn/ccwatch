"""Tests for file operations"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import AppConfig
from data.processors import get_jsonl_files


class TestGetJsonlFiles:
    """Test file search functionality"""

    def test_find_all_jsonl_files_in_directory(self, claude_projects_with_jsonl):
        """Test finding JSONL files in Claude projects directory"""
        projects_dir, expected_files = claude_projects_with_jsonl

        # Create test config
        test_config = AppConfig()
        test_config.claude_projects_path = projects_dir

        # Get files using dependency injection
        files = get_jsonl_files(test_config)

        # Should find all JSONL files
        assert len(files) == len(expected_files), f"Expected {len(expected_files)} files but found {len(files)}"

        # Convert to Path objects for comparison
        found_paths = {Path(f) for f in files}
        expected_paths = set(expected_files)

        assert found_paths == expected_paths, "Found files do not match expected files"

    def test_files_sorted_newest_first(self, claude_projects_with_jsonl):
        """Test that files are sorted by modification time (newest first)"""
        projects_dir, jsonl_files = claude_projects_with_jsonl

        # Modify file times to ensure specific order
        for i, file_path in enumerate(jsonl_files):
            # Set modification time - older files have lower timestamp
            mod_time = datetime.now().timestamp() - (len(jsonl_files) - i) * 10
            os.utime(file_path, (mod_time, mod_time))

        # Mock the config
        test_config = AppConfig()
        test_config.claude_projects_path = projects_dir
        # Get files using dependency injection
        files = get_jsonl_files(test_config)

        # Verify they're sorted by modification time (newest first)
        mtimes = [os.path.getmtime(f) for f in files]
        assert mtimes == sorted(mtimes, reverse=True), "Files are not sorted by modification time (newest first)"

    def test_get_jsonl_files_empty_directory(self, tmp_path):
        """Test behavior with empty Claude projects directory"""
        empty_dir = tmp_path / ".claude" / "projects"
        empty_dir.mkdir(parents=True)

        # Mock the config
        test_config = AppConfig()
        test_config.claude_projects_path = empty_dir
        # Get files using dependency injection
        files = get_jsonl_files(test_config)

        # Should return empty list
        assert files == [], "Expected empty list for empty directory"

    def test_missing_directory_shows_warning(self, tmp_path, caplog):
        """Test behavior when Claude projects directory doesn't exist"""
        nonexistent_dir = tmp_path / "does_not_exist"

        # Mock the config
        test_config = AppConfig()
        test_config.claude_projects_path = nonexistent_dir
        # Get files using dependency injection
        files = get_jsonl_files(test_config)

        # Should return empty list and log warning
        assert files == [], "Expected empty list for non-existent directory"

        # Check that warning was logged
        assert any("ClaudeCode projects directory not found" in record.message for record in caplog.records), (
            "Expected warning log message about missing directory"
        )

        # The log message is actually in the extra data, not in the message itself
        # So we just need to verify the warning was logged - the path is in structured logging data

    def test_get_jsonl_files_with_subdirectories(self, tmp_path):
        """Test finding JSONL files in nested subdirectories"""
        projects_dir = tmp_path / ".claude" / "projects"
        projects_dir.mkdir(parents=True)

        # Create nested structure with JSONL files
        (projects_dir / "project1").mkdir()
        (projects_dir / "project1" / "logs.jsonl").touch()

        (projects_dir / "project2" / "subdir").mkdir(parents=True)
        (projects_dir / "project2" / "subdir" / "data.jsonl").touch()

        (projects_dir / "project3" / "deep" / "nested").mkdir(parents=True)
        (projects_dir / "project3" / "deep" / "nested" / "file.jsonl").touch()

        # Create non-JSONL files that should be ignored
        (projects_dir / "project1" / "readme.md").touch()
        (projects_dir / "project2" / "data.json").touch()

        # Mock the config
        test_config = AppConfig()
        test_config.claude_projects_path = projects_dir
        # Get files using dependency injection
        files = get_jsonl_files(test_config)

        # Should find only JSONL files
        assert len(files) == 3, f"Expected 3 JSONL files in nested directories but found {len(files)}"

        # Check that all JSONL files are found
        file_names = {Path(f).name for f in files}
        assert file_names == {"logs.jsonl", "data.jsonl", "file.jsonl"}, f"Unexpected file names: {file_names}"

    def test_get_jsonl_files_with_custom_pattern(self, claude_projects_with_jsonl):
        """Test using custom JSONL pattern"""
        projects_dir, jsonl_files = claude_projects_with_jsonl

        # Add some files that won't match custom pattern
        extra_file = projects_dir / "project1" / "other.jsonl"
        extra_file.touch()

        # Mock config with custom pattern
        test_config = AppConfig()
        test_config.claude_projects_path = projects_dir
        test_config.jsonl_pattern = "**/logs_*.jsonl"  # Only match files starting with logs_
        # Get files using dependency injection
        files = get_jsonl_files(test_config)

        # Should only find files matching the pattern
        file_names = {Path(f).name for f in files}
        assert all(name.startswith("logs_") for name in file_names)
        assert "other.jsonl" not in file_names

    def test_fast_file_search_with_many_files(self, tmp_path):
        """Test performance with many files"""
        projects_dir = tmp_path / ".claude" / "projects"

        # Create many project directories with files
        for i in range(10):
            project_dir = projects_dir / f"project_{i}"
            project_dir.mkdir(parents=True)

            # Create multiple JSONL files per project
            for j in range(5):
                (project_dir / f"log_{j}.jsonl").touch()

        # Create test config
        test_config = AppConfig()
        test_config.claude_projects_path = projects_dir

        # Measure time to find files
        import time

        start_time = time.time()
        files = get_jsonl_files(test_config)
        elapsed = time.time() - start_time

        # Should find all 50 files quickly
        assert len(files) == 50, f"Expected 50 files but found {len(files)}"
        assert elapsed < 1.0, f"File search took {elapsed:.2f}s, expected < 1.0s"

    @pytest.mark.parametrize(
        ("pattern", "expected_count"),
        [
            ("**/*.jsonl", 2),  # All JSONL files
            ("**/project1/*.jsonl", 1),  # Only in project1
            ("**/*_0.jsonl", 1),  # Only files ending with _0
            ("**/nonexistent/*.jsonl", 0),  # No matches
        ],
    )
    def test_glob_patterns_filter_correctly(self, claude_projects_with_jsonl, monkeypatch, pattern, expected_count):
        """Test different glob patterns"""
        projects_dir, _ = claude_projects_with_jsonl

        # Mock config with specific pattern
        test_config = AppConfig()
        test_config.claude_projects_path = projects_dir
        test_config.jsonl_pattern = pattern
        # Get files using dependency injection
        files = get_jsonl_files(test_config)

        # Check expected count
        assert len(files) == expected_count, (
            f"Pattern '{pattern}' expected {expected_count} files but found {len(files)}"
        )

    def test_single_file_in_deep_nested_directory(self, tmp_path):
        """Test finding a single file in very deep directory structure (boundary case)"""
        projects_dir = tmp_path / ".claude" / "projects"

        # Create a very deep nested structure (20 levels)
        deep_path = projects_dir
        for i in range(20):
            deep_path = deep_path / f"level_{i}"

        deep_path.mkdir(parents=True)
        jsonl_file = deep_path / "deep.jsonl"
        jsonl_file.touch()

        # Create test config
        test_config = AppConfig()
        test_config.claude_projects_path = projects_dir

        # Should still find the file
        files = get_jsonl_files(test_config)
        assert len(files) == 1, f"Expected to find 1 file in deep directory but found {len(files)}"
        assert files[0].endswith("deep.jsonl"), "Should find the deeply nested file"

    def test_files_with_special_characters_in_names(self, tmp_path):
        """Test handling files with special characters (boundary case)"""
        projects_dir = tmp_path / ".claude" / "projects" / "test"
        projects_dir.mkdir(parents=True)

        # Create files with special characters
        special_names = [
            "file with spaces.jsonl",
            "file-with-dashes.jsonl",
            "file_with_underscores.jsonl",
            "file.multiple.dots.jsonl",
            "file@special#chars.jsonl",
        ]

        for name in special_names:
            (projects_dir / name).touch()

        # Create test config
        test_config = AppConfig()
        test_config.claude_projects_path = tmp_path / ".claude" / "projects"

        # Should find all files
        files = get_jsonl_files(test_config)
        assert len(files) == len(special_names), (
            f"Expected {len(special_names)} files with special characters but found {len(files)}"
        )

    def test_maximum_files_performance_boundary(self, tmp_path):
        """Test performance with maximum reasonable number of files (1000)"""
        projects_dir = tmp_path / ".claude" / "projects"

        # Create 1000 files across 100 projects
        for i in range(100):
            project_dir = projects_dir / f"project_{i:03d}"
            project_dir.mkdir(parents=True)
            for j in range(10):
                (project_dir / f"log_{j:02d}.jsonl").touch()

        # Create test config
        test_config = AppConfig()
        test_config.claude_projects_path = projects_dir

        # Measure time
        import time

        start_time = time.time()
        files = get_jsonl_files(test_config)
        elapsed = time.time() - start_time

        assert len(files) == 1000, f"Expected 1000 files but found {len(files)}"
        assert elapsed < 2.0, f"Finding 1000 files took {elapsed:.2f}s, expected < 2.0s"
