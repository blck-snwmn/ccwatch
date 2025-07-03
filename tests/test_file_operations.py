"""Tests for file operations"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app import get_jsonl_files
from config import AppConfig


class TestGetJsonlFiles:
    """Test file search functionality"""

    def test_get_jsonl_files_with_existing_files(self, claude_projects_with_jsonl, monkeypatch):
        """Test finding JSONL files in Claude projects directory"""
        projects_dir, expected_files = claude_projects_with_jsonl

        # Mock the config to use our test directory
        test_config = AppConfig()
        test_config.claude_projects_path = projects_dir
        monkeypatch.setattr("app.config", test_config)

        # Get files
        files = get_jsonl_files()

        # Should find all JSONL files
        assert len(files) == len(expected_files)

        # Convert to Path objects for comparison
        found_paths = {Path(f) for f in files}
        expected_paths = set(expected_files)

        assert found_paths == expected_paths

    def test_get_jsonl_files_sorted_by_mtime(self, claude_projects_with_jsonl, monkeypatch):
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
        monkeypatch.setattr("app.config", test_config)

        # Get files
        files = get_jsonl_files()

        # Verify they're sorted by modification time (newest first)
        mtimes = [os.path.getmtime(f) for f in files]
        assert mtimes == sorted(mtimes, reverse=True)

    def test_get_jsonl_files_empty_directory(self, tmp_path, monkeypatch):
        """Test behavior with empty Claude projects directory"""
        empty_dir = tmp_path / ".claude" / "projects"
        empty_dir.mkdir(parents=True)

        # Mock the config
        test_config = AppConfig()
        test_config.claude_projects_path = empty_dir
        monkeypatch.setattr("app.config", test_config)

        # Get files
        files = get_jsonl_files()

        # Should return empty list
        assert files == []

    def test_get_jsonl_files_nonexistent_directory(self, tmp_path, monkeypatch, capsys):
        """Test behavior when Claude projects directory doesn't exist"""
        nonexistent_dir = tmp_path / "does_not_exist"

        # Mock the config
        test_config = AppConfig()
        test_config.claude_projects_path = nonexistent_dir
        monkeypatch.setattr("app.config", test_config)

        # Mock streamlit warning
        warnings = []

        def mock_warning(msg):
            warnings.append(msg)

        monkeypatch.setattr("streamlit.warning", mock_warning)

        # Get files
        files = get_jsonl_files()

        # Should return empty list and show warning
        assert files == []
        assert len(warnings) == 1
        assert "ClaudeCode projects directory not found" in warnings[0]
        assert str(nonexistent_dir) in warnings[0]

    def test_get_jsonl_files_with_subdirectories(self, tmp_path, monkeypatch):
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
        monkeypatch.setattr("app.config", test_config)

        # Get files
        files = get_jsonl_files()

        # Should find only JSONL files
        assert len(files) == 3

        # Check that all JSONL files are found
        file_names = {Path(f).name for f in files}
        assert file_names == {"logs.jsonl", "data.jsonl", "file.jsonl"}

    def test_get_jsonl_files_with_custom_pattern(self, claude_projects_with_jsonl, monkeypatch):
        """Test using custom JSONL pattern"""
        projects_dir, jsonl_files = claude_projects_with_jsonl

        # Add some files that won't match custom pattern
        extra_file = projects_dir / "project1" / "other.jsonl"
        extra_file.touch()

        # Mock config with custom pattern
        test_config = AppConfig()
        test_config.claude_projects_path = projects_dir
        test_config.jsonl_pattern = "**/logs_*.jsonl"  # Only match files starting with logs_
        monkeypatch.setattr("app.config", test_config)

        # Get files
        files = get_jsonl_files()

        # Should only find files matching the pattern
        file_names = {Path(f).name for f in files}
        assert all(name.startswith("logs_") for name in file_names)
        assert "other.jsonl" not in file_names

    def test_get_jsonl_files_performance(self, tmp_path, monkeypatch):
        """Test performance with many files"""
        projects_dir = tmp_path / ".claude" / "projects"

        # Create many project directories with files
        for i in range(10):
            project_dir = projects_dir / f"project_{i}"
            project_dir.mkdir(parents=True)

            # Create multiple JSONL files per project
            for j in range(5):
                (project_dir / f"log_{j}.jsonl").touch()

        # Mock the config
        test_config = AppConfig()
        test_config.claude_projects_path = projects_dir
        monkeypatch.setattr("app.config", test_config)

        # Measure time to find files
        import time

        start_time = time.time()
        files = get_jsonl_files()
        elapsed = time.time() - start_time

        # Should find all 50 files quickly
        assert len(files) == 50
        assert elapsed < 1.0  # Should complete in less than 1 second

    @pytest.mark.parametrize(
        ("pattern", "expected_count"),
        [
            ("**/*.jsonl", 2),  # All JSONL files
            ("**/project1/*.jsonl", 1),  # Only in project1
            ("**/*_0.jsonl", 1),  # Only files ending with _0
            ("**/nonexistent/*.jsonl", 0),  # No matches
        ],
    )
    def test_get_jsonl_files_various_patterns(self, claude_projects_with_jsonl, monkeypatch, pattern, expected_count):
        """Test different glob patterns"""
        projects_dir, _ = claude_projects_with_jsonl

        # Mock config with specific pattern
        test_config = AppConfig()
        test_config.claude_projects_path = projects_dir
        test_config.jsonl_pattern = pattern
        monkeypatch.setattr("app.config", test_config)

        # Get files
        files = get_jsonl_files()

        # Check expected count
        assert len(files) == expected_count
