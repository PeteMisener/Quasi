"""An R fixture must be real, complete, checksum-consistent and not self-certified."""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd
import pytest
from validation.compare_phase6_r import _validate_producer, compare


def test_missing_executed_R_producer_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match='R producer outputs'):
        compare(tmp_path)


def test_incomplete_R_provenance_manifest_rejected(tmp_path):
    (tmp_path/'versions.txt').write_text('R=4.4.3\ndid=2.2\n')
    with pytest.raises(ValueError, match='package-version'):
        compare(tmp_path)


def test_tampered_R_producer_fixture_rejected_before_parity_comparison(tmp_path):
    (tmp_path/'versions.txt').write_text('R=4.4.3\ndid=2.2\nDRDID=1.2\n')
    file=tmp_path/'mpdta_input.csv'
    file.write_text('g,t\n1,2\n')
    pd.DataFrame([{'filename':file.name,'md5':hashlib.md5(file.read_bytes()).hexdigest()}]).to_csv(tmp_path/'checksums.csv', index=False)
    file.write_text('g,t\n1,3\n')
    with pytest.raises(ValueError, match='Corrupted R reference'):
        compare(tmp_path)


def test_empty_or_path_traversal_manifest_rejected(tmp_path):
    (tmp_path/'versions.txt').write_text('R=4.4.3\ndid=2.2\nDRDID=1.2\n')
    pd.DataFrame([{'filename':'../unrelated.csv', 'md5':'0'*32}]).to_csv(tmp_path/'checksums.csv', index=False)
    with pytest.raises(ValueError, match='invalid checksum inventory'):
        compare(tmp_path)
