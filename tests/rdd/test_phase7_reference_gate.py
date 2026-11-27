"""Reference can only be run with actual externally produced R output."""
import os
from pathlib import Path
import pytest
from validation.compare_phase7_r import compare


def test_missing_r_reference_fails_closed(tmp_path):
    fixture=Path(__file__).resolve().parents[2]/'validation/fixtures/rdd_phase7_fixed.csv'
    with pytest.raises(FileNotFoundError,match='BLOCKED'):
        compare(fixture,tmp_path/'unavailable_r_output.csv',tmp_path/'out.json')


@pytest.mark.parity
def test_external_rdrobust_reference(tmp_path):
    actual=os.environ.get('QUASI_PHASE7_R_REFERENCE')
    if not actual:
        pytest.skip('Authentic R rdrobust output not supplied; reference parity NOT certified')
    fixture=Path(__file__).resolve().parents[2]/'validation/fixtures/rdd_phase7_fixed.csv'
    compare(fixture,Path(actual),tmp_path/'phase7_comparison.json')
