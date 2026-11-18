"""metadata.txt vs the FILE_HEADER: where the firmware version and device ID come from."""

import pytest

from cass_logger_dev.cass_commands import CassCommands
from test_stream_format import file_header, source_info, tick, FW


def _stream_bin(path, device_id="AB123"):
    path.write_bytes(file_header(device_id=device_id) + source_info() + tick(40000))
    return path


def test_needs_metadata_txt(tmp_path):
    new = _stream_bin(tmp_path / "LOG_1.bin")
    legacy = tmp_path / "LOG_2.bin"
    legacy.write_bytes(bytes(80))
    assert not CassCommands.needs_metadata_txt([new])
    assert CassCommands.needs_metadata_txt([new, legacy])


def test_find_metadata_prefers_metadata_txt(tmp_path):
    _stream_bin(tmp_path / "LOG_1.bin")
    CassCommands.write_metadata_txt(tmp_path, "0.12-i2c_1", "OLD01")
    assert CassCommands.find_and_parse_metadata(str(tmp_path)) == {
        "firmware_version": "0.12-i2c_1", "device_id": "OLD01"}


def test_find_metadata_falls_back_to_headers(tmp_path):
    _stream_bin(tmp_path / "LOG_1.bin")
    _stream_bin(tmp_path / "LOG_2.bin", device_id="")
    assert CassCommands.find_and_parse_metadata(str(tmp_path)) == {
        "firmware_version": FW, "device_id": "AB123"}
    assert CassCommands.find_and_parse_metadata(str(tmp_path), first_only=False) == [
        {"firmware_version": FW, "device_id": "AB123"},
        {"firmware_version": FW, "device_id": None}]


def test_find_metadata_none(tmp_path):
    (tmp_path / "LOG_1.bin").write_bytes(bytes(80))  # legacy, and no metadata.txt
    assert CassCommands.find_and_parse_metadata(str(tmp_path)) is None


@pytest.mark.parametrize("bad", ["", "ABCDEF", "box", "AB\n", "AB©"])
def test_put_device_id_rejects_bad_ids(bad):
    # Rejected before anything goes to the serial port.
    cass = CassCommands.__new__(CassCommands)
    with pytest.raises(ValueError):
        cass.put_device_ID(bad)
