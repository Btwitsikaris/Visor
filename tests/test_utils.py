from src import utils


def test_fix_plate_exact():
    assert utils.fix_plate("DL3CAB1234") == "DL3CAB1234"
    assert utils.fix_plate("MH12AB1234") == "MH12AB1234"


def test_fix_plate_ocr_confusions():
    assert utils.fix_plate("MH12AB123O") == "MH12AB1230"   # O -> 0 in digit slot
    assert utils.fix_plate("DL8CAB1234") == "DL8CAB1234"
    assert utils.fix_plate("0L3CAB1234") == "DL3CAB1234"   # 0 -> O in letter slot, O -> D (valid state)


def test_fix_plate_rejects_garbage():
    assert utils.fix_plate("HELLO") is None
    assert utils.fix_plate("") is None


def test_state_code_is_checked():
    assert utils.fix_plate("JL6SBF5363") == "DL6SBF5363"   # J is not a state code; D is (OCR confusion)
    assert utils.fix_plate("ZZ6SBF5363") is None           # right shape, impossible state code


def test_pick_plate_votes_per_character():
    reads = [("DL6SBF5363", 0.5), ("DL6SBI5363", 0.3), ("DL6SBF5368", 0.4)]
    assert utils.pick_plate(reads) == ("DL6SBF5363", True)


def test_pick_plate_votes():
    reads = [("DL3CAB1234", 0.6), ("DL3CAB1Z34", 0.3), ("DL3CAB1234", 0.5)]
    assert utils.pick_plate(reads) == ("DL3CAB1234", True)


def test_pick_plate_fallbacks():
    assert utils.pick_plate([("AB12", 0.9)]) == ("AB12", False)
    assert utils.pick_plate([]) == ("UNREADABLE", False)


def test_geometry():
    zone = utils.rider_zone((100, 200, 200, 300))
    assert zone[1] < 200 and zone[3] == 300
    assert utils.point_in_box((150, 150), zone)
    assert not utils.point_in_box((10, 10), zone)
    assert utils.norm_name("No-Helmet") == "no_helmet"


def test_db_roundtrip(tmp_path):
    from src import db
    conn = db.connect(tmp_path / "t.db")
    db.insert(conn, {"detected_at": "2026-01-01T00:00:00", "plate": "DL3CAB1234", "plate_valid": 1})
    db.export_csv(tmp_path / "t.db", tmp_path / "t.csv")
    assert "DL3CAB1234" in (tmp_path / "t.csv").read_text()