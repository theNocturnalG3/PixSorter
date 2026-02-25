from pixsorter.infra.resources import resource_root, resource_path

def test_resource_root_exists():
    rr = resource_root()
    assert rr.exists()
    assert rr.is_dir()

def test_resource_path_missing_returns_empty():
    assert resource_path("assets/definitely_missing_file.xyz") == ""