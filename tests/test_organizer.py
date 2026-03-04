from pathlib import Path

from wildlife_monitoring.organizer import iter_images, relocate_image


def test_iter_images_filters_supported_extensions(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.txt").write_text("ignore")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.png").write_bytes(b"y")

    images = sorted(p.name for p in iter_images(tmp_path, recursive=True))

    assert images == ["a.jpg", "c.png"]


def test_relocate_image_moves_file(tmp_path: Path) -> None:
    source = tmp_path / "image.jpg"
    source.write_bytes(b"bytes")

    destination = relocate_image(source, tmp_path / "out", "Deer", copy_files=False)

    assert destination.exists()
    assert destination.name == "image.jpg"
    assert not source.exists()
