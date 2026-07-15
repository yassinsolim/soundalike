import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import soundalike.ml.jamendo_fulltrack as jamendo_fulltrack
from soundalike.ml.jamendo_fulltrack import (
    JamendoValidationError,
    load_jamendo_context,
    safe_relative_path,
)


TRACKS = (
    (1, 11, 21, "01/1.mp3", "genre---rock"),
    (2, 12, 22, "02/2.mp3", "genre---jazz"),
    (3, 13, 23, "03/3.mp3", "mood/theme---happy"),
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_tsv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS"]
    for track_id, artist_id, album_id, relative, tag in rows:
        lines.append(
            f"track_{track_id:07d}\tartist_{artist_id:06d}\t"
            f"album_{album_id:06d}\t{relative}\t31.5\t{tag}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_fixture(tmp_path: Path, *, completion: bool = True):
    metadata = tmp_path / "metadata"
    data = metadata / "data"
    download = data / "download"
    audio = tmp_path / "audio"
    state = tmp_path / "state"
    download.mkdir(parents=True)
    audio.mkdir()
    state.mkdir()

    payloads = {}
    for track_id, _artist, _album, relative, _tag in TRACKS:
        payload = f"synthetic-mp3-{track_id}".encode()
        path = audio / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        payloads[relative] = payload

    _write_tsv(data / "raw_30s.tsv", TRACKS)
    _write_tsv(data / "raw_30s_cleantags_50artists.tsv", TRACKS)
    (data / "autotagging.tsv").write_text(
        "raw_30s_cleantags_50artists.tsv", encoding="utf-8"
    )
    meta_lines = [
        "TRACK_ID\tARTIST_ID\tALBUM_ID\tTRACK_NAME\tARTIST_NAME\t"
        "ALBUM_NAME\tRELEASEDATE\tURL"
    ]
    for track_id, artist_id, album_id, _relative, _tag in TRACKS:
        meta_lines.append(
            f"track_{track_id:07d}\tartist_{artist_id:06d}\t"
            f"album_{album_id:06d}\tTrack {track_id}\tArtist {artist_id}\t"
            f"Album {album_id}\t2020-01-01\thttp://www.jamendo.com/track/{track_id}"
        )
    (data / "raw.meta.tsv").write_text(
        "\n".join(meta_lines) + "\n", encoding="utf-8"
    )
    license_blocks = []
    for track_id, _artist, _album, relative, _tag in TRACKS:
        license_blocks.append(
            f"{relative}\n"
            f"Track {track_id} by Artist from Jamendo: "
            f"http://www.jamendo.com/track/{track_id}\n"
            "Available under a Creative Commons Attribution-Non-Commercial-"
            "Share-Alike license: http://creativecommons.org/licenses/by-nc-sa/3.0/"
        )
    (metadata / "audio_licenses.txt").write_text(
        "\n\n".join(license_blocks) + "\n", encoding="utf-8"
    )
    for part, row in zip(("train", "validation", "test"), TRACKS):
        _write_tsv(
            data / "splits" / "split-0" / f"autotagging-{part}.tsv", [row]
        )

    archive_name = "raw_30s_audio-00.tar"
    archive_manifest = f"{'a' * 64} {archive_name}\n".encode()
    track_manifest = "".join(
        f"{_sha(payloads[relative])} {relative}\n"
        for _track, _artist, _album, relative, _tag in TRACKS
    ).encode()
    (download / "raw_30s_audio_sha256_tars.txt").write_bytes(archive_manifest)
    (download / "raw_30s_audio_sha256_tracks.txt").write_bytes(track_manifest)
    archive_hash = _sha(archive_manifest)
    track_hash = _sha(track_manifest)
    archive_marker = {
        "schema_version": 1,
        "archive": archive_name,
        "archive_sha256": "a" * 64,
        "archive_bytes": 100,
        "track_count": len(TRACKS),
        "track_bytes": sum(map(len, payloads.values())),
        "manifests": {
            "archives_sha256": archive_hash,
            "tracks_sha256": track_hash,
        },
        "tracks": [
            {
                "path": relative,
                "sha256": _sha(payloads[relative]),
                "bytes": len(payloads[relative]),
            }
            for _track, _artist, _album, relative, _tag in TRACKS
        ],
    }
    (state / f"{archive_name}.verified.json").write_text(
        json.dumps(archive_marker), encoding="utf-8"
    )
    if completion:
        (state / "collection.complete.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "collection": "raw_30s/audio",
                    "archive_count": 1,
                    "track_count": len(TRACKS),
                    "track_bytes": sum(map(len, payloads.values())),
                    "manifests": {
                        "archives_sha256": archive_hash,
                        "tracks_sha256": track_hash,
                    },
                }
            ),
            encoding="utf-8",
        )
    return metadata, audio, state


def load_fixture(tmp_path: Path):
    metadata, audio, state = make_fixture(tmp_path)
    return load_jamendo_context(
        metadata,
        audio,
        state,
        production=False,
        expected_tracks=3,
        expected_archives=1,
        fold_indices=(0,),
        verify_audio_hashes=True,
    )


def test_completion_marker_is_mandatory_before_metadata_parse(tmp_path):
    metadata, audio, state = make_fixture(tmp_path, completion=False)
    (metadata / "data" / "raw_30s.tsv").write_text("broken", encoding="utf-8")
    with pytest.raises(JamendoValidationError, match="collection.complete.json"):
        load_jamendo_context(
            metadata,
            audio,
            state,
            production=False,
            expected_tracks=3,
            expected_archives=1,
            fold_indices=(0,),
        )


def test_strict_context_audits_hashes_licenses_joins_and_folds(tmp_path):
    context = load_fixture(tmp_path)
    assert [track.track_id for track in context.tracks] == [1, 2, 3]
    assert context.tracks[0].fold_parts == ("train",)
    assert context.tracks[1].fold_parts == ("validation",)
    assert context.tracks[2].fold_parts == ("test",)
    assert context.tracks[0].license.permits_commercial_use is False
    assert len(context.source_fingerprint) == 64
    assert context.metadata_commit == "synthetic-fixture"


def test_official_fold_labels_are_retained_separately_from_raw_tags(tmp_path):
    metadata, audio, state = make_fixture(tmp_path)
    track_id, artist_id, album_id, relative, _raw_tag = TRACKS[2]
    _write_tsv(
        metadata / "data" / "splits" / "split-0" / "autotagging-test.tsv",
        [(track_id, artist_id, album_id, relative, "genre---official")],
    )
    context = load_jamendo_context(
        metadata,
        audio,
        state,
        production=False,
        expected_tracks=3,
        expected_archives=1,
        fold_indices=(0,),
        verify_audio_hashes=True,
    )
    assert context.by_track_id[track_id].tags == ("mood/theme---happy",)
    assert context.folds[0].track_tags[track_id] == ("genre---official",)


def _make_git_checkout(path: Path, tracked=None):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is unavailable")
    assert git is not None
    path.mkdir()
    commands = (
        ("init", "--quiet"),
        ("config", "user.name", "Soundalike test"),
        ("config", "user.email", "soundalike@example.invalid"),
    )
    for arguments in commands:
        subprocess.run(
            [git, "-C", str(path), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    for relative, content in (tracked or {"tracked.txt": "fixture"}).items():
        destination = path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    subprocess.run(
        [git, "-C", str(path), "add", "."],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [git, "-C", str(path), "commit", "--quiet", "-m", "fixture"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = subprocess.run(
        [git, "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        [git, "-C", str(path), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return git, commit, tree


def _commit_tracked_symlink(
    metadata: Path, git: str, target: str, *, placeholder: bool = False
):
    relative = "tracked-link"
    link = metadata / relative
    if placeholder:
        payload = os.fsencode(target)
        link.write_bytes(payload)
        blob = subprocess.run(
            [git, "-C", str(metadata), "hash-object", "-w", "--stdin"],
            input=payload,
            check=True,
            capture_output=True,
        ).stdout.decode("ascii").strip()
        subprocess.run(
            [
                git,
                "-C",
                str(metadata),
                "update-index",
                "--add",
                "--cacheinfo",
                f"120000,{blob},{relative}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("symlinks are unavailable")
        subprocess.run(
            [git, "-C", str(metadata), "add", relative],
            check=True,
            capture_output=True,
            text=True,
        )
    subprocess.run(
        [git, "-C", str(metadata), "commit", "--quiet", "-m", "tracked symlink"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = subprocess.run(
        [git, "-C", str(metadata), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        [git, "-C", str(metadata), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return link, commit, tree


def _pin_test_checkout(monkeypatch, commit: str, tree: str) -> None:
    monkeypatch.setattr(jamendo_fulltrack, "EXPECTED_METADATA_COMMIT", commit)
    monkeypatch.setattr(jamendo_fulltrack, "EXPECTED_METADATA_TREE", tree)


def _payload_command(tmp_path: Path, name: str):
    script = tmp_path / f"{name}.py"
    marker = tmp_path / f"{name}-executed.txt"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[1]).write_text('executed', encoding='ascii')\n"
        "raise SystemExit(1)\n",
        encoding="ascii",
    )
    command = f'"{sys.executable}" "{script}" "{marker}"'
    return command, marker


def test_metadata_commit_never_executes_repository_fsmonitor(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    git, commit, tree = _make_git_checkout(metadata)
    command, marker = _payload_command(tmp_path, "fsmonitor")
    subprocess.run(
        [git, "-C", str(metadata), "config", "core.fsmonitor", command],
        check=True,
        capture_output=True,
        text=True,
    )
    _pin_test_checkout(monkeypatch, commit, tree)

    assert jamendo_fulltrack._metadata_commit(metadata, production=True) == commit
    assert not marker.exists()


def test_metadata_commit_never_executes_content_filters(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    git, commit, tree = _make_git_checkout(
        metadata,
        {
            ".gitattributes": "tracked.txt filter=hostile\n",
            "tracked.txt": "fixture\n",
        },
    )
    markers = []
    for kind in ("clean", "smudge", "process"):
        command, marker = _payload_command(tmp_path, f"filter-{kind}")
        markers.append(marker)
        subprocess.run(
            [git, "-C", str(metadata), "config", f"filter.hostile.{kind}", command],
            check=True,
            capture_output=True,
            text=True,
        )
    subprocess.run(
        [git, "-C", str(metadata), "config", "filter.hostile.required", "true"],
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = metadata / "tracked.txt"
    stat = tracked.stat()
    # Preserve clean bytes but force an insecure `git status` to invoke the filter.
    os.utime(tracked, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    _pin_test_checkout(monkeypatch, commit, tree)

    assert jamendo_fulltrack._metadata_commit(metadata, production=True) == commit
    assert not any(marker.exists() for marker in markers)


def test_metadata_commit_ignores_inherited_git_config(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    _git, commit, tree = _make_git_checkout(metadata)
    command, marker = _payload_command(tmp_path, "inherited-fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", command)
    _pin_test_checkout(monkeypatch, commit, tree)

    assert jamendo_fulltrack._metadata_commit(metadata, production=True) == commit
    assert not marker.exists()


def test_metadata_commit_ignores_git_environment_repository_redirection(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    _git, commit, tree = _make_git_checkout(metadata)
    redirected = tmp_path / "redirected"
    _make_git_checkout(redirected, {"other.txt": "not the approved repository"})
    monkeypatch.setenv("GIT_DIR", str(redirected / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(redirected))
    _pin_test_checkout(monkeypatch, commit, tree)

    assert jamendo_fulltrack._metadata_commit(metadata, production=True) == commit


def test_metadata_commit_rejects_gitfile_repository_redirection(tmp_path, monkeypatch):
    source = tmp_path / "source"
    git, commit, tree = _make_git_checkout(source)
    metadata = tmp_path / "metadata"
    subprocess.run(
        [
            git,
            "-C",
            str(source),
            "worktree",
            "add",
            "--quiet",
            "--detach",
            str(metadata),
            commit,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (metadata / ".git").is_file()
    _pin_test_checkout(monkeypatch, commit, tree)

    with pytest.raises(JamendoValidationError, match="concrete directory"):
        jamendo_fulltrack._metadata_commit(metadata, production=True)


def test_metadata_commit_rejects_commondir_repository_redirection(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    _git, commit, tree = _make_git_checkout(metadata)
    (metadata / ".git" / "commondir").write_text("../redirected\n", encoding="ascii")
    _pin_test_checkout(monkeypatch, commit, tree)

    with pytest.raises(JamendoValidationError, match="commondir"):
        jamendo_fulltrack._metadata_commit(metadata, production=True)


def test_metadata_commit_rejects_alternate_object_store(tmp_path, monkeypatch):
    source = tmp_path / "source"
    git, commit, tree = _make_git_checkout(source)
    metadata = tmp_path / "metadata"
    subprocess.run(
        [git, "clone", "--quiet", "--shared", str(source), str(metadata)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (metadata / ".git" / "objects" / "info" / "alternates").is_file()
    _pin_test_checkout(monkeypatch, commit, tree)

    with pytest.raises(JamendoValidationError, match="objects/info/alternates"):
        jamendo_fulltrack._metadata_commit(metadata, production=True)


def test_metadata_commit_rejects_unexpected_head_commit(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    git, expected_commit, expected_tree = _make_git_checkout(metadata)
    (metadata / "tracked.txt").write_text("replacement", encoding="utf-8")
    subprocess.run(
        [git, "-C", str(metadata), "commit", "--quiet", "-am", "drift"],
        check=True,
        capture_output=True,
        text=True,
    )
    _pin_test_checkout(monkeypatch, expected_commit, expected_tree)

    with pytest.raises(JamendoValidationError, match="commit drift"):
        jamendo_fulltrack._metadata_commit(metadata, production=True)


def test_metadata_commit_rejects_index_tree_drift(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    git, commit, tree = _make_git_checkout(metadata)
    (metadata / "tracked.txt").write_text("replacement", encoding="utf-8")
    subprocess.run(
        [git, "-C", str(metadata), "add", "tracked.txt"],
        check=True,
        capture_output=True,
        text=True,
    )
    _pin_test_checkout(monkeypatch, commit, tree)

    with pytest.raises(JamendoValidationError, match="index tree drift"):
        jamendo_fulltrack._metadata_commit(metadata, production=True)


def test_metadata_commit_rejects_tracked_worktree_drift(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    _git, commit, tree = _make_git_checkout(metadata)
    _pin_test_checkout(monkeypatch, commit, tree)
    (metadata / "tracked.txt").write_text("modified", encoding="utf-8")

    with pytest.raises(JamendoValidationError, match="tracked modifications"):
        jamendo_fulltrack._metadata_commit(metadata, production=True)


def test_metadata_commit_accepts_tracked_symlink(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    git, _commit, _tree = _make_git_checkout(metadata)
    _link, commit, tree = _commit_tracked_symlink(metadata, git, "tracked.txt")
    _pin_test_checkout(monkeypatch, commit, tree)

    assert jamendo_fulltrack._metadata_commit(metadata, production=True) == commit


def test_metadata_commit_hashes_symlink_without_following_target(
    tmp_path, monkeypatch
):
    metadata = tmp_path / "metadata"
    git, _commit, _tree = _make_git_checkout(metadata)
    link, commit, tree = _commit_tracked_symlink(
        metadata, git, "missing-target"
    )
    _pin_test_checkout(monkeypatch, commit, tree)

    assert link.is_symlink()
    assert not link.exists()
    assert jamendo_fulltrack._metadata_commit(metadata, production=True) == commit


def test_metadata_commit_rejects_tampered_symlink_payload(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    git, _commit, _tree = _make_git_checkout(metadata)
    link, commit, tree = _commit_tracked_symlink(metadata, git, "tracked.txt")
    _pin_test_checkout(monkeypatch, commit, tree)
    link.unlink()
    link.symlink_to("different-target.txt")

    with pytest.raises(JamendoValidationError, match="tracked modifications"):
        jamendo_fulltrack._metadata_commit(metadata, production=True)


def test_metadata_commit_accepts_core_symlinks_false_placeholder(
    tmp_path, monkeypatch
):
    metadata = tmp_path / "metadata"
    git, _commit, _tree = _make_git_checkout(metadata)
    link, commit, tree = _commit_tracked_symlink(
        metadata, git, "tracked.txt", placeholder=True
    )
    _pin_test_checkout(monkeypatch, commit, tree)

    assert not link.is_symlink()
    assert link.read_bytes() == b"tracked.txt"
    assert jamendo_fulltrack._metadata_commit(metadata, production=True) == commit


def test_completion_manifest_hash_cannot_be_replaced_by_correct_counts(tmp_path):
    metadata, audio, state = make_fixture(tmp_path)
    marker_path = state / "collection.complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["manifests"]["tracks_sha256"] = "f" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(JamendoValidationError, match="track manifest hash mismatch"):
        load_jamendo_context(
            metadata,
            audio,
            state,
            production=False,
            expected_tracks=3,
            expected_archives=1,
            fold_indices=(0,),
        )


def test_artist_crossing_official_fold_is_rejected(tmp_path):
    metadata, audio, state = make_fixture(tmp_path)
    crossing_tracks = list(TRACKS)
    crossing = list(crossing_tracks[1])
    crossing[1] = TRACKS[0][1]
    crossing_tracks[1] = tuple(crossing)
    _write_tsv(metadata / "data" / "raw_30s.tsv", crossing_tracks)
    _write_tsv(
        metadata / "data" / "raw_30s_cleantags_50artists.tsv", crossing_tracks
    )
    meta_path = metadata / "data" / "raw.meta.tsv"
    meta_path.write_text(
        meta_path.read_text(encoding="utf-8").replace(
            "track_0000002\tartist_000012", "track_0000002\tartist_000011"
        ),
        encoding="utf-8",
    )
    _write_tsv(
        metadata
        / "data"
        / "splits"
        / "split-0"
        / "autotagging-validation.tsv",
        [tuple(crossing)],
    )
    with pytest.raises(JamendoValidationError, match="artist 11 crosses"):
        load_jamendo_context(
            metadata,
            audio,
            state,
            production=False,
            expected_tracks=3,
            expected_archives=1,
            fold_indices=(0,),
        )


def test_unknown_license_authority_is_rejected(tmp_path):
    metadata, audio, state = make_fixture(tmp_path)
    path = metadata / "audio_licenses.txt"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "http://creativecommons.org/licenses/by-nc-sa/3.0/",
            "https://example.com/unlicensed",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(JamendoValidationError, match="license authority"):
        load_jamendo_context(
            metadata,
            audio,
            state,
            production=False,
            expected_tracks=3,
            expected_archives=1,
            fold_indices=(0,),
        )


def test_by_nc_nd_license_forbids_commercial_use_and_derivatives(tmp_path):
    metadata, audio, state = make_fixture(tmp_path)
    path = metadata / "audio_licenses.txt"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "Attribution-Non-Commercial-Share-Alike license: "
            "http://creativecommons.org/licenses/by-nc-sa/3.0/",
            "Attribution-Non-Commercial-No-Derivatives license: "
            "http://creativecommons.org/licenses/by-nc-nd/3.0/",
            1,
        ),
        encoding="utf-8",
    )
    context = load_jamendo_context(
        metadata,
        audio,
        state,
        production=False,
        expected_tracks=3,
        expected_archives=1,
        fold_indices=(0,),
        verify_audio_hashes=True,
    )
    assert context.tracks[0].license.permits_commercial_use is False
    assert context.tracks[0].license.permits_derivatives is False


@pytest.mark.parametrize(
    "value", ("../x.mp3", "https://example.com/x.mp3", "C:/x.mp3", "x\\y.mp3")
)
def test_unsafe_paths_are_rejected(value):
    with pytest.raises(JamendoValidationError):
        safe_relative_path(value)


def test_audio_symlink_is_rejected_when_supported(tmp_path):
    metadata, audio, state = make_fixture(tmp_path)
    source = audio / "01" / "actual.mp3"
    source.write_bytes((audio / "01" / "1.mp3").read_bytes())
    (audio / "01" / "1.mp3").unlink()
    try:
        (audio / "01" / "1.mp3").symlink_to(source)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(JamendoValidationError, match="symlink"):
        load_jamendo_context(
            metadata,
            audio,
            state,
            production=False,
            expected_tracks=3,
            expected_archives=1,
            fold_indices=(0,),
        )
