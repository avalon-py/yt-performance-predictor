"""Aturan video_id di gerbang: baris test yang pernah dilatih model live/anchor dikeluarkan menurut id,
bukan lewat pagar waktu. Ditambahkan setelah benchmark VM: baris yang disisipkan mundur di waktu menggeser
awal test kandidat ke sebelum train_end live walau live tidak melihatnya, dan pagar waktu menolak seluruh
perbandingan.

    python -m pytest tests/early_fusion/test_gate_unseen.py -q
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from early_fusion import live_bundle as lb
from early_fusion import promotion as pm
from early_fusion.promotion import sha256_file
from tests.early_fusion.test_retrain import (FakeDeps, REPO, gate, gated, quiet,  # noqa: F401
                                             torch_or_stub)


# ------------------------------------------------------------------ fungsi murni
def test_unseen_mask_excludes_only_ids_a_model_trained_on():
    ids = ["a", "b", "c", "d", "e"]
    assert pm.unseen_mask(ids).all()
    assert pm.unseen_mask(ids, None, set()).all()                         # None/kosong diabaikan
    assert pm.unseen_mask(ids, {"b", "x"}).tolist() == [True, False, True, True, True]
    assert pm.unseen_mask(ids, {"b"}, ["d", "e"]).tolist() == [True, False, True, False, False]   # gabungan
    assert pm.unseen_mask([], {"a"}).tolist() == []


def test_insufficient_evidence_decision_shape():
    d = pm.insufficient_evidence(versioned_path="x.pt", n_test=1000, n_evaluated=120, min_unseen=300,
                                 extra={"run_id": "r"})
    assert d["promoted"] is False and d["reason"] == pm.REASON_INSUFFICIENT and d["new_spearman"] is None
    assert (d["n_test"], d["n_evaluated"], d["min_unseen"], d["run_id"]) == (1000, 120, 300, "r")
    assert pm.MIN_UNSEEN == 300


# ------------------------------------------------------------------ live_bundle: berkas split
def _split(train, val, test, h="hash1"):
    return {"snapshot_hash": h, "train_ids": train, "val_ids": val, "test_ids": test}


def _bundle(path, payload=b"w", **side):
    path.write_bytes(payload)
    lb.sidecar_path(path).write_text(json.dumps({"sha256": sha256_file(path), "train_end": "2026-02-01", **side}))
    return path


def test_bootstrap_writes_compact_split_for_latest_and_anchor_and_is_idempotent(tmp_path):
    src = tmp_path / "v1.pt"; src.write_bytes(b"v1")
    sj = tmp_path / "split.json"; sj.write_text(json.dumps(_split(["a", "b"], ["c"], ["d", "e"])))
    res = lb.bootstrap(src, tmp_path / "b", train_end="t", snapshot_n_total=5, split_json=sj)
    assert res == {"m6_latest": "dibuat + split", "m6_anchor": "dibuat + split"}
    for n in ("m6_latest", "m6_anchor"):
        data = json.loads(lb.split_path(tmp_path / "b" / f"{n}.pt").read_text())
        assert data == {"snapshot_hash": "hash1", "train_ids": ["a", "b"], "val_ids": ["c"]}     # id test tidak ikut
        assert lb.read_seen_ids(tmp_path / "b" / f"{n}.pt") == {"a", "b", "c"}
    assert lb.bootstrap(src, tmp_path / "b", train_end="t", snapshot_n_total=5, split_json=sj) == {
        "m6_latest": "sudah ada (identik)", "m6_anchor": "sudah ada (identik)"}
    # bundle yang sudah di-bootstrap tanpa split (kasus VM): menjalankan ulang dengan --split-json melengkapinya
    lb.split_path(tmp_path / "b" / "m6_latest.pt").unlink()
    again = lb.bootstrap(src, tmp_path / "b", train_end="t", snapshot_n_total=5, split_json=sj)
    assert again["m6_latest"] == "sudah ada (identik) + split ditulis" and again["m6_anchor"] == "sudah ada (identik)"
    assert lb.status(tmp_path / "b")["m6_latest"]["has_split"] is True


def test_read_seen_ids_is_none_without_a_split_file(tmp_path):
    pt = tmp_path / "x.pt"; pt.write_bytes(b"1")
    assert lb.read_seen_ids(pt) is None and lb.status(tmp_path)  # status tidak gagal


def test_promote_writes_split_rotates_it_and_never_leaves_a_stale_one(tmp_path):
    b = tmp_path / "bundles"
    sj1 = tmp_path / "s1.json"; sj1.write_text(json.dumps(_split(["a"], ["b"], ["c"], "h1")))
    sj2 = tmp_path / "s2.json"; sj2.write_text(json.dumps(_split(["a", "c"], ["b"], ["d"], "h2")))
    lb.promote(_bundle(tmp_path / "c1.pt", b"one"), b, split_json=sj1)
    assert lb.read_seen_ids(b / "m6_latest.pt") == {"a", "b"} and not lb.split_path(b / "m6_prev.pt").exists()
    lb.promote(_bundle(tmp_path / "c2.pt", b"two"), b, split_json=sj2)
    assert lb.read_seen_ids(b / "m6_latest.pt") == {"a", "b", "c"}            # milik kandidat kedua
    assert lb.read_seen_ids(b / "m6_prev.pt") == {"a", "b"}                    # split live lama ikut ke prev
    # promosi tanpa split: split latest lama (milik model yang diganti) HARUS hilang, bukan dibiarkan salah
    lb.promote(_bundle(tmp_path / "c3.pt", b"three"), b)
    assert lb.read_seen_ids(b / "m6_latest.pt") is None
    assert lb.read_seen_ids(b / "m6_prev.pt") == {"a", "b", "c"}


# ------------------------------------------------------------------ stage_gate dengan aturan video_id
def _test_ids(g):
    s = pd.read_parquet(g.run.snapshot_path, columns=["video_id", "split"])
    return s.loc[s["split"] == "test", "video_id"].tolist(), s.loc[s["split"] != "test", "video_id"].tolist()


def _write_live_split(g, seen_test):
    test, rest = _test_ids(g)
    lb.split_path(g.live).write_text(json.dumps(
        {"snapshot_hash": "legacy", "train_ids": rest + list(seen_test), "val_ids": []}))
    return test


def test_gate_excludes_rows_live_trained_on_even_when_the_timestamp_guard_would_refuse(gated, monkeypatch):
    """Skenario benchmark VM: train_end live LEBIH BARU dari awal test kandidat, tetapi live hanya melihat
    sebagian kecil baris test menurut id."""
    monkeypatch.setattr(pm, "MIN_UNSEEN", 5)
    gated.deps.live_meta = {"train_end": "2999-01-01 00:00:00+00:00"}
    with pytest.raises(SystemExit, match="tidak adil"):                    # tanpa daftar id: pagar waktu menolak (lama)
        gate(gated)
    test, _ = _test_ids(gated)
    _write_live_split(gated, test[:3])                                      # live melihat 3 baris test
    d = gate(gated)
    assert d["exclusion_rule"] == "video_id" and d["n_test"] == len(test)
    assert d["n_excluded_seen"] == 3 and d["n_evaluated"] == len(test) - 3
    assert d["reason"] in (pm.REASON_PROMOTE, pm.REASON_REJECT, pm.REASON_HARD_GATE)       # keputusan nyata, bukan SystemExit
    row = json.loads(gated.log.read_text().splitlines()[-1])
    assert row["exclusion_rule"] == "video_id" and row["n_excluded_seen"] == 3


def test_gate_without_any_split_files_keeps_the_strict_timestamp_rule(gated):
    d = gate(gated)                                                         # live_meta bawaan: train_end lama
    assert d["exclusion_rule"] == "timestamp" and d["n_excluded_seen"] == 0 and d["n_evaluated"] == d["n_test"]


def test_gate_refuses_to_promote_on_too_little_unseen_evidence(gated, monkeypatch, tmp_path):
    monkeypatch.setattr(pm, "MIN_UNSEEN", 8)
    test, _ = _test_ids(gated)
    _write_live_split(gated, test[:-2])                                     # live melihat hampir semua baris test
    d = gate(gated, promote=True, bundles_dir=tmp_path / "bundles")
    assert d["promoted"] is False and d["reason"] == pm.REASON_INSUFFICIENT and d["new_spearman"] is None
    assert d["n_evaluated"] == 2 and d["min_unseen"] == 8 and d["applied"] is False
    assert not (tmp_path / "bundles").exists()                             # tidak ada yang disalin
    assert json.loads(gated.log.read_text().splitlines()[-1])["reason"] == pm.REASON_INSUFFICIENT
    assert gated.run.stage_info("gate")["promoted"] is False               # bukan kegagalan task


def test_gate_applies_the_anchor_split_too(gated, monkeypatch):
    monkeypatch.setattr(pm, "MIN_UNSEEN", 5)
    gated.deps = type("D", (FakeDeps,), {"score": lambda self, path, df, idx, store, device: (
        (df["target"].values[idx], {"train_end": "2999-01-01"}) if Path(path).name.startswith("anchor")
        else FakeDeps.score(self, path, df, idx, store, device))})()
    anchor = gated.live.parent / "anchor_v1.pt"; anchor.write_bytes(b"a")
    test, rest = _test_ids(gated)
    lb.split_path(anchor).write_text(json.dumps({"snapshot_hash": "x", "train_ids": rest + test[:2], "val_ids": []}))
    gated.deps.live_meta = {"train_end": "2000-01-01 00:00:00+00:00"}
    d = gate(gated, anchor_path=anchor)                                    # train_end anchor "2999" tidak dipakai: ada daftar id
    assert d["exclusion_rule"] == "video_id" and d["n_excluded_seen"] == 2 and d["anchor_sha256"] == sha256_file(anchor)


def test_first_promotion_has_no_exclusions(gated):
    d = gate(gated, live_path=None, first_promotion=True)
    assert d["exclusion_rule"] == "timestamp" and d["n_excluded_seen"] == 0 and d["promoted"] is True


def test_promotion_through_the_gate_also_writes_the_candidate_split(gated, tmp_path, monkeypatch):
    b = tmp_path / "bundles"
    d = gate(gated, promote=True, bundles_dir=b)
    assert d["applied"] is True
    seen = lb.read_seen_ids(b / "m6_latest.pt")
    run_split = json.loads(gated.run.split_path.read_text())
    assert seen == set(run_split["train_ids"]) | set(run_split["val_ids"])
    assert not set(run_split["test_ids"]) & seen                           # baris test kandidat tidak "dilihat"
    # live berikutnya = kandidat ini; kandidat baru dibandingkan setelah baris itu dikeluarkan menurut id
    monkeypatch.setattr(pm, "MIN_UNSEEN", 1)
    again = gate(gated, live_path=b / "m6_latest.pt", anchor_path=None)
    assert again["exclusion_rule"] == "video_id" and again["n_excluded_seen"] == 0   # test-nya sendiri belum dilihat
