"""Fetch a processed source + its result from Drive for local debugging."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drive_sync as ds

svc = ds.get_service()
folders = ds.ensure_layout(svc)
for folder_key in ("processed", "done", "failed", "inbox"):
    fid = folders[folder_key]
    res = (
        svc.files()
        .list(q=f"'{fid}' in parents and trashed = false", fields="files(id, name, size)")
        .execute()
        .get("files", [])
    )
    for f in res:
        print(f"{folder_key}: {f['name']} ({int(f.get('size', 0)) / 1e6:.0f} MB) id={f['id']}")

if len(sys.argv) > 2:
    ds.download(svc, sys.argv[1], sys.argv[2])
    print("downloaded", sys.argv[2])
