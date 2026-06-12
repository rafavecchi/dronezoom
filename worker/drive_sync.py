"""Google Drive plumbing for the worker: auth, folder layout, poll,
download, upload, move. Token persists in token.json after a one-time
browser consent on this PC."""

import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

HERE = os.path.dirname(os.path.abspath(__file__))
SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN = os.path.join(HERE, "token.json")
CLIENT_SECRET = os.path.join(HERE, "client_secret.json")

FOLDER_MIME = "application/vnd.google-apps.folder"


def get_service():
    creds = None
    if os.path.exists(TOKEN):
        creds = Credentials.from_authorized_user_file(TOKEN, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CLIENT_SECRET):
                raise FileNotFoundError(
                    f"missing {CLIENT_SECRET} — create a Desktop-app OAuth client in "
                    "the Google Cloud console and save its JSON there"
                )
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN, "w") as f:
            f.write(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _find_or_create(svc, name, parent=None):
    q = f"name = '{name}' and mimeType = '{FOLDER_MIME}' and trashed = false"
    if parent:
        q += f" and '{parent}' in parents"
    res = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
    if res:
        return res[0]["id"]
    meta = {"name": name, "mimeType": FOLDER_MIME}
    if parent:
        meta["parents"] = [parent]
    return svc.files().create(body=meta, fields="id").execute()["id"]


def ensure_layout(svc):
    root = _find_or_create(svc, "DroneZoom")
    return {
        "root": root,
        "inbox": _find_or_create(svc, "inbox", root),
        "done": _find_or_create(svc, "done", root),
        "processed": _find_or_create(svc, "processed", root),
        "failed": _find_or_create(svc, "failed", root),
    }


def list_inbox(svc, inbox_id):
    q = (
        f"'{inbox_id}' in parents and trashed = false "
        "and (mimeType contains 'video/' or name contains '.mp4' or name contains '.MP4')"
    )
    res = (
        svc.files()
        .list(q=q, fields="files(id, name, size, modifiedTime)", orderBy="modifiedTime")
        .execute()
    )
    return res.get("files", [])


def download(svc, file_id, dest_path):
    req = svc.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as f:
        dl = MediaIoBaseDownload(f, req, chunksize=16 * 1024 * 1024)
        done = False
        while not done:
            _, done = dl.next_chunk()


def upload(svc, folder_id, local_path, name, mime="video/mp4"):
    media = MediaFileUpload(local_path, mimetype=mime, resumable=True)
    return (
        svc.files()
        .create(body={"name": name, "parents": [folder_id]}, media_body=media, fields="id")
        .execute()["id"]
    )


def upload_text(svc, folder_id, name, text):
    import io

    from googleapiclient.http import MediaIoBaseUpload

    media = MediaIoBaseUpload(io.BytesIO(text.encode()), mimetype="text/plain")
    svc.files().create(
        body={"name": name, "parents": [folder_id]}, media_body=media, fields="id"
    ).execute()


def move(svc, file_id, from_id, to_id):
    svc.files().update(fileId=file_id, addParents=to_id, removeParents=from_id).execute()
