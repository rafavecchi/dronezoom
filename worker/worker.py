"""DroneZoom worker: poll Drive/DroneZoom/inbox for clips, run the
pipeline, upload results to done/, file sources into processed/ (or
failed/ with an error note). Designed to run forever at logon."""

import datetime
import os
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LOG = os.path.join(HERE, "worker.log")
POLL_SEC = 60


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def handle(svc, folders, item):
    import drive_sync as ds
    from process_clip import process

    name = item["name"]
    size_mb = int(item.get("size", 0)) / 1e6
    log(f"job: {name} ({size_mb:.0f} MB)")
    with tempfile.TemporaryDirectory(prefix="dronezoom-") as tmp:
        src = os.path.join(tmp, name)
        out_name = f"{os.path.splitext(name)[0]}-dronezoom.mp4"
        out = os.path.join(tmp, out_name)
        log("  downloading…")
        ds.download(svc, item["id"], src)
        t0 = time.time()
        stats = process(src, out, log=log)
        log(
            f"  done in {time.time() - t0:.0f}s — blob {stats['blob_found_pct']:.0f}%, "
            f"{stats['out_bytes'] / 1e6:.0f} MB"
        )
        log("  uploading result…")
        ds.upload(svc, folders["done"], out, out_name)
        ds.move(svc, item["id"], folders["inbox"], folders["processed"])
    log(f"job complete: {out_name} -> DroneZoom/done")


def main():
    # single-instance lock: duplicate workers double-process jobs. The
    # socket releases automatically if the process dies.
    import socket

    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", 47821))
        lock.listen(1)
    except OSError:
        print("another DroneZoom worker is already running — exiting")
        return

    log("worker starting")
    svc = None
    folders = None
    while True:
        try:
            if svc is None:
                import drive_sync as ds

                svc = ds.get_service()
                folders = ds.ensure_layout(svc)
                log("Drive connected; watching DroneZoom/inbox")
            import drive_sync as ds

            for item in ds.list_inbox(svc, folders["inbox"]):
                try:
                    handle(svc, folders, item)
                except Exception as e:
                    log(f"job FAILED: {item['name']}: {e}")
                    log(traceback.format_exc())
                    try:
                        ds.upload_text(
                            svc,
                            folders["failed"],
                            f"{item['name']}.error.txt",
                            f"{e}\n\n{traceback.format_exc()}",
                        )
                        ds.move(svc, item["id"], folders["inbox"], folders["failed"])
                    except Exception:
                        log("  (could not file the failure in Drive)")
        except FileNotFoundError as e:
            log(f"waiting for setup: {e}")
            svc = None
            time.sleep(300)
            continue
        except Exception as e:
            log(f"poll error ({e}); reconnecting next cycle")
            svc = None
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
