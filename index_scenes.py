"""Build a local scene index from LeRobot exports or directories of first-frame images."""
import argparse
import base64
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

TASK_PATTERNS = {
    "pick_place": (r"单臂简单抓放|简单抓放|抓取水果|水果抓放", r"\b(?:plate|fruit pick[\s_-]*and[\s_-]*place)\b"),
    "cosmetics": (r"化妆品收纳|化妆品整理|透明收纳盒", r"\b(?:cosmetics?|compartment|organizer|organiser|lipstick|lip gloss|makeup|eyeshadow|lotion pump bottle|highlighter compact|contour compact)\b"),
    "sweeping": (r"垃圾清扫|扫垃圾|扫把|簸箕|纸团", r"\b(?:sweep(?:ing)?|dustpan|paper balls?|trash cleanup|garbage cleanup)\b"),
}


def classify_task(text):
    matches = [key for key, patterns in TASK_PATTERNS.items()
               if any(re.search(pattern, text, re.I) for pattern in patterns)]
    return matches[0] if len(matches) == 1 else None


def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def source_arg(value):
    alias, separator, path = value.partition("=")
    if not separator or not re.fullmatch(r"[a-zA-Z0-9_-]+", alias) or not path:
        raise argparse.ArgumentTypeError("Use a stable dataset alias, e.g. official=/path/to/dataset")
    return alias, Path(path)


def camera_key(info, requested):
    keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    if requested:
        if requested not in keys:
            raise ValueError(f"Camera {requested!r} not found. Available: {', '.join(keys)}")
        return requested
    rgb = [key for key in keys if "depth" not in key.lower() and "infra" not in key.lower()]
    if not rgb:
        raise ValueError("No RGB video camera. For image datasets, export the first frames to an image directory.")
    for pattern in (r"head|top|overhead", r"front|main|primary"):
        matches = [key for key in rgb if re.search(pattern, key, re.I)]
        if len(matches) == 1:
            return matches[0]
    if len(rgb) == 1:
        return rgb[0]
    raise ValueError(f"Specify --camera. Available: {', '.join(rgb)}")


def lerobot_entries(root, requested_camera):
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8-sig"))
    version = str(info.get("codebase_version", "")).lstrip("v")
    template = info.get("video_path")
    if not template or info.get("storage_format") not in (None, "parquet"):
        raise ValueError("Expected a LeRobot Parquet/MP4 export or a directory of first-frame images.")
    camera = camera_key(info, requested_camera)
    if version.startswith("2."):
        chunk_size = info.get("chunks_size") or 1000
        rows = json_lines(root / "meta/episodes.jsonl")
        for row in rows:
            index = row["episode_index"]
            path = template.format(video_key=camera, episode_index=index, episode_chunk=index // chunk_size)
            yield str(index), row, root / path, 0.0, camera
    elif version.startswith("3."):
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ValueError("LeRobot v3 requires pyarrow: python -m pip install pyarrow") from error
        shards = sorted((root / "meta/episodes").rglob("*.parquet"))
        if not shards:
            raise ValueError("No episode Parquet files found under meta/episodes.")
        prefix = f"videos/{camera}/"
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            names = parquet.schema_arrow.names
            columns = [name for name in names if name.startswith(prefix) or name in
                       {"episode_index", "length", "tasks", "task", "task_category", "scene_id", "scene_name"}]
            for row in parquet.read(columns=columns).to_pylist():
                path = template.format(video_key=camera, chunk_index=row[prefix + "chunk_index"],
                                       file_index=row[prefix + "file_index"])
                start = float(row[prefix + "from_timestamp"])
                end = row.get(prefix + "to_timestamp")
                if not math.isfinite(start) or start < 0 or (end is not None and float(end) <= start):
                    raise ValueError(f"Invalid video timestamp for Episode {row['episode_index']}.")
                yield str(row["episode_index"]), row, root / path, start, camera
    else:
        raise ValueError(f"Unsupported LeRobot codebase version: {version!r}")


def entries(root, camera):
    if (root / "meta/info.json").is_file():
        yield from lerobot_entries(root, camera)
        return
    paths = sorted(path for path in root.iterdir() if path.is_file() and
                   path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not paths:
        raise ValueError("Select the exact dataset root containing meta/info.json, or a folder of first-frame images.")
    for path in paths:
        yield path.name, {}, path, None, camera or "head"


def first_frame(path, timestamp, ffmpeg):
    if timestamp is None:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    if not path.is_file():
        raise ValueError("找不到此 Episode 对应的相机视频")
    result = subprocess.run([ffmpeg, "-v", "error", "-ss", str(timestamp), "-i", str(path),
                             "-map", "0:v:0", "-frames:v", "1", "-vf", "scale=480:-2",
                             "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
                            capture_output=True, timeout=90, check=True)
    if not result.stdout:
        raise ValueError("指定起始时间没有可解码的画面")
    with Image.open(io.BytesIO(result.stdout)) as image:
        return image.convert("RGB")


def make_visual(image, crop):
    preview = image.copy()
    preview.thumbnail((320, 240))
    stream = io.BytesIO()
    preview.save(stream, format="JPEG", quality=72)
    box = tuple(round(value * dimension) for value, dimension in zip(crop, image.size * 2))
    table = image.crop(box).resize((48, 36), Image.Resampling.BILINEAR)
    pixels = np.asarray(table, dtype=np.float32) / 255
    horizontal = np.diff(pixels, axis=1, append=pixels[:, -1:])
    vertical = np.diff(pixels, axis=0, append=pixels[-1:])
    features = np.concatenate((pixels * 0.65, horizontal * 0.8, vertical * 0.8), axis=2)
    return base64.b64encode(stream.getvalue()).decode("ascii"), features


def read_sources(sources, args):
    rows = []
    vectors = {}
    for source, alias, root in sources:
        root = root.resolve(strict=True)
        needs_video = (root / "meta/info.json").is_file()
        if needs_video and not args.ffmpeg:
            raise ValueError("Video extraction requires FFmpeg on PATH or --ffmpeg /path/to/ffmpeg.")
        for index, metadata, path, timestamp, camera in entries(root, args.camera):
            key = f"{alias}:{index}"
            tasks = metadata.get("tasks") or metadata.get("task") or []
            task = " / ".join(map(str, tasks)) if isinstance(tasks, list) else str(tasks)
            category_text = str(metadata.get("task_category") or "")
            task_type = (args.dataset_tasks.get(alias) or
                         (category_text if category_text in TASK_PATTERNS else classify_task(category_text)) or
                         classify_task(task))
            if not task_type and re.fullmatch(r"pick[\s_-]*(?:and[\s_-]*)?place", category_text.strip(), re.I):
                task_type = "pick_place"
            scene_id = metadata.get("scene_id")
            scene_name = metadata.get("scene_name")
            record = {"id": key, "dataset": alias, "episode": index, "source": source,
                      "task": task, "taskType": task_type, "camera": camera, "time": timestamp,
                      "group": None, "method": "review"}
            if task_type and (scene_id is not None or scene_name):
                record.update(group="label:" + str(scene_id if scene_id is not None else scene_name),
                              groupName=str(scene_name or scene_id), method="label")
            rows.append(record)
            try:
                path.resolve().relative_to(root)
                image = first_frame(path, timestamp, args.ffmpeg)
                record["thumbnail"], vectors[key] = make_visual(image, args.crop)
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                record["error"] = "首帧提取失败，请检查视频、时间戳或解码器。"
                print(f"{key}: {error}", file=sys.stderr)
            if len(rows) % 25 == 0:
                print(f"Indexed {len(rows)} episodes...", file=sys.stderr, flush=True)
    episode_ids = [record["id"] for record in rows]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Duplicate episode IDs inside a dataset. Check its metadata.")
    return rows, vectors


def assign_groups(rows, vectors, threshold):
    def distance(left, right):
        difference = np.abs(vectors[left["id"]] - vectors[right["id"]])
        # Emphasize changed areas so the static table does not dominate the comparison.
        flat = difference.mean(axis=2).ravel()
        top = max(1, flat.size // 10)
        return float(np.partition(flat, -top)[-top:].mean())

    explicit = {}
    for row in rows:
        if row["taskType"] and row["group"] and row["id"] in vectors:
            explicit.setdefault((row["taskType"], row["camera"], row["group"]), []).append(row)
    unmatched = {}
    for row in rows:
        if not row["taskType"] or row["group"] or row["method"] == "manual" or row["id"] not in vectors:
            continue
        candidates = sorted((min(distance(row, item) for item in group), key, group[0])
                            for (task_type, camera, key), group in explicit.items()
                            if task_type == row["taskType"] and camera == row["camera"])
        if candidates and candidates[0][0] <= threshold:
            if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < threshold * 0.2:
                continue
            _, key, sample = candidates[0]
            row.update(group=key, groupName=sample["groupName"], method="match")
        else:
            unmatched.setdefault((row["taskType"], row["camera"]), []).append(row)

    for samples in unmatched.values():
        groups = []
        for row in samples:
            eligible = []
            for group in groups:
                largest = 0.0
                for member in group:
                    largest = max(largest, distance(row, member))
                    if largest > threshold:
                        break
                if largest <= threshold:
                    eligible.append((largest, group[0]["id"], group))
            eligible.sort(key=lambda choice: (choice[0], choice[1]))
            if len(eligible) > 1 and eligible[1][0] - eligible[0][0] < threshold * 0.2:
                continue
            if eligible:
                eligible[0][2].append(row)
            else:
                groups.append([row])
        for group in groups:
            if len(group) < 2:
                continue
            digest = hashlib.sha256(group[0]["id"].encode()).hexdigest()[:10]
            for row in group:
                row.update(group="visual:" + digest, method="visual")


def apply_manual_assignments(rows, path):
    saved = json.loads(path.read_text(encoding="utf-8-sig"))
    manual = {item["id"]: item for item in saved["episodes"] if item.get("method") == "manual"}
    for row in rows:
        item = manual.get(row["id"])
        if item is None:
            continue
        task_type = item.get("taskType")
        if task_type not in TASK_PATTERNS and not (task_type is None and "taskType" in item):
            continue
        group = item.get("group")
        if group is not None and (not isinstance(group, str) or not re.fullmatch(r"(label|visual|manual):.+", group)):
            raise ValueError(f"Invalid scene group for {row['id']}.")
        row.update(taskType=task_type, group=group if task_type else None,
                   groupName=str(item.get("groupName") or ""), method="manual")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", action="append", type=source_arg, default=[])
    parser.add_argument("--own", action="append", type=source_arg, default=[])
    parser.add_argument("--task", action="append", default=[], metavar="ALIAS=TASK",
                        help="Single-task input: pick_place, cosmetics, or sweeping")
    parser.add_argument("--camera", help="Exact LeRobot camera key; otherwise auto-detect")
    parser.add_argument("--crop", nargs=4, type=float, default=(0.05, 0.14, 0.95, 0.96),
                        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--threshold", type=float, default=0.035, help="Smaller value groups more conservatively")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--assignments", type=Path, help="Previously exported scene-assignments.json")
    parser.add_argument("--output", type=Path, default=Path("scene-index.json"))
    args = parser.parse_args()
    sources = [("official", *entry) for entry in args.official] + [("own", *entry) for entry in args.own]
    if not sources or len({alias for _, alias, _ in sources}) != len(sources):
        parser.error("Provide at least one --official or --own input and give each dataset a distinct alias.")
    aliases = {alias for _, alias, _ in sources}
    args.dataset_tasks = {}
    for entry in args.task:
        alias, _, task_type = entry.partition("=")
        if alias not in aliases or task_type not in TASK_PATTERNS or alias in args.dataset_tasks:
            parser.error("--task must be a unique ALIAS=pick_place, ALIAS=cosmetics, or ALIAS=sweeping.")
        args.dataset_tasks[alias] = task_type
    left, top, right, bottom = args.crop
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1 and 0 < args.threshold <= 1):
        parser.error("Crop coordinates must be ordered and all parameters in the supported 0–1 range.")
    try:
        if any(args.output.resolve().is_relative_to(root.resolve()) for _, _, root in sources):
            raise ValueError("Choose an output outside the dataset folders.")
        rows, vectors = read_sources(sources, args)
        if not rows:
            raise ValueError("No Episodes were indexed.")
        if args.assignments:
            apply_manual_assignments(rows, args.assignments)
        assign_groups(rows, vectors, args.threshold)
        index = {"schemaVersion": 2, "crop": list(args.crop), "threshold": args.threshold, "episodes": rows}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(index, stream, ensure_ascii=False, separators=(",", ":"))
                stream.close()
                temporary.replace(args.output)
            finally:
                stream.close()
                temporary.unlink(missing_ok=True)
        print(f"{len(rows)} episodes | {sum(row['group'] is not None for row in rows)} grouped | "
              f"{sum('error' in row for row in rows)} frame errors | {args.output}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Indexing failed: {error}\n")


if __name__ == "__main__":
    main()
