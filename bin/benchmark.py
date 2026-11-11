"""Matched, seeded benchmarks for the local MiniMax-H3-Image HTTP server."""
import argparse
import os
import base64
import hashlib
import json
from pathlib import Path
import struct
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(os.environ.get("H3_ROOT", Path(__file__).resolve().parents[1]))
EXAMPLES = ROOT / "examples"
EDIT_IMAGE = EXAMPLES / "reference1.png"
EDIT_PROMPT = EXAMPLES / "revised_prompt.edit.txt"
EDIT_WARMUP = "Keep the two people and the scene; change the lamps to warm amber."
if not (EDIT_IMAGE.is_file() and EDIT_PROMPT.is_file()):
    EDIT_IMAGE = EXAMPLES / "edit-reference.png"
    EDIT_PROMPT = EXAMPLES / "edit-prompt.txt"
    EDIT_WARMUP = "Keep the composition and make the background a lighter blue."

CASES = {
    "smoke": ("1024x1024", 94, "Photograph of a single ripe orange with a curled leaf on a plain ivory table, soft daylight.", False),
    "product": ("2048x2048", 731, "Square editorial studio photograph of a dark cobalt glass perfume bottle on pale travertine. The centered cream label reads exactly 'NIGHT TIDE' in large engraved serif letters and 'EAU DE PARFUM' in small capitals beneath. Dappled blue caustics, crisp glass edges, fine pores in the stone, dark reflections through the translucent bottle, restrained palette and generous negative space. No other text.", False),
    "portrait": ("2048x2048", 1234, "An intimate natural-light portrait of an elderly woman in a botanical conservatory, reading a small red clothbound book. Warm brown skin with visible freckles and fine wrinkles, silver curls lit from behind, intricate indigo knit cardigan. Soft fern fronds out of focus in the foreground; detailed hands and realistic skin texture. A candid 85mm color photograph with gentle shadows and no lettering.", False),
    "edit": ("2048x2048", 42, EDIT_PROMPT, True),
    "native-4k": ("4096x4096", 802, "An overhead editorial photograph of an oceanographer's desk: a coiled orange rope, silver calipers, a pale-blue sea chart with fine contour lines, a black-and-white photograph of a lighthouse, a few tiny shells and a weathered brass compass. On a pale notecard write 'NORTH STAR' in neat bold navy capitals. One shaft of late afternoon sunlight, precise paper fibers and metal scratches, realistic soft shadows, a carefully balanced square composition.", False),
}
WARMUP = "A quiet wood workshop at dawn, oak shavings curling in clear window light, realistic documentary photography, subtle neutral tones."


def form(fields, reference):
    boundary = "h3_" + uuid4().hex
    out = []
    for name, value in fields.items():
        out.extend((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode(),
                    (json.dumps(value) if isinstance(value, (dict, list)) else str(value)).encode(), b"\r\n"))
    out.extend((f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"reference1.png\"\r\n"
                "Content-Type: image/png\r\n\r\n".encode(), reference.read_bytes(),
                f"\r\n--{boundary}--\r\n".encode()))
    return b"".join(out), "multipart/form-data; boundary=" + boundary


def make_request(args, size, seed, prompt, edit, name, directory, compression=None, *, reference_image=None):
    payload = {
        "model": "MiniMaxAI/MiniMax-H3-Image", "prompt": prompt, "size": size,
        "seed": seed, "generator_device": "cpu", "num_inference_steps": 6,
        "guidance_scale": 1, "quality": args.quality, "response_format": "b64_json",
        "output_format": "png", "output_compression": args.compression if compression is None else compression,
    }
    if args.opt:
        payload["h3_opt"] = args.opt
    if name == "smoke":
        payload["min_generation_area"] = 1024
    if args.stage_profile:
        payload["perf_dump_path"] = str((directory / f"{name}.perf.json").resolve())
    if args.attention:
        if edit:
            raise ValueError("Use a server-wide attention override for edit measurements.")
        payload["attention_backend_override"] = args.attention
    if edit:
        body, content_type = form(payload, reference_image or EDIT_IMAGE)
    else:
        body, content_type = json.dumps(payload).encode(), "application/json"
    endpoint = args.url.rstrip("/") + "/v1/images/" + ("edits" if edit else "generations")
    req = Request(endpoint, body, headers={"Content-Type": content_type})
    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=args.timeout) as res:
            raw = res.read()
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read(5000).decode(errors='replace')}") from exc
    dt = time.perf_counter() - t0
    result = json.loads(raw)
    png = base64.b64decode(result["data"][0]["b64_json"], validate=True)
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Server did not return a PNG")
    dimensions = struct.unpack(">II", png[16:24])
    if dimensions != tuple(map(int, size.split("x"))):
        raise RuntimeError(f"Image dimensions {dimensions} did not match {size}")
    return png, {"wall_seconds": round(dt, 4), "image_bytes": len(png),
                 "sha256": hashlib.sha256(png).hexdigest(), "size": size, "seed": seed,
                 "task": "edit" if edit else "text-to-image", "quality": args.quality,
                 "output_compression": payload["output_compression"], "attention_override": args.attention, "h3_opt": args.opt}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--url", default="http://127.0.0.1:30000")
    p.add_argument("--output", type=Path, default=ROOT / "runs")
    p.add_argument("--quality", choices=["exact", "lossless"], default="lossless")
    p.add_argument("--cases", nargs="+", choices=CASES, default=["product", "portrait", "edit", "native-4k"])
    p.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--warmup-matched", action="store_true",
                   help="also prewarm each exact prompt/shape before timed trials")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--seed-offset", type=int, default=0,
                   help="add an offset to all case seeds for additional accuracy trials")
    p.add_argument("--compression", type=int, default=1)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--opt", type=json.loads, default=None)
    p.add_argument("--attention", default=None)
    p.add_argument("--stage-profile", action="store_true")
    args = p.parse_args()
    path = args.output / args.label
    path.mkdir(parents=True, exist_ok=True)
    seen = set()
    for key in args.cases:
        size, seed, prompt, edit = CASES[key]
        seed += args.seed_offset
        prompt = prompt.read_text().strip() if isinstance(prompt, Path) else prompt
        if args.warmup and (size, edit) not in seen:
            warm_prompt = EDIT_WARMUP if edit else WARMUP
            _, record = make_request(args, size, 19, warm_prompt, edit, f"warm-{key}", path)
            print(json.dumps({"label": args.label, "case": key, "warmup": True, **record}), flush=True)
            seen.add((size, edit))
        if args.warmup_matched:
            _, record = make_request(args, size, seed, prompt, edit, f"matched-{key}", path)
            print(json.dumps({"label": args.label, "case": key,
                              "matched_warmup": True, **record}), flush=True)
        for trial in range(1, args.repeat + 1):
            name = key if trial == 1 else f"{key}-{trial}"
            png, record = make_request(args, size, seed, prompt, edit, name, path)
            (path / (name + ".png")).write_bytes(png)
            row = {"label": args.label, "case": key, "trial": trial, **record, "prompt": prompt}
            if edit:
                row["reference_sha256"] = hashlib.sha256(EDIT_IMAGE.read_bytes()).hexdigest()
            with (path / "results.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "prompt"}), flush=True)


if __name__ == "__main__":
    main()
