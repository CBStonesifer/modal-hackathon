# Modal — Hackathon Reference

Compiled from https://modal.com/docs/guide. Lens: egoverse video dataset — curation, diversity scoring, success/failure classification.

---

## 0. The 60-second mental model

- `modal.App` = deployable namespace. `@app.function` / `@app.cls` = independently autoscaling units.
- `modal.Image` = the container filesystem, built by chaining methods. No Dockerfile needed.
- `modal.Volume` = persistent shared filesystem. Model weights, features, outputs.
- `.map()` = fan out over thousands of inputs. This is the whole point for dataset work.
- `modal deploy` = stable public URL. `modal run` = ephemeral one-shot. `modal serve` = live-reload dev.

```bash
pip install modal && modal setup
modal run pipeline.py          # iterate
modal deploy pipeline.py       # ship, get URL
modal shell pipeline.py::fn    # debug inside the image
```

---

## 1. Running inference — the pattern that matters

**Never load a model inside the function body.** Use `@app.cls` + `@modal.enter()` so weights load once per container and are reused across every call routed to it.

```python
import modal

app = modal.App("ego-inference")
MODEL_DIR = "/models"
weights = modal.Volume.from_name("model-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})   # fast HF weight transfer
)

@app.cls(gpu="L40S", image=image, volumes={MODEL_DIR: weights}, scaledown_window=300)
class Scorer:
    @modal.enter()                     # runs once per container boot
    def load(self):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(f"{MODEL_DIR}/my-model").cuda()

    @modal.method()                    # only decorated methods are remotely callable
    def score(self, clip_path: str) -> float:
        return float(self.model(load_clip(clip_path)).item())

    @modal.exit()                      # 30s grace period, also fires on preemption
    def cleanup(self):
        pass
```

Call it:

```python
Scorer().score.remote(path)              # blocking single call
Scorer().score.map(paths)                # fan out (this is what you want)
Scorer().score.spawn(path)               # fire-and-forget -> FunctionCall handle
```

From another app / process:

```python
Scorer = modal.Cls.from_name("ego-inference", "Scorer")
Scorer().score.remote(path)
```

**Gotchas**
- A container is not "warm" until every `@modal.enter()` finishes — no inputs route to it mid-load. Load multiple models concurrently with `ThreadPoolExecutor` inside one `enter()`.
- `__init__` / `__enter__` for setup is deprecated. Use `@modal.enter()`.
- Import GPU-only packages (`torch`) *inside* the function body if your laptop doesn't have them.
- A class with a `@modal.batched` method can't also have `@modal.method()`s.

### Model weights: Volume, not Image

Baking multi-GB weights into an Image bloats build/push and re-downloads on every rebuild. Download once into a Volume:

```python
@app.function(volumes={MODEL_DIR: weights}, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def download_model(repo_id: str):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, local_dir=f"{MODEL_DIR}/{repo_id}")
    weights.commit()
```

Run it once (`modal run app.py::download_model --repo-id ...`), then every container reads from the Volume.

### Throughput levers — pick the right one

| Lever | Use for | Code |
|---|---|---|
| `@modal.batched` | Stateless per-item inference (embeddings, frame classification) where inputs trickle in | `@modal.batched(max_batch_size=32, wait_ms=100)` |
| `@modal.concurrent` | Server-style processes that batch internally (vLLM), or I/O-bound work | `@modal.concurrent(max_inputs=96, target_inputs=80)` |
| `.map()` | You already have the full list of N items | `f.map(items)` |

```python
@app.cls(gpu="L4")
class Embedder:
    @modal.enter()
    def load(self): self.model = load_model()

    @modal.batched(max_batch_size=64, wait_ms=100)
    def embed(self, clips: list[str]) -> list[list[float]]:
        return self.model.encode(clips).tolist()   # len(out) MUST equal len(in)
```

Don't stack `@modal.batched` on a vLLM-style server — vLLM does its own continuous batching; feed it with `@modal.concurrent` instead.

### GPU selection

`T4` `L4` `A10` `L40S` `A100-40GB` `A100-80GB` `H100` (`H100!` pins it) `H200` `B200` `B300` `RTX-PRO-6000`

```python
gpu="A100"              # single
gpu="H100:2"            # 2 GPUs in one container
gpu=["H100", "L40S"]    # ordered fallback under shortage
```

For video feature extraction / embeddings: **start on T4 or L4.** Decode + S3 read is usually the bottleneck, not GPU compute. Step up to L40S (48GB) for mid-size VLMs, A100/H100 only for large VLMs. Max 8 GPUs/container (4 for A10).

Pricing (on-demand, per hour): T4 $0.59 · L4 $0.80 · A10 $1.10 · L40S $1.95 · A100-40GB $2.10 · A100-80GB $2.50 · H100 $3.95. CPU $0.0000131/core-sec, memory $0.00000222/GiB-sec.

### Cold starts

Container boot is ~1s; everything else is your init code.

```python
@app.cls(gpu="L4", min_containers=1, buffer_containers=2, scaledown_window=300)
```

- `scaledown_window` — idle seconds before shutdown (default 60, range 2s–20min).
- `min_containers` — always-warm floor. **Set this to 1 a few minutes before you demo**, otherwise the first judge click eats a cold start.
- `buffer_containers` — spare capacity kept live *while active* (bursts).

Adjust live without redeploying:

```python
modal.Function.from_name("app", "fn").update_autoscaler(min_containers=1)
```

### Memory snapshots — only if init is compute-bound

```python
@app.function(enable_memory_snapshot=True)          # CPU snapshot
```
```python
@app.cls(gpu="h100", enable_memory_snapshot=True,
         experimental_options={"enable_gpu_snapshot": True})   # alpha
class Llm:
    @modal.enter(snap=True)
    def init(self): ...
```

Split-phase trick — snapshot the CPU-side load, move to GPU after restore:

```python
@modal.enter(snap=True)
def load(self):  self.model = SentenceTransformer(path, device="cpu")
@modal.enter(snap=False)
def to_gpu(self): self.model.to("cuda")
```

3–10x faster boots when init is compute-heavy. **Won't help if you're just loading weights off disk** — that's storage-bound. Snapshots only generate on `modal deploy`, not `modal run`. Incompatible with multi-GPU. Calling `torch.cuda.is_available()` pre-snapshot can wedge CUDA into seeing zero devices.

---

## 2. Fan-out over the dataset

```python
f.map(items)                                # ordered results, blocking iterator
f.map(items, return_exceptions=True)        # failures come back as values, not a crash
f.map(items, order_outputs=False)           # faster drain if order doesn't matter
f.starmap([(a, b), ...])                    # unpacks tuples into positional args
f.for_each(items)                           # discards returns (write to Volume instead)
f.spawn_map(items)                          # non-blocking, returns FunctionCall
modal.FunctionCall.gather(fc1, fc2)
```

**`retries=modal.Retries(...)` + `.map(..., return_exceptions=True)` is the combination for any batch over ~1k items.** Without it, one corrupt video kills a run after thousands of successful completions.

Platform ceilings: 2,000 pending inputs unassigned · 25,000 total in flight · 1,000 concurrent per `.map()` invocation · **1,000,000 pending for `.spawn()`**. Exceeding → `Resource Exhausted`. A single `.map()` over 50k items is fine (Modal paces submission); manual `spawn` loops need chunking.

Don't wrap in Python's builtin `map()` — that runs sequentially.

### Preemption is the default (and that's good)

All Modal functions are preemptible by default — this *is* the spot discount, no flag needed. On preemption Modal restarts the function on the same input automatically. `nonpreemptible=True` costs 3x and **isn't available for GPU functions anyway**. Design for it: make your per-episode function idempotent and write partial results as you go.

### Reading the dataset from S3

```python
@app.function(
    gpu="L4",
    volumes={"/videos": modal.CloudBucketMount("ego-videos", secret=s3_creds, read_only=True)},
)
def score(key: str) -> float:
    return model(f"/videos/{key}")
```

`CloudBucketMount(bucket_name, secret=, oidc_auth_role_arn=, key_prefix=, read_only=)`. Supports S3/GCS/R2. `key_prefix="episodes/2024/"` mounts a subtree.

**Gotcha:** built on AWS Mountpoint — **no append mode, no seek+write**. Write outputs to a Volume, not back into the mount.

### Budget guardrails

`max_containers` caps peak concurrent spend rate. `timeout` caps worst-case cost of one stuck input. Those two knobs matter more than GPU choice.

```python
@app.function(gpu="L4", max_containers=100, timeout=1800,
              retries=modal.Retries(max_retries=3, backoff_coefficient=2.0))
```

Timeout defaults to 300s (range 1s–24h) and measures execution only, not queue wait. With `retries=3` and `timeout=1800`, worst case for one input is ~2h — plan wall-clock accordingly.

Region: leave `region=` unset. Broad regions cost 1.5x, narrow 1.75x. Only co-locate with your bucket if you've measured egress as the bottleneck. Return values >2MiB always route through `us-east` — another reason to write features to a Volume rather than returning them.

---

## 3. Storage

### Volume — features, checkpoints, outputs

```python
vol = modal.Volume.from_name("features", create_if_missing=True)

@app.function(volumes={"/data": vol})
def process(key: str):
    open(f"/data/{key}.npz", "wb").write(feats)   # MUST be under the mount path
    vol.commit()                                   # make visible to other containers
```

- Writing to `/foo.npz` instead of `/data/foo.npz` **silently writes to ephemeral disk and vanishes.** Most common Modal bug.
- Auto-commits every few seconds and on shutdown, but commit explicitly if a dashboard needs to see it now.
- `vol.reload()` in a long-lived container to pick up other containers' writes. Volume looks empty mid-reload; can't reload with files open.
- v1: 500k inode cap, degrades past ~50k files, ≤5 concurrent commits, last-write-wins. **For 20k+ episodes × multiple output files, use `version=2`.** Concurrent writes to *distinct* files are fine; to the *same* file are not.
- CLI: `modal volume ls/get/put NAME`. Dashboard download caps at 16MB — use the CLI for anything bigger.

### Dict — scores, status, small JSON

```python
scores = modal.Dict.from_name("episode-scores", create_if_missing=True)
scores["ep_0042"] = 0.83
dict(scores.items())
```

Perfect for per-episode confidence scores a web endpoint reads. `from_name(create_if_missing=True)` persists across deploys; `.ephemeral()` is throwaway only. Small values only — video files go on a Volume.

### Queue — cross-worker coordination

```python
with modal.Queue.ephemeral() as q, modal.Dict.ephemeral() as d:
    q.put_many(keys)
    batch = q.get_many(500, timeout=5)
    if "stop" in d: return          # shared kill switch
```

Only reach for this if you need dynamic work distribution or early termination ("stop once we have 5,000 clips above threshold"). Plain `.map()` is simpler and enough for score-everything-then-filter.

### Secrets

```bash
modal secret create hf-secret HF_TOKEN=...
```
```python
@app.function(secrets=[modal.Secret.from_name("hf-secret")])
def f(): os.environ["HF_TOKEN"]
```
Also `Secret.from_dict({...})`, `Secret.from_dotenv()`.

---

## 4. Images

```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .uv_pip_install("torch", "transformers")     # faster than pip_install
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("mypkg")            # local module by name
    .add_local_dir("assets", remote_path="/assets")
)
```

Also `.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12").entrypoint([])`, `.from_dockerfile()`, `.run_commands()`, `.run_function()`.

**Gotchas**
- Layers cache per method call; changing one invalidates it *and everything after*. Put volatile steps (your code) last.
- `add_local_*` mounts at container start, not baked into the image — fast iteration, no rebuild. Pass `copy=True` if a later `run_commands`/`run_function` must see those files.
- Modal auto-includes the module containing your decorated functions. Extra local packages need `add_local_python_source`.
- Force rebuild: `force_build=True`, or `MODAL_FORCE_BUILD=1`.
- Multi-file projects: `modal deploy -m src.app` and use relative imports.

---

## 5. Serving — dashboards and a URL to hand a judge

`modal deploy app.py` prints `https://<workspace>--<app>-<fn>.modal.run` and keeps running with no client attached. Iterate with `modal serve` (ephemeral, `-dev` suffix, dies on Ctrl-C), deploy once before judging.

### JSON endpoint

```python
@app.function(image=modal.Image.debian_slim().pip_install("fastapi[standard]"))
@modal.fastapi_endpoint(method="GET")
def scores():
    return dict(score_dict.items())
```

Renamed from `@modal.web_endpoint` — use `fastapi_endpoint`. Default method is GET.

### Full app + Gradio dashboard

```python
@app.function(image=image, max_containers=1)
@modal.asgi_app()
def ui():
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from gradio.routes import mount_gradio_app

    with gr.Blocks(title="Subset A vs B") as blocks:
        gr.Markdown("# Diversity comparison")

    web_app = FastAPI()
    web_app.mount("/videos", StaticFiles(directory="/data/videos"))
    return mount_gradio_app(app=web_app, blocks=blocks, path="/")
```

- `max_containers=1` for Gradio — otherwise session state splits across containers.
- Stack `@modal.concurrent(max_inputs=100)` above `@modal.asgi_app()` or every request gets its own container.

### Streamlit / any port

```python
@app.function()
@modal.concurrent(max_inputs=100)
@modal.web_server(8000)
def run():
    subprocess.Popen("streamlit run /root/app.py --server.port 8000 "
                     "--server.enableCORS=false --server.enableXsrfProtection=false", shell=True)
```

**The process must bind `0.0.0.0`, not `localhost`**, or Modal's proxy can't reach it — blank page.

### Streaming (progressive confidence updates)

```python
@app.function(gpu="any")
def compute():
    for i, s in enumerate(scores):
        yield f"data: {i} {s}\n\n".encode()

@modal.fastapi_endpoint()
def hook():
    return StreamingResponse(compute.remote_gen(), media_type="text/event-stream")
```

`text/event-stream` prevents proxy buffering. In `async def`, use `.map.aio()` — don't block the event loop.

### Auth (skip it unless you need it)

Endpoints are unlisted-but-public. If you want a gate, the fastest is a bearer token via FastAPI `Depends(HTTPBearer())` checked against a Secret. Modal also has proxy auth tokens (`Modal-Key`/`Modal-Secret` headers, configured in the dashboard) that reject before your function runs.

### Scheduled jobs

```python
@app.function(schedule=modal.Period(minutes=10))
@app.function(schedule=modal.Cron("0 6 * * *", timezone="America/New_York"))
```

`Period` resets its timer on every redeploy; `Cron` fires on wall clock. Activate with `modal deploy`. Can't pause — remove the decorator and redeploy.

---

## 6. Applied to the three tracks

### The Curation Engine

Two-pass. Pass 1 is embarrassingly parallel and cheap; pass 2 is a single CPU function doing set selection.

```python
@app.cls(gpu="L4", volumes={"/videos": s3_mount, "/feat": vol},
         timeout=1800, max_containers=100,
         retries=modal.Retries(max_retries=3, backoff_coefficient=2.0))
class Featurizer:
    @modal.enter()
    def load(self): self.model = load_encoder()

    @modal.method()
    def run(self, key: str) -> dict:
        f = self.model(f"/videos/{key}")
        np.save(f"/feat/{key}.npy", f)      # features to Volume, not return value
        return {"key": key, "score": heuristic(f)}   # small scalars only

@app.local_entrypoint()
def main():
    keys = list_keys()
    rows = [r for r in Featurizer().run.map(keys, return_exceptions=True)
            if not isinstance(r, Exception)]
    vol.commit()
    keep = select_subset(rows)      # runs locally or as a CPU function
```

Validation report = a second `.map()` over the kept vs dropped subsets computing proxy metrics, rendered in the Gradio dashboard.

### Quantitative Diversity Measurement

The pitch is "not LLM-as-judge" — so the whole thing is embedding geometry, which is exactly Modal's sweet spot.

- One `@modal.batched` embedding class over every episode → feature Volume.
- Diversity score computed on the feature matrix in a **CPU** function (`gpu=None`) — determinantal point process log-det, mean pairwise cosine distance, coverage of k-means cells, whatever. Cheap, no GPU.
- Store per-subset scores in a `modal.Dict`, render the comparison in the Gradio dashboard.
- Compute embeddings **once**, then score arbitrarily many candidate subsets for free. Don't re-run the GPU pass per subset.

### The Human Reward Model

Success/failure from video + annotations.

- `@app.cls` with a VLM or video encoder, `@modal.batched` over clips.
- Per-episode label + confidence into a `modal.Dict`; clip thumbnails/segments onto a Volume.
- Confidence-over-time viewer: `@modal.asgi_app()` serving `StaticFiles` off the Volume for the video, plus a `/scores/{episode}` route reading the Dict. Stream with `text/event-stream` if you want it to fill in live.
- Prevalence audit is just an aggregate over the Dict — a `@modal.fastapi_endpoint` returning counts.

---

## 7. Demo-day checklist

1. `modal deploy` (not `serve`) — the URL survives your laptop closing.
2. `min_containers=1` on the demo class ~5 min before you present.
3. Verify the Volume actually has data: `modal volume ls NAME`.
4. Check every write path starts with the mount prefix.
5. Have `modal volume get` fallbacks for anything you'd hate to lose to a flaky URL.
