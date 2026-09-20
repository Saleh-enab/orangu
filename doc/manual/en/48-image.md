\newpage

# Image generation

`orangu-server` draws pictures as well as text. Point it at an image
model and every chat turn in the web console — and every
`/v1/chat/completions` or `/v1/images/generations` request — is answered
with a picture instead of a reply. This chapter is the practical side:
what to download, how to start the server, and what the web console's
settings do. The *Inference server* chapter's **Image generation** section
has the reference for every key, and the *HTTP endpoints* chapter the
API; the *Inference server internals* chapter says how the pipeline is
built.

orangu serves text and images. There is no audio or video path.

![A 512 × 512 picture from the prompt "Create an image of a cat", eight
steps on a twelve-core ARM board](images/orangu-image-cat.png)

## The model, in four files

The image model is **Qwen-Image** (Alibaba's 20-billion-parameter
text-to-image transformer, the 2512 release), served from GGUF like any
language model. It needs three companions beside it, and the server
fetches all four with one command:

```sh
orangu-server download unsloth/Qwen-Image-2512-GGUF:Q4_K_M
```

- **The transformer** — `Qwen-Image-2512-Q4_K_M.gguf` from
  `unsloth/Qwen-Image-2512-GGUF`, 12.3 GiB: the model that draws.
- **The text encoder** — `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` from
  `unsloth/Qwen2.5-VL-7B-Instruct-GGUF`, 4.4 GiB: the picture is
  conditioned on this model's reading of the prompt.
- **The VAE** — `qwen_image_vae.safetensors` from
  `Comfy-Org/Qwen-Image_ComfyUI`, 243 MiB: turns the model's latents into
  pixels, and an attached picture's pixels back into latents.
- **The Lightning adapter** —
  `Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors` from
  `lightx2v/Qwen-Image-2512-Lightning`, 811 MiB: a picture in eight steps
  instead of fifty.

A companion the models directory already holds is skipped, so a second
Qwen-Image quantization fetches only itself. Any quantization of the
transformer works; `Q4_K_M` is the one measured throughout this manual.
The text encoder is an ordinary language model — `list` shows it, and it
can be served alone — and the VAE and the adapter are read as published,
since nobody converts either to GGUF.

**Memory.** Serving at `Q4_K_M` with the adapter merged in takes about
16 GB of resident memory, most of it file-backed: the transformer, the
encoder and the merged weights are mapped from disk and shared with the
page cache. A machine with 32 GB is comfortable; 16 GB is tight.

**Time.** A picture is minutes on a CPU, not seconds. On the twelve-core
ARM board this manual's numbers come from, the defaults — 1024 × 1024,
eight steps — take about 20 minutes; 512 × 512 takes 3½. A discrete GPU
is faster; an integrated one usually is not, and the server measures
before it commits: under `backend = auto` it times one transformer
linear on the device and on the CPU and keeps whichever wins, saying so
at startup (`[image] calibration …`). The wait is never a surprise:
the startup log says what a picture at the defaults costs on this
machine, and the console counts it down.

## Starting the server

The image model is served in its own role, `image`, and `--image` is the
flag that asks for it:

```sh
orangu-server --image
```

That prints the model table with every model that is not an image model
greyed, and pre-selects the first image model — Enter takes it. The
server then reports the three companions it found, the adapter it is
serving (`[image] LoRA … found under models`), and the wait:

```
[image] a picture at the defaults (1024x1024, 8 steps, guidance off) takes about 23 min here; image_size = 512x512 would be about 4 min
```

A configuration that names the model directly does the same without the
table, and `orangu-server -i` writes one — pick the image model at its
`model` prompt and the wizard asks for the picture keys, each offering
what the server does without it:

```ini
[orangu-server]
models = ~/.cache/huggingface/hub
model = unsloth/Qwen-Image-2512-GGUF:Q4_K_M

[web]
port = 8200
```

Nothing else is needed. The adapter under `models` is used because it is
there (`image_lora = auto`), and with it the steps and guidance the
adapter was made for (8, off). The keys, should you want them, are in
the *Inference server* chapter's **What a request gets**: `image_size`,
`image_steps`, `image_cfg_scale`, `image_negative_prompt`,
`image_strength`, `image_format`, and `image_lora` (`auto`, `none` for
the base model at its fifty guided steps, or another adapter's file —
the 4-step Lightning file halves the wait for a rougher picture).

The picker will not serve an image model in a language model's role
(`--code 16` is an error that says so), nor a language model under
`--image`; `bundle` refuses image models, which do not fit one file.

## In the web console

Open the console (`http://<host>:8200`, or whatever `[web].port` says).
The topbar names the model; the gear at its right is **Settings**.

### Downloading from the console

**Settings › Models** is the model manager: the same table as
`orangu-server list`, with a download box above it. Type
`unsloth/Qwen-Image-2512-GGUF:Q4_K_M` and press the download button; the
transformer and its three companions arrive together, with progress per
file. When it is there, its row's **Load** button restarts the
server on it — the console reconnects on its own — and from then on the
model name in the topbar is the image model and the pane's header says
`image` beside it.

### Picture settings

**Settings › Image** holds what every picture gets when the prompt does
not say otherwise. Each row has an **(i)** that explains it on hover.

![Settings › Image](images/orangu-image-settings.png)

| setting | choices | |
| :-- | :-- | :-- |
| **Size** | 256 × 256 up to 1280 × 720, or *Other…* for any `WIDTHxHEIGHT` in multiples of 16 | 1024 × 1024 is the model's native size and its best pictures. 512 × 512 is the smallest that still follows the prompt reliably — a preview in a couple of minutes. At 256 × 256 the model wanders: a cat asked for, a man drawn |
| **Steps** | 4, 8, 20, 50, or a number | The time is linear in them. The 8-step adapter is made for 8; the 4-step file for 4; the base model without an adapter needs 50 |
| **Guidance** | Off, 4, or a number | Classifier-free guidance pushes the picture towards the prompt and away from the negative one; it doubles the work of every step. Off under a Lightning adapter, which was trained without it; 4 is Qwen-Image's own setting for the base model |
| **Negative prompt** | text | What the picture is pushed away from — *blurry, text, extra limbs*. Read only when guidance is on |
| **Strength** | 0 – 1 | For a picture started from an attached one: how much of the schedule to run over it. 1 ignores the attachment's content and keeps only its size; 0 returns it unchanged; 0.6 keeps the composition and redraws the rest |
| **Format** | PNG, JPEG, GIF, WebP, SVG | What the picture comes back as. PNG and WebP are lossless, JPEG smaller, GIF one frame on 256 colours, SVG a document of the picture's size carrying it as PNG (there is no pixels-to-vector). A picture started from an attachment comes back in the attachment's own format |

Under the rows a line says what a picture at these settings costs on
this server — from the server's own measured rate, before anything is
sent — and updates as you change them. **Reset** puts the form back to
the configuration file's values.

The dialog's footer is shared by every pane: **Save** makes what the
panes hold the server's — these become the defaults every turn draws at,
from any browser, until the server restarts — and closes; **Cancel** (or
the ×, or Escape) drops the edits.

The usual rhythm on a slow machine: set **512 × 512** and **4 steps**,
Save, and send the prompt for a preview in about two minutes; when the
composition is right, set **1024 × 1024** and **8 steps** and send it
again for the picture. A seed is drawn per picture and shown in its
caption; the API takes a `seed` to draw the same one again at another
size.

### Drawing

Type a prompt and send it. The robot blinks beside a countdown —
*Starting · 3m 20s*, then *Step 2/8 · 2m 05s* — which is the server's
estimate, corrected at every step. The reply is the picture at the size
set, with a save control under it that downloads the full-size file and
a caption with the size, steps and seed. Pictures are kept beside the
session and come back through **History**.

**Attach** a picture (the **+** menu's *Image* item; PNG, JPEG, GIF,
WebP and SVG are read) and the model starts from it instead of from
noise, at the attachment's own proportions and running *Strength* of the
schedule — a rough sketch redrawn as a painting, or a photograph in
another style. The reply comes back in the attachment's format.

A long picture can be stopped with the console's Stop button, or by
closing the tab: the work ends within seconds, at the next transformer
block. Ctrl+C on the server ends it as quickly, whatever it was doing.

### MCP servers, while you are there

**Settings › MCP** is the same dialog's third pane, unrelated to
pictures: the MCP servers named in `orangu-server.conf` for orangu
clients to use, with Add, Edit and Delete, written to the file by the
same Save. It is described with the configuration in the *Inference
server* chapter.

## Through the API

Everything the console does, a client can:

- `POST /v1/images/generations` is OpenAI's Images API with the local
  model's knobs added (`size`, `steps`, `cfg_scale`, `negative_prompt`,
  `seed`, `output_format`, and an `image` with `strength` to start
  from). `stream: true` sends a progress event per step, the first one
  — step 0 — carrying the estimate before any work.
- `POST /v1/chat/completions` on the same server treats the last user
  message as the prompt, and its last `image_url` part as the picture to
  start from; the answer is the picture as a markdown image.
- `GET /props` reports the companions, the adapter, the defaults, the
  measured rate and what a picture costs at it; `POST /props` sets the
  defaults — it is what the Image pane's Save calls.

All three are documented field by field in the *HTTP endpoints* chapter.

## What to expect, and what not

- **The first picture after a start is slower to announce than to draw.**
  The estimate comes from a startup calibration until a picture has been
  drawn; from then on it is within a few percent.
- **The first start with an adapter merges it** — the low-rank adapter is
  folded into the transformer's weights, a couple of minutes once — and
  keeps the result under `<models>/orangu-merged/` (8.6 GiB), so every
  later start maps it in seconds. Delete the directory to reclaim the
  space; it is rebuilt on demand.
- **Guidance and the negative prompt are the base model's tools.** Under
  the Lightning adapter guidance is off and the negative prompt is not
  read; turning guidance on doubles the time and does not improve the
  adapter's pictures.
- **The adapter's steps are the adapter's.** The 8-step file at 4 steps
  draws, but softer; the 4-step file is the one made for 4 — half the
  wait of the 8-step one, for a visibly rougher picture at 1024 × 1024.
- **Small is not a preview of large.** Below 512 pixels the model drifts
  from the prompt; use 512 × 512 to check a prompt, not 256.
