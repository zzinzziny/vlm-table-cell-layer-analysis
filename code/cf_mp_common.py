import torch, os
DEVMAP = os.environ.get("CFMP_DEVICE_MAP", "cuda:0")
INSTR = "Reply with the answer only. No explanation, no reasoning, no bullet points."

def load(model_id, backend):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    if backend == "llava_ov2":
        from transformers import AutoModelForCausalLM, AutoConfig
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map=DEVMAP, trust_remote_code=True).eval()
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=DEVMAP, trust_remote_code=True).eval()
        return proc, model
    proc = AutoProcessor.from_pretrained(model_id, **({"fix_mistral_regex": True} if backend == "mistral3" else {}))
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map=DEVMAP).eval()
    return proc, model

def image_token_id(model, backend):
    c = model.config
    return getattr(c, "image_token_index", None) if backend == "mistral3" else c.image_token_id

def build(proc, model, images, question, backend, thinking_flag=True, max_soft_tokens=1120):
    if backend == "gemma":
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": f"{question}\n{INSTR}"}]
        inp = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True, return_dict=True, add_generation_prompt=True, return_tensors="pt", max_soft_tokens=max_soft_tokens)
        inp = {k: v for k, v in inp.items() if k != "num_soft_tokens_per_image"}; grids = None
    elif backend == "mistral3":
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": f"{question}\n{INSTR}"}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=images, return_tensors="pt")
        grids = [(int(h) // 28, int(w) // 28) for h, w in inp["image_sizes"].tolist()] if "image_sizes" in inp else None
    else:
        content = [{"type": "image", "image": "img"} for _ in images] + [{"type": "text", "text": f"{question}\n{INSTR}"}]
        kw = {"enable_thinking": False} if (thinking_flag and backend == "qwen") else {}
        text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, **kw)
        inp = proc(text=[text], images=images, return_tensors="pt")
        grids = [(hp // 2, wp // 2) for t, hp, wp in inp["image_grid_thw"].tolist()] if "image_grid_thw" in inp else None
    return {k: v.to(model.device) for k, v in inp.items() if hasattr(v, "to")}, grids

def page_token_positions(ids, itid, it, grids=None):
    vis = (ids == itid).nonzero().squeeze(1).tolist()
    out, off = [], 0
    for pi, p in enumerate(it["pages"]):
        gh, gw = (grids[pi] if grids is not None else p["grid_hw"])
        n = gh * gw; out.append(vis[off:off + n]); off += n
    assert off == len(vis), ("image token total", off, len(vis))
    if grids is not None:
        assert list(grids[it["b_page_pos"]]) == list(it["grid_hw"]), ("B page grid", grids[it["b_page_pos"]], it["grid_hw"])
    return out

def b_rows(ids, itid, it, grids=None):
    pos = page_token_positions(ids, itid, it, grids)[it["b_page_pos"]]
    return [pos[k] for k in it["vision_token_indices"]]
