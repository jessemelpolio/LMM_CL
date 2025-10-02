import torch
import torch.nn as nn

import os
from transformers import AutoModel, AutoImageProcessor, AutoConfig
try:
    # Some repos expose only AutoProcessor; keep as optional fallback
    from transformers import AutoProcessor  # type: ignore
except Exception:  # pragma: no cover
    AutoProcessor = None  # type: ignore
from llava.utils import rank0_print


class HFVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower.replace("hf:", "", 1)
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            self.load_model()
        else:
            # Build config-only stub for sizing (poolers/projectors) during model __init__
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name, trust_remote_code=True)
            self.config = self.cfg_only

    def load_model(self):
        # Load the model-provided image processor. Prefer AutoImageProcessor; fall back to AutoProcessor if needed.
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name, trust_remote_code=True)
        except Exception as e_img:
            if AutoProcessor is not None:
                try:
                    self.image_processor = AutoProcessor.from_pretrained(self.vision_tower_name, trust_remote_code=True)
                except Exception as e_proc:
                    raise RuntimeError(
                        f"Failed to load image processor for '{self.vision_tower_name}'. "
                        f"Tried AutoImageProcessor and AutoProcessor. Errors: {e_img} | {e_proc}"
                    )
            else:
                raise RuntimeError(
                    f"Failed to load image processor for '{self.vision_tower_name}'. "
                    f"Please ensure a recent transformers version and that the model repo provides a compatible processor. Original error: {e_img}"
                )
        rank0_print(f"Loaded image processor: {self.image_processor}")

        # Determine the target CUDA device for this process early, so any fallback paths
        # can reliably reference it.
        try:
            local_rank_env = os.environ.get("LOCAL_RANK")
            rank_env = os.environ.get("RANK", "0")
            local_rank = int(local_rank_env) if local_rank_env is not None else int(rank_env)
        except Exception:
            local_rank = 0
        try:
            current = torch.cuda.current_device()
        except Exception:
            current = 0
        device_id = local_rank if os.environ.get("LOCAL_RANK") is not None else current
        # Choose CUDA when available; otherwise stay on CPU for environments without GPUs
        if torch.cuda.is_available():
            target_device = torch.device(f"cuda:{device_id}")
        else:
            target_device = torch.device("cpu")
        # Default to fp16 on CUDA for broad compatibility; allow override via env
        env_dt = (os.environ.get("LMMS_EVAL_DTYPE") or os.environ.get("HFV_TOWER_DTYPE") or "").lower()
        if target_device.type == "cuda":
            if env_dt in ("bfloat16", "bf16"):
                target_dtype = torch.bfloat16
            elif env_dt in ("float16", "fp16", "half", ""):
                target_dtype = torch.float16
            else:
                target_dtype = torch.float16
        else:
            target_dtype = torch.float32

        # Force a materialized CPU load to avoid meta tensors (which cannot be .to()-moved).
        # Avoid device_map/low_cpu_mem_usage shortcuts that may initialize on 'meta'.
        try:
            # Ensure config does not request meta init; force CPU init for safe materialization
            cfg = AutoConfig.from_pretrained(self.vision_tower_name, trust_remote_code=True)
            if hasattr(cfg, "init_device") and str(getattr(cfg, "init_device")).lower() == "meta":
                rank0_print(f"[HFVision] Overriding init_device=meta -> cpu for {self.vision_tower_name}")
                setattr(cfg, "init_device", "cpu")
            # Optionally guard via env as well: forcefully override meta init
            prev_pid_raw = os.environ.get("PYTORCH_INIT_DEVICE")
            prev_pid = (prev_pid_raw or "").lower()
            if prev_pid and prev_pid != "cpu":
                rank0_print(f"[HFVision] Forcing PYTORCH_INIT_DEVICE=cpu (was {prev_pid}) to avoid meta init")
            os.environ["PYTORCH_INIT_DEVICE"] = "cpu"

            rank0_print(f"[HFVision] Loading vision tower on CPU (materialized), low_cpu_mem_usage=False: {self.vision_tower_name}")
            # Load fully on CPU first to ensure real tensors (avoid 'meta'), then move to CUDA.
            # Avoid passing device_map to prevent Accelerate dispatch (which can create meta tensors).
            self.vision_tower = AutoModel.from_pretrained(
                self.vision_tower_name,
                torch_dtype=torch.float32,  # load safely on CPU
                trust_remote_code=True,
                low_cpu_mem_usage=False,
                config=cfg,
            )
            # Move to the local process device (multi-GPU friendly)
            self.vision_tower = self.vision_tower.to(device=target_device, dtype=target_dtype)
        except NotImplementedError:
            # Materialize any meta tensors manually on CPU, then move to CUDA.
            rank0_print(f"[HFVision] NotImplementedError due to meta tensors; attempting manual materialization for {self.vision_tower_name}")
            vt = AutoModel.from_pretrained(
                self.vision_tower_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
                config=cfg if 'cfg' in locals() else None,
            )

            def _set_by_path(root, path, tensor, is_buffer=False):
                parts = path.split(".")
                obj = root
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                leaf = parts[-1]
                if is_buffer:
                    # Remove existing buffer (which may be meta) then register new one
                    try:
                        obj._buffers[leaf] = tensor
                    except Exception:
                        obj.register_buffer(leaf, tensor)
                else:
                    setattr(obj, leaf, torch.nn.Parameter(tensor))

            # Replace any meta params/buffers with real CPU tensors and log what we touch
            meta_param_names = [n for n, p in vt.named_parameters(recurse=True) if getattr(p, "is_meta", False)]
            meta_buffer_names = [n for n, b in vt.named_buffers(recurse=True) if getattr(b, "is_meta", False)]
            if meta_param_names or meta_buffer_names:
                head_p = ", ".join(meta_param_names[:5])
                head_b = ", ".join(meta_buffer_names[:5])
                rank0_print(f"[HFVision] Found meta tensors - params: {len(meta_param_names)} [{head_p}{' ...' if len(meta_param_names)>5 else ''}], buffers: {len(meta_buffer_names)} [{head_b}{' ...' if len(meta_buffer_names)>5 else ''}]")
            for n in meta_param_names:
                # Determine current shape via getattr chain
                try:
                    obj = vt
                    for p in n.split('.')[:-1]:
                        obj = getattr(obj, p)
                    shp = getattr(obj, n.split('.')[-1]).shape
                except Exception:
                    shp = ()
                _set_by_path(vt, n, torch.zeros(shp, dtype=torch.float32), is_buffer=False)
            for n in meta_buffer_names:
                try:
                    obj = vt
                    for p in n.split('.')[:-1]:
                        obj = getattr(obj, p)
                    shp = getattr(obj, n.split('.')[-1]).shape
                except Exception:
                    shp = ()
                _set_by_path(vt, n, torch.zeros(shp, dtype=torch.float32), is_buffer=True)

            # Manually move remaining tensors to CUDA to avoid Module._apply convert on meta
            for n, p in vt.named_parameters(recurse=True):
                if getattr(p, "is_meta", False):
                    _set_by_path(vt, n, torch.zeros(p.shape, dtype=target_dtype, device=target_device), is_buffer=False)
                else:
                    p.data = p.data.to(device=target_device, dtype=target_dtype)
            for n, b in vt.named_buffers(recurse=True):
                if getattr(b, "is_meta", False):
                    _set_by_path(vt, n, torch.zeros(b.shape, dtype=target_dtype, device=target_device), is_buffer=True)
                else:
                    # Assign back into the module at the correct buffer slot
                    parts = n.split('.')
                    obj = vt
                    for p in parts[:-1]:
                        obj = getattr(obj, p)
                    obj._buffers[parts[-1]] = b.to(device=target_device, dtype=target_dtype)

            self.vision_tower = vt

        # At this point we expect a fully materialized module on the right device. In
        # some environments a few params/buffers (e.g., DINOv3 cls/register tokens)
        # can still linger on 'meta'. Proactively fix them and ensure consistent dtype/device.
        def _materialize_and_move(module: nn.Module, device: torch.device, dtype: torch.dtype):
            meta_p = [n for n, p in module.named_parameters(recurse=True) if getattr(p, "is_meta", False)]
            meta_b = [n for n, b in module.named_buffers(recurse=True) if getattr(b, "is_meta", False)]
            if meta_p or meta_b:
                head_p = ", ".join(meta_p[:5])
                head_b = ", ".join(meta_b[:5])
                rank0_print(
                    f"[HFVision] Post-load meta tensors detected — params: {len(meta_p)} [{head_p}{' ...' if len(meta_p)>5 else ''}], "
                    f"buffers: {len(meta_b)} [{head_b}{' ...' if len(meta_b)>5 else ''}]"
                )

            # Helper identical to the fallback path for safe in-place replacement
            def _set_by_path(root, path, tensor, is_buffer=False):
                parts = path.split(".")
                obj = root
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                leaf = parts[-1]
                if is_buffer:
                    try:
                        obj._buffers[leaf] = tensor
                    except Exception:
                        obj.register_buffer(leaf, tensor)
                else:
                    setattr(obj, leaf, torch.nn.Parameter(tensor))

            # Replace any meta tensors with real tensors on the target device
            for n in meta_p:
                # Best-effort shape recovery
                shp = None
                try:
                    obj = module
                    for p in n.split('.')[:-1]:
                        obj = getattr(obj, p)
                    shp = getattr(obj, n.split('.')[-1]).shape
                except Exception:
                    pass
                shp = shp or ()
                _set_by_path(module, n, torch.zeros(shp, dtype=dtype, device=device), is_buffer=False)
            for n in meta_b:
                shp = None
                try:
                    obj = module
                    for p in n.split('.')[:-1]:
                        obj = getattr(obj, p)
                    shp = getattr(obj, n.split('.')[-1]).shape
                except Exception:
                    pass
                shp = shp or ()
                _set_by_path(module, n, torch.zeros(shp, dtype=dtype, device=device), is_buffer=True)

            # Ensure everything is on the expected device/dtype
            for n_p, p in module.named_parameters(recurse=True):
                if getattr(p, "is_meta", False):
                    # Replace lingering meta parameters
                    _set_by_path(module, n_p, torch.zeros(p.shape, dtype=dtype, device=device), is_buffer=False)
                    continue
                if p.data.device != device or p.data.dtype != dtype:
                    p.data = p.data.to(device=device, dtype=dtype)
            for n_b, b in module.named_buffers(recurse=True):
                if getattr(b, "is_meta", False):
                    # Replace lingering meta buffers
                    _set_by_path(module, n_b, torch.zeros(b.shape, dtype=dtype, device=device), is_buffer=True)
                    continue
                if b.device != device or b.dtype != dtype:
                    parts = n_b.split('.')
                    obj = module
                    for p in parts[:-1]:
                        obj = getattr(obj, p)
                    obj._buffers[parts[-1]] = b.to(device=device, dtype=dtype)

        _materialize_and_move(self.vision_tower, target_device, target_dtype)
        # Record device/dtype from a parameter (reliable for modules)
        try:
            sample_param = next(self.vision_tower.parameters())
            self._device = sample_param.device
            self._dtype = sample_param.dtype
        except StopIteration:
            # Fallbacks
            self._device = target_device if 'target_device' in locals() else torch.device('cuda:0')
            self._dtype = torch.bfloat16
        self.config = self.vision_tower.config

        if hasattr(self.vision_tower, "vision_model"):
            # Some HF models expose an inner .vision_model; ensure it is also clean.
            self.vision_tower = self.vision_tower.vision_model
            _materialize_and_move(self.vision_tower, target_device, target_dtype)
            # Refresh device/dtype in case they changed with the submodule swap
            try:
                sample_param = next(self.vision_tower.parameters())
                self._device = sample_param.device
                self._dtype = sample_param.dtype
            except StopIteration:
                self._device = target_device
                self._dtype = target_dtype

        # Restore previous PYTORCH_INIT_DEVICE to avoid affecting later loads in this process
        try:
            if prev_pid_raw is None:
                # remove if we created it
                if "PYTORCH_INIT_DEVICE" in os.environ:
                    del os.environ["PYTORCH_INIT_DEVICE"]
            else:
                os.environ["PYTORCH_INIT_DEVICE"] = prev_pid_raw
        except Exception:
            pass
        self.vision_tower.requires_grad_(False)
        # self.vision_tower.eval()
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        select_feature_type = self.select_feature

        if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
            select_every_k_layer = len(image_forward_outs.hidden_states) // 4
            image_features = torch.cat(
                [
                    image_forward_outs.hidden_states[i]
                    for i in range(
                        select_every_k_layer + self.select_layer,
                        len(image_forward_outs.hidden_states),
                        select_every_k_layer,
                    )
                ],
                dim=-1,
            )
            select_feature_type = select_feature_type.replace("slicefour_", "")
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]

        # Handle models that insert special tokens (CLS and/or register tokens),
        # including DINOv3 ViT variants. We compute offsets explicitly to keep either
        # only patches ("patch") or CLS+patches ("cls_patch") while dropping register tokens.
        try:
            expected_patches = (self.config.image_size // self.config.patch_size) ** 2
        except Exception:
            expected_patches = None

        seq_len = image_features.shape[1]
        num_register_tokens = getattr(self.config, "num_register_tokens", 0) or 0

        if expected_patches is not None:
            # Heuristic: CLS exists when total tokens equals patches + registers + 1
            cls_present = (seq_len == expected_patches + num_register_tokens + 1) or (
                seq_len > expected_patches + num_register_tokens
            )
        else:
            cls_present = False

        if select_feature_type == "patch":
            # Keep only patch tokens: drop optional CLS (if present) and register tokens
            start = 0
            if cls_present:
                start += 1
            start += num_register_tokens
            if start > 0:
                image_features = image_features[:, start:]
            # Safety fallback: if still more than expected patches, trim small surplus
            if expected_patches is not None and image_features.shape[1] > expected_patches and image_features.shape[1] - expected_patches <= 8:
                image_features = image_features[:, -(expected_patches):]
        elif select_feature_type == "cls_patch":
            # Keep CLS (if present) + patches; drop register tokens
            if num_register_tokens > 0:
                if cls_present:
                    cls_tok = image_features[:, :1]
                    patch_tok = image_features[:, 1 + num_register_tokens :]
                    image_features = torch.cat([cls_tok, patch_tok], dim=1)
                else:
                    image_features = image_features[:, num_register_tokens:]
        else:
            raise ValueError(f"Unexpected select feature: {select_feature_type}")
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        # Always resolve from current parameters to avoid stale cached dtype
        try:
            return next(self.vision_tower.parameters()).dtype
        except StopIteration:
            # Fallback to last known dtype
            return getattr(self, "_dtype", torch.float16 if torch.cuda.is_available() else torch.float32)

    @property
    def device(self):
        # Always resolve from current parameters to avoid stale cached device
        try:
            return next(self.vision_tower.parameters()).device
        except StopIteration:
            # Fallback to last known device
            return getattr(self, "_device", torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    def to(self, *args, **kwargs):
        """
        DDP-aware move. If caller passes device="cuda" with no index (as in some builders),
        map it to the local rank's device to avoid placing all towers on cuda:0.
        Also materialize any lingering meta params/buffers after the move.
        """
        # Normalize kwargs
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)

        # Allow positional variant: to(device[, dtype])
        if len(args) >= 1 and device is None:
            device = args[0]
            args = args[1:]
        if len(args) >= 1 and dtype is None and not isinstance(args[0], (str, torch.device)):
            dtype = args[0]
            args = args[1:]

        # Resolve target device
        target_device = None
        if device is not None:
            if isinstance(device, str):
                if device == "cuda":
                    # Map bare 'cuda' to local rank
                    try:
                        local_rank_env = os.environ.get("LOCAL_RANK")
                        if local_rank_env is not None:
                            idx = int(local_rank_env)
                        else:
                            idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
                    except Exception:
                        idx = 0
                    target_device = torch.device(f"cuda:{idx}") if torch.cuda.is_available() else torch.device("cpu")
                else:
                    target_device = torch.device(device)
            elif isinstance(device, torch.device):
                target_device = device
        # Resolve target dtype
        target_dtype = dtype if dtype is not None else None

        # Move the wrapped module
        if target_device is not None and target_dtype is not None:
            self.vision_tower = self.vision_tower.to(device=target_device, dtype=target_dtype)
        elif target_device is not None:
            self.vision_tower = self.vision_tower.to(device=target_device)
        elif target_dtype is not None:
            self.vision_tower = self.vision_tower.to(dtype=target_dtype)

        # After move, ensure no meta tensors remain and refresh cached fallbacks
        def _set_by_path(root, path, tensor, is_buffer=False):
            parts = path.split(".")
            obj = root
            for p in parts[:-1]:
                obj = getattr(obj, p)
            leaf = parts[-1]
            if is_buffer:
                try:
                    obj._buffers[leaf] = tensor
                except Exception:
                    obj.register_buffer(leaf, tensor)
            else:
                setattr(obj, leaf, torch.nn.Parameter(tensor))

        td = self.device
        dt = self.dtype if target_dtype is None else target_dtype
        meta_p = [n for n, p in self.vision_tower.named_parameters(recurse=True) if getattr(p, "is_meta", False)]
        meta_b = [n for n, b in self.vision_tower.named_buffers(recurse=True) if getattr(b, "is_meta", False)]
        for n in meta_p:
            # Best-effort shape recovery
            shp = None
            try:
                obj = self.vision_tower
                for p in n.split('.')[:-1]:
                    obj = getattr(obj, p)
                shp = getattr(obj, n.split('.')[-1]).shape
            except Exception:
                shp = ()
            _set_by_path(self.vision_tower, n, torch.zeros(shp, dtype=dt, device=td), is_buffer=False)
        for n in meta_b:
            shp = None
            try:
                obj = self.vision_tower
                for p in n.split('.')[:-1]:
                    obj = getattr(obj, p)
                shp = getattr(obj, n.split('.')[-1]).shape
            except Exception:
                shp = ()
            _set_by_path(self.vision_tower, n, torch.zeros(shp, dtype=dt, device=td), is_buffer=True)

        # Ensure all tensors match device/dtype after replacements
        for n, p in self.vision_tower.named_parameters(recurse=True):
            if p.data.device != td or p.data.dtype != dt:
                p.data = p.data.to(device=td, dtype=dt)
        for n, b in self.vision_tower.named_buffers(recurse=True):
            if b.device != td or b.dtype != dt:
                parts = n.split('.')
                obj = self.vision_tower
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                obj._buffers[parts[-1]] = b.to(device=td, dtype=dt)

        # Update fallbacks
        try:
            sample_param = next(self.vision_tower.parameters())
            self._device = sample_param.device
            self._dtype = sample_param.dtype
        except StopIteration:
            pass

        # Return self for chaining
        return self

    @property
    def hidden_size(self):
        try:
            _hidden_size = self.config.hidden_size
        except:
            _hidden_size = self.config.vision_config.hidden_size
        if "slicefour" in self.select_feature:
            _hidden_size *= 4
        return _hidden_size

    @property
    def num_patches(self):
        _num_patches = (self.config.image_size // self.config.patch_size) ** 2
        if "cls_patch" in self.select_feature:
            _num_patches += 1
        return _num_patches

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size
