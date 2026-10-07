"""Run the pinned upstream construction statements without rewriting the network."""
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from .protocol import COMMIT, CONFIG, BACKBONE_URL, BACKBONE_SHA256


def source(root):
    root = Path(root).resolve()
    revision = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True).strip()
    if revision != COMMIT:
        raise ValueError(f'Unreviewed official commit: {revision}; expected {COMMIT}')
    subprocess.run(['git','-C',str(root),'diff','--quiet','HEAD','--'], check=True)
    if not (root/'LICENSE').is_file():
        raise FileNotFoundError(root/'LICENSE')
    return dict(url=CONFIG['official_url'], actual_commit=revision, tracked_source_modified=False,
                license='Apache-2.0; upstream LICENSE retained', root=str(root))


def construction_ast(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    def assigned(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    start = next(i for i,n in enumerate(train.body) if assigned(n,'encoder_name'))
    end = next(i for i,n in enumerate(train.body) if assigned(n,'lr_scheduler'))
    # Complete official block: backbone, ViTill, bottleneck, decoder, initialization,
    # optimizer and scheduler. Never execute upstream train/evaluate/main.
    return ast.Module(body=train.body[start:end+1], type_ignores=[])


def weight_identity(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Prepare official backbone first: {path}')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    if h.hexdigest() != BACKBONE_SHA256:
        raise ValueError('Requires original official DINOv2 base reg4 pretrained weights; SHA256 mismatch')
    return dict(path=str(path), url=BACKBONE_URL, sha256=h.hexdigest())


@contextmanager
def cwd(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def symbols(root, dataset):
    root = Path(root).resolve()
    source(root)
    # Bare-name upstream imports are allowed only in a dedicated process. Fail on
    # cached models/dataset/utils from another method instead of evicting them.
    prefixes = ('models','dataset','utils','optimizers','dinov1','dinov2','beit')
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in prefixes:
            filename = getattr(module, '__file__', None)
            namespace_paths = list(getattr(module,'__path__',()))
            belongs = (Path(filename).resolve().is_relative_to(root) if filename else
                       bool(namespace_paths) and all(Path(p).resolve().is_relative_to(root) for p in namespace_paths))
            if not belongs:
                raise RuntimeError(f'Official module name collision: {name}; use a new CLI process')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    name = '_dinomaly_official_' + dataset
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root/f'dinomaly_{dataset}_sep.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # vit_encoder creates a relative empty cache folder at import time.
        with tempfile.TemporaryDirectory(prefix='dinomaly_import_') as scratch, cwd(scratch):
            spec.loader.exec_module(module)
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in prefixes and getattr(module,'__file__',None):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise RuntimeError('Wrong official import: ' + name)
    return sys.modules['_dinomaly_official_' + dataset]


def build(module, root, backbone, device):
    import torch
    namespace = dict(vars(module), total_iters=5000, device=device)
    encoder_module = module.vit_encoder
    original = encoder_module.download_cached_file
    def local_only(url, *args, **kwargs):
        if url != BACKBONE_URL:
            raise ValueError('Unexpected official backbone request: ' + str(url))
        return str(Path(backbone).resolve())
    encoder_module.download_cached_file = local_only
    try:
        path = Path(root)/f'dinomaly_{module.__name__.rsplit("_",1)[-1]}_sep.py'
        exec(compile(construction_ast(path), str(path), 'exec'), namespace)
    finally:
        encoder_module.download_cached_file = original
    model, trainable = namespace['model'], namespace['trainable']
    # Upstream load() uses strict=False. Fail closed on a wrong/incomplete local
    # encoder checkpoint while retaining the same official loaded parameters.
    model.encoder.load_state_dict(torch.load(backbone, map_location='cpu', weights_only=False), strict=True)
    optimized = {id(p) for g in namespace['optimizer'].param_groups for p in g['params']}
    expected = {id(p) for p in trainable.parameters()}
    encoder = {id(p) for p in model.encoder.parameters()}
    if optimized != expected or optimized & encoder:
        raise ValueError('Optimizer must contain official bottleneck + decoder only')
    return model, trainable, namespace['optimizer'], namespace['lr_scheduler']


def score(model, images, module):
    import torch
    import torch.nn.functional as F
    kernel = sys.modules['utils'].get_gaussian_kernel(kernel_size=5, sigma=4).to(images.device)
    model.eval()
    with torch.no_grad():
        en, de = model(images)
        maps, _ = sys.modules['utils'].cal_anomaly_maps(en, de, images.shape[-1])
        maps = F.interpolate(maps, size=256, mode='bilinear', align_corners=False)
        maps = kernel(maps)
        flat = maps.flatten(1)
        scores = torch.sort(flat, dim=1, descending=True)[0][:, :int(flat.shape[1]*.01)].mean(1)
    return maps[:,0], scores
