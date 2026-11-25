"""Frozen fast-policy resources in the same repository or installed tree.

Resource validation does not import a model runtime. Inference is explicitly
CPU-only with the already validated four threads, never a provider fallback.
"""
from __future__ import annotations

import sys
import hashlib
from pathlib import Path
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location

MODEL_SHA256 = 'a8addb1a221100dd251b7846cc3ba3898ec3698c9d3b9b35768d5965472ebd5a'
UINET21_REV = '1fc50378753bd35fb239bebb77cb315c019915ff'
UINET21_SHA256 = '5895807a112ca6eec82bbe2dd252a27b140add528e1f33db841eef7c92057824'
UINET21_ROOT = Path('.local/task280-uinet21') / UINET21_REV
MASTER_SHA256 = {
    'ProduceCard': 'fd16544ed56224a3d60c498f3530216d7772d3bcb1d9493daafdfe9d8c7c75b9',
    'ProduceCardCustomize': '90bffffca0030377dadd200cf202d7a7a82c3f3b1cf0aee5d6817fbce410242b',
    'ProduceCardGrowEffect': '3365eb10168c5d1b5e620f0a2e27bfd06428bfbe7f92d2652bb26d6639f8e9cd',
    'ProduceCardSearch': '3aa067170939a8279821c443435629d303925d81758d55c246d873aa7d062350',
    'ProduceExamEffect': '8721ee6630d7379e8249ae4c456798ee54fa5bd718b61981409d83c7258c7970',
    'ProduceExamTrigger': '02b151af4058d38366a6e7f5bb176892985baf7360a763b2da75d07a4300083e',
    'ProduceExamStatusEnchant': '775c5d6ad1c2feaca1b8c4f98890a531a8335c7121e9c2ce6aa1030529038371',
    'ProduceDrink': 'efe9d056ab669e320b2ee4e1e31b015d5ad912910b0f214a0fd498b99f1d8a6a',
    'ProduceDrinkEffect': '0e9d9a2943dbbbdbedab0a675456f409d194f2cb086add66662f64cf87fd4c0f',
}


def _master_path():
    if __package__.startswith('agent.'):
        from agent.utils.runtime_paths import resolve_packaged_runtime_path as resolve_runtime_path
    else:
        from utils.runtime_paths import resolve_packaged_runtime_path as resolve_runtime_path

    return resolve_runtime_path(Path('assets/data/hif_fast/master'), Path('data/hif_fast/master'))


def _uinet21_root():
    if __package__.startswith('agent.'):
        from agent.utils.runtime_paths import resolve_packaged_runtime_path as resolve_runtime_path
    else:
        from utils.runtime_paths import resolve_packaged_runtime_path as resolve_runtime_path

    return resolve_runtime_path(UINET21_ROOT, Path('data/hif_fast/uinet21') / UINET21_REV)


def _verify_resources(expected_files):
    for file, expected in expected_files:
        if not file.is_file():
            raise ValueError('fast_runtime_frozen_resource_missing:' + file.name)
        with file.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError('fast_runtime_frozen_resource_changed:' + file.name)


def require_resources():
    if __package__.startswith('agent.'):
        from agent.utils.runtime_paths import resolve_packaged_runtime_path as resolve_runtime_path
    else:
        from utils.runtime_paths import resolve_packaged_runtime_path as resolve_runtime_path

    model = resolve_runtime_path(Path('assets/resource/base/model/hif_fast/gkms_uinet.onnx'),
        Path('resource/base/model/hif_fast/gkms_uinet.onnx'))
    master = _master_path()
    package = Path(__file__).resolve().parent / '_vendor/gkms_uinet'
    files = [package / (name + '.py') for name in ('__init__', 'customize', 'encode', 'feat', 'master', 'obs', 'policy')]
    files += [package / 'data' / name for name in ('enums.json', 'uiobs.v1.schema.json', 'vocab.v1.json')]
    files.append(package / 'LICENSE')
    if any(not file.is_file() for file in files):
        raise ValueError('fast_runtime_package_incomplete')
    _verify_resources([(model, MODEL_SHA256),
        *((master / (name + '.json'), sha) for name, sha in MASTER_SHA256.items())])
    return model, master


def _with_native_master(master_type, directory, native_rules):
    if not native_rules:
        return directory
    from .idol_road_card_rules import extend_uinet_master

    return extend_uinet_master(master_type(directory))


def load_model(*, native_rules=False):
    model, master = require_resources()
    from ._vendor.gkms_uinet import Policy
    from ._vendor.gkms_uinet.master import Master

    policy = Policy(model, _with_native_master(Master, master, native_rules), threads=4)
    policy.sess.disable_fallback()
    if policy.sess.get_providers() != ['CPUExecutionProvider']:
        raise ValueError('fast_runtime_requires_cpu_only')
    return policy


def require_uinet21_resources(model_path=None, *, model_sha256=UINET21_SHA256):
    root = _uinet21_root()
    model = root / 'model/gkms_uinet.onnx' if model_path is None else Path(model_path).resolve()
    if not model.is_file():
        raise ValueError('uinet21_local_weight_missing')
    with model.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != model_sha256:
            raise ValueError('unexpected_uinet21_weight')
    package = root / 'gkms_uinet'
    files = [package / (name + '.py') for name in (
        '__init__', 'customize', 'encode', 'encode2', 'feat', 'feat2', 'master', 'obs', 'policy')]
    files += [package / 'data' / name for name in (
        'enums.json', 'uiobs.v2.schema.json', 'vocab.v1.json', 'vocab.v2.json')]
    if any(not file.is_file() for file in files):
        raise ValueError('uinet21_local_package_incomplete')
    master = _master_path()
    _verify_resources((master / (name + '.json'), sha) for name, sha in MASTER_SHA256.items())
    return model, master


@lru_cache(maxsize=1)
def _uinet21_package():
    package = _uinet21_root() / 'gkms_uinet'
    name = '_gkh_frozen_gkms_uinet21'
    spec = spec_from_file_location(name, package / '__init__.py', submodule_search_locations=[str(package)])
    module = module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_uinet21(model_path=None, *, model_sha256=UINET21_SHA256, native_rules=False, tower_cards=False):
    from importlib import import_module

    model, master = require_uinet21_resources(model_path, model_sha256=model_sha256)
    package = _uinet21_package()
    master_type = import_module(package.__name__ + '.master').Master
    policy = package.Policy(model, _with_native_master(master_type, master, native_rules or tower_cards), threads=4)
    if policy.version != 2:
        raise ValueError('uinet21_uiobs_v2_required')
    inputs = import_module(package.__name__ + '.encode2').INPUTS
    if set(policy.inputs) != set(inputs) or not {'logit', 'adv', 'value'} <= {
        output.name for output in policy.sess.get_outputs()
    }:
        raise ValueError('uinet21_tensor_interface_mismatch')
    policy.sess.disable_fallback()
    if policy.sess.get_providers() != ['CPUExecutionProvider']:
        raise ValueError('uinet21_cpu_provider_required')
    return policy


@lru_cache(maxsize=1)
def uinet21_master():
    import json

    if __package__.startswith('agent.'):
        from agent.master_data import shared_master_rows
    else:
        from master_data import shared_master_rows

    names = ('ProduceExamEffect', 'ProduceDrink', 'ProduceDrinkEffect', 'ProduceCard', 'ProduceCardSearch',
             'ProduceExamStatusEnchant', 'ProduceExamTrigger')
    shared = shared_master_rows('gkh-model', names, digests=MASTER_SHA256)
    if shared is not None:
        return shared
    master = _master_path()
    return {name: json.loads((master / (name + '.json')).read_text(encoding='utf-8'))
            for name in names}


def battle_identity_master():
    """Read the additional scenario tables from the installed master directory."""
    import json

    root = _master_path()
    files = {name: root / f'{name}.json' for name in
             ('Character', 'IdolCard', 'ProduceStepAuditionDifficulty', 'ProduceExamBattleConfig')}
    for name, path in files.items():
        if not path.is_file():
            raise ValueError(f'battle_identity_master_missing:{name}')
    tables = {name: json.loads(path.read_text(encoding='utf-8-sig')) for name, path in files.items()}
    for name, rows in tables.items():
        if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) or not row.get('id') for row in rows):
            raise ValueError(f'battle_identity_master_invalid:{name}')
    return tables


def uinet21_observation_builder():
    """Prepare the model-side projection with the same frozen exam-effect table."""
    from functools import partial

    from .high_score.gkms_v2_observation import build_v2_observation

    table = uinet21_master()['ProduceExamEffect']
    return partial(build_v2_observation, exam_effects={row['id']: row for row in table})
